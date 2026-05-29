import json
import re
import logging

from go2_vlm_core.vlm_client import Qwen3VLWrapper

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
Ты управляешь четвероногим роботом-собакой Unitree Go2 по имени Марвин.
Пример запроса пользователя: "Марвин, пройди вперед".
Преобразуй запрос пользователя и изображение в СТРОГИЙ JSON.
НИКОГДА не пиши текст вне JSON.

Формат ответа:
{
  "observation": "<коротко: что видно на кадре и где цель>",
  "mode": "action" | "response",
  "command": "<string or null>",
  "value": <число или 0.0>,
  "response": "<string or null>"
}

ВАЖНО: задачи могут требовать НЕСКОЛЬКИХ шагов. Каждый раз ты получаешь свежий
кадр и историю прошлых шагов. Выдавай ОДНУ команду за раз и жди следующего кадра,
прежде чем планировать дальше.

Поле "observation" ОБЯЗАТЕЛЬНО для action — 1-2 предложения о том, что ты видишь
сейчас (положение цели в кадре: слева/справа/по центру, далеко/близко). Это
твоя память между шагами.

Правила:
1. Если нужно выполнить действие:
- mode = "action"
- command строго из списка:
  ["MoveForward", "MoveBackward", "MoveRight", "MoveLeft", "StopMove",
   "TurnLeft", "TurnRight", "StandUp", "StandDown", "Sit", "Dance1", "Dance2", "Hello"]
- value - число: метры для движения, градусы для поворота. Дефолты:
  MoveForward/MoveBackward=1.0, MoveLeft/MoveRight=0.5, TurnLeft/TurnRight=45.0,
  остальные=0.0
- response = null
- ПРЕДПОЧИТАЙ маленькие шаги (0.3-1.0 м, 15-45°), чтобы потом скорректировать
  курс по новому кадру.

Пример (1-й шаг к цели):
{
  "observation": "Зелёный стул виден в левой части кадра, примерно 2 метра",
  "mode": "action",
  "command": "TurnLeft",
  "value": 20.0,
  "response": null
}

2. Если цель ПРОПАЛА из кадра:
- Посмотри в историю observation: где она была в прошлый раз?
- Выдай TurnLeft/TurnRight в ту сторону, где видел цель в последний раз.
- Если повернулся уже 2-3 раза подряд и так и не нашёл — сдавайся через mode=response.

3. Когда задача ВЫПОЛНЕНА (дошёл до цели / сделал что просили / нечего больше делать):
- mode = "response"
- command = null
- value = 0.0
- response = "<краткий итог для пользователя>"

Пример завершения:
{
  "observation": "Я прямо перед зелёным стулом",
  "mode": "response",
  "command": null,
  "value": 0.0,
  "response": "Я у стула."
}

4. Если пользователь просто общается (вопрос, приветствие):
- mode = "response", response = текст для озвучки.

5. Если команда неясна — mode="response", задай уточняющий вопрос.

6. НИКОГДА не добавляй пояснений вне JSON.
"""


DEFAULT_VALUES = {
    "MoveForward":  1.0,
    "MoveBackward": 1.0,
    "MoveRight":    0.5,
    "MoveLeft":     0.5,
    "StopMove":     0.0,
    "TurnLeft":     45.0,
    "TurnRight":    45.0,
    "StandUp":      0.0,
    "StandDown":    0.0,
    "Sit":          0.0,
    "Dance1":       0.0,
    "Dance2":       0.0,
    "Hello":        0.0,
}

COMMAND_LIST = list(DEFAULT_VALUES.keys())

_FALLBACK = {
    "observation": "",
    "mode": "response",
    "command": None,
    "value": 0.0,
    "response": "Не понял команду, повтори, пожалуйста",
}


class VLMParser:
    """
    Parses natural language user input into structured robot commands via VLM.

    Sends a request to the VLM, expects a strict JSON response, validates it,
    and fills in default values where needed.

    Returns a dict with fields:
        mode     - "action" or "response"
        command  - command name from COMMAND_LIST, or null
        value    - float parameter (distance in meters or angle in degrees)
        response - text for TTS, or null
    """

    def __init__(self, wrapper: Qwen3VLWrapper, system_prompt: str = SYSTEM_PROMPT, max_history_turns: int = 5):
        """
        :param wrapper: Qwen3VLWrapper instance for sending requests
        :param system_prompt: System prompt passed to the model on every request
        :param max_history_turns: Max number of (user, assistant) pairs to keep
        """
        self.wrapper = wrapper
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self._history: list[dict] = []

    def _extract_json(self, text: str):
        """
        Attempts to parse JSON from model output.
        Falls back to regex extraction if the model wrapped JSON in extra text.

        :param text: Raw model output string
        :return: Parsed dict, or None if parsing failed
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.error("Failed to parse JSON: %s", text)
        return None

    def _validate(self, data) -> bool:
        """
        Validates the structure of the parsed model response.

        :param data: Parsed dict from model output
        :return: True if valid, False otherwise
        """
        if data is None:
            logger.warning("Model returned invalid JSON")
            return False
        if "mode" not in data:
            logger.warning("Missing 'mode' in response")
            return False
        if data["mode"] == "action":
            if not data.get("command", "").strip():
                logger.warning("Action mode without command")
                return False
            if data["command"].strip() not in COMMAND_LIST:
                logger.warning("Unknown command from model: %s", data["command"])
                return False
        if data["mode"] == "response" and not data.get("response"):
            logger.warning("Response mode without text")
            return False
        return True

    def _resolve_value(self, command: str, raw_value) -> float:
        """
        Converts raw value from model response to float.
        Falls back to DEFAULT_VALUES if conversion fails.

        :param command: Command name, used to look up the default
        :param raw_value: Value from model response (may be string, number, or None)
        :return: Float value
        """
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            default = DEFAULT_VALUES.get(command, 0.0)
            logger.warning(
                "Command '%s' has no valid value (got %r), using default %s",
                command, raw_value, default,
            )
            return default

    def clear_history(self):
        self._history.clear()

    def parse(self, user_text: str, image=None):
        """
        Sends user input to the VLM and returns a structured response dict.

        :param user_text: Natural language command from the user
        :param image: Optional image from the camera
        :return: Validated response dict, or _FALLBACK on failure
        """
        user_content = [{"type": "text", "text": user_text}]
        if image:
            user_content.append(self.wrapper._encode_image(image))

        self._history.append({"role": "user", "content": user_content})

        raw = self.wrapper.chat_with_history(
            history=self._history,
            system_prompt=self.system_prompt,
            temperature=0.2,
        )
        logger.debug("RAW MODEL OUTPUT: %s", raw)

        self._history.append({"role": "assistant", "content": raw})

        max_messages = self.max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

        data = self._extract_json(raw)

        if not self._validate(data):
            return _FALLBACK

        data.setdefault("observation", "")

        if data["mode"] == "action":
            command = data["command"].strip()
            data["command"] = command
            data["value"] = self._resolve_value(command, data.get("value"))
            if not data["observation"]:
                logger.warning("Action without observation (history will be weaker)")

        return data
