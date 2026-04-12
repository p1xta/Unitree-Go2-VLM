import json
import re
import logging

from go2_vlm_core.vlm_client import Qwen3VLWrapper

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
Ты управляешь четвероногим роботом-собакой Unitree Go2.
Твоя задача - преобразовать запрос пользователя и изображение в СТРОГИЙ JSON.
НИКОГДА не пиши текст вне JSON.
Формат ответа:
{
  "mode": "action" | "response",
  "command": "<string or null>",
  "value": <число или 0.0>,
  "response": "<string or null>"
}
Правила:
1. Если пользователь просит выполнить действие:
- mode = "action"
- command = команда строго из списка:
  ["MoveForward", "MoveBackward", "MoveRight", "MoveLeft", "StopMove", "TurnLeft", "TurnRight", "StandUp", "StandDown", "Sit", "Dance1", "Dance2", "Hello"]
- value - число: расстояние в метрах или угол в градусах. Если не указано — используй дефолт:
  MoveForward/MoveBackward = 1.0, MoveLeft/MoveRight = 0.5, TurnLeft/TurnRight = 90.0, остальные = 0.0
- response = null

Пример:
{
  "mode": "action",
  "command": "MoveForward",
  "value": 1.0,
  "response": null
}

2. Если пользователь просто общается:
- mode = "response"
- command = null
- value = 0.0
- response = "текст, который робот должен сказать вслух"

Пример:
{
  "mode": "response",
  "command": null,
  "value": 0.0,
  "response": "Привет! Я готов помочь."
}

3. Используй изображение:
- если пользователь спрашивает про окружение - опиши его в response

4. Если команда неясна:
- mode = "response"
- задай уточняющий вопрос

5. НИКОГДА не добавляй пояснений вне JSON.
"""


DEFAULT_VALUES = {
    "MoveForward":  1.0,
    "MoveBackward": 1.0,
    "MoveRight":    0.5,
    "MoveLeft":     0.5,
    "StopMove":     0.0,
    "TurnLeft":     90.0,
    "TurnRight":    90.0,
    "StandUp":      0.0,
    "StandDown":    0.0,
    "Sit":          0.0,
    "Dance1":       0.0,
    "Dance2":       0.0,
    "Hello":        0.0,
}

COMMAND_LIST = list(DEFAULT_VALUES.keys())

_FALLBACK = {
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

    def __init__(self, wrapper: Qwen3VLWrapper, system_prompt: str = SYSTEM_PROMPT):
        """
        :param wrapper: Qwen3VLWrapper instance for sending requests
        :param system_prompt: System prompt passed to the model on every request
        """
        self.wrapper = wrapper
        self.system_prompt = system_prompt

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

    def parse(self, user_text: str, image=None):
        """
        Sends user input to the VLM and returns a structured response dict.

        :param user_text: Natural language command from the user
        :param image: Optional image from the camera
        :return: Validated response dict, or _FALLBACK on failure
        """
        raw = self.wrapper.chat(
            prompt=user_text,
            image=image,
            system_prompt=self.system_prompt,
            temperature=0.2,
        )
        logger.debug("RAW MODEL OUTPUT: %s", raw)

        data = self._extract_json(raw)

        if not self._validate(data):
            return _FALLBACK

        if data["mode"] == "action":
            command = data["command"].strip()
            data["command"] = command
            data["value"] = self._resolve_value(command, data.get("value"))

        return data
