import json
import re
import logging

from vlm_client import Qwen3VLWrapper

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
Ты управляешь четвероногим роботом-собакой Unitree Go2.
Твоя задача — преобразовать запрос пользователя и изображение в СТРОГИЙ JSON.
НИКОГДА не пиши текст вне JSON.
Формат ответа:
{
  "mode": "action" | "response",
  "command": "<string or null>",
  "value": { ... },
  "response": "<string or null>"
}
Правила:
1. Если пользователь просит выполнить действие:
- mode = "action"
- command = команда строго из списка:
  ["MoveForward", "MoveBackward", "MoveRight", "MoveLeft", "StopMove", "TurnLeft", "TurnRight", "StandUp", "StandDown", "Sit", "Dance1", "Dance2", "Hello"]
- value — параметры движения (если нужны)
- response = null

Пример:
{
  "mode": "action",
  "command": "Move",
  "value": 1.0,
  "response": null
}

2. Если пользователь просто общается:
- mode = "response"
- command = null
- value = {}
- response = "текст, который робот должен сказать вслух"

Пример:
{
  "mode": "response",
  "command": null,
  "value": {},
  "response": "Привет! Я готов помочь."
}

3. Используй изображение:
- если есть препятствие — НЕ выбирай движение вперед
- если пользователь спрашивает про окружение — опиши его в response

4. Если команда неясна:
- mode = "response"
- задай уточняющий вопрос

5. НИКОГДА не добавляй пояснений вне JSON.
"""


class VLMParser:
    def __init__(self, wrapper: Qwen3VLWrapper, system_prompt: str):
        self.wrapper = wrapper
        self.system_prompt = system_prompt

    def _extract_json(self, text: str):
        """
        Validates and cleans model response if needed.
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

        logger.error(f"Failed to parse JSON: {text}")
        raise ValueError("Invalid JSON from model")

    def parse(self, user_text: str, image=None):
        """
        Sends request to model.
        Returns response in JSON format.
        """
        raw = self.wrapper.chat(
            prompt=user_text,
            image=image,
            system_prompt=self.system_prompt,
            temperature=0.2,
        )

        logger.info(f"RAW MODEL OUTPUT: {raw}")

        data = self._extract_json(raw)

        if "mode" not in data:
            raise ValueError("Missing 'mode' in response")

        if data["mode"] == "action" and not data.get("command"):
            raise ValueError("Action mode without command")

        if data["mode"] == "response" and not data.get("response"):
            raise ValueError("Response mode without text")

        return data
