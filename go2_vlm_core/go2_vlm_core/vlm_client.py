import base64
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Qwen3VLWrapper:
    """
    A lightweight wrapper for interacting with a Qwen VLM model via OpenWebUI API.

    Features:
    - Text-based chat
    - Optional image input (base64 encoded)
    - Optional outlet pipeline triggering
    - Designed for integration with systems like ROS2 (no stdout usage)

    Notes:
    - Streaming is intentionally disabled for simplicity and compatibility.
    - Responses are returned as plain text.
    """

    def __init__(
        self,
        base_url,
        api_key,
        model="qwen3-vl:8b",
        timeout=300,
    ):
        """
        Initialize the API client.

        :param base_url: Base URL of the OpenWebUI server
        :param api_key: API key for authentication
        :param model: Model name registered in OpenWebUI
        :param timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def _encode_image(self, image):
        """
        Convert an image into base64 data URL format.

        :param image: Path to image file or raw bytes
        :return: dict formatted for OpenWebUI API
        """
        if isinstance(image, (str, os.PathLike)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

            with open(path, "rb") as f:
                img_bytes = f.read()

            ext = path.suffix.lower()

        elif isinstance(image, bytes):
            img_bytes = image
            ext = ".jpg"
        else:
            raise TypeError("image must be a file path or bytes")

        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(ext, "image/jpeg")

        b64 = base64.b64encode(img_bytes).decode()

        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
        }

    def _post(self, endpoint, payload):
        """
        Send POST request to OpenWebUI API.

        :param endpoint: API endpoint path
        :param payload: JSON payload
        :return: parsed JSON response
        """
        url = f"{self.base_url}{endpoint}"

        try:
            resp = self.client.post(url, json=payload)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise

        return resp.json()

    def chat(
        self,
        prompt,
        image=None,
        system_prompt=None,
        temperature=0.7,
        max_tokens=1024,
    ):
        content = [{"type": "text", "text": prompt}]

        if image:
            content.append(self._encode_image(image))

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": content})

        return self._chat_messages(messages, temperature, max_tokens)

    def chat_with_history(
        self,
        history,
        system_prompt=None,
        temperature=0.7,
        max_tokens=1024,
    ):
        """
        Send a chat request with full conversation history.

        :param history: List of {"role": "user"/"assistant", "content": ...} dicts
        :param system_prompt: Optional system instruction prepended before history
        :param temperature: Sampling temperature
        :param max_tokens: Max tokens in response
        :return: model response text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(history)

        return self._chat_messages(messages, temperature, max_tokens)

    def _chat_messages(self, messages, temperature, max_tokens):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        data = self._post("/api/chat/completions", payload)

        try:
            return data["choices"][0]["message"].get("content", "")
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid response format: {data}")
            raise RuntimeError("Model response parsing failed") from e


if __name__ == "__main__":
    wrapper = Qwen3VLWrapper(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        model=os.getenv("MODEL"),
    )

    response = wrapper.chat("Привет! Как дела?")
    logger.info(f"Response: {response}")
