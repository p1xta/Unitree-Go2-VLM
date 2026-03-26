import os
import base64
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dotenv import load_dotenv

import httpx

ImageSource = Union[str, os.PathLike, bytes]
load_dotenv()


class Qwen3VLWrapper:
    """
    Обёртка для Qwen3-VL через OpenWebUI API (по официальному гайду)
    Поддержка:
    - текст
    - изображения (base64)
    - outlet pipeline (опционально)
    - streaming вывода текста во время генерации
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "qwen3-vl:8b",
        timeout: int = 300,
    ):
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

    def _encode_image(self, image: ImageSource) -> Dict[str, Any]:
        if isinstance(image, (str, os.PathLike)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Изображение не найдено: {path}")
            with open(path, "rb") as f:
                img_bytes = f.read()
            ext = path.suffix.lower()
        elif isinstance(image, bytes):
            img_bytes = image
            ext = ".jpg"
        else:
            raise TypeError("image должен быть путём или bytes")

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
            "image_url": {"url": f"data:{mime_type};base64,{b64}"}
        }

    def _post(self, endpoint: str, payload: dict):
        url = f"{self.base_url}{endpoint}"

        resp = self.client.post(url, json=payload)

        if resp.status_code != 200:
            raise Exception(f"API error {resp.status_code}: {resp.text}")

        return resp.json()

    def _stream_post(self, endpoint: str, payload: dict):
        url = f"{self.base_url}{endpoint}"

        with self.client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                raise Exception(f"API error {resp.status_code}: {resp.text}")

            for line in resp.iter_lines():
                if line:
                    # иногда приходят bytes, декодируем
                    line = line.decode("utf-8") if isinstance(line, bytes) else line
                    yield line

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
        use_outlet: bool = False,
        stream: bool = False,
    ):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if stream:
            full_text = ""

            for line in self._stream_post("/api/chat/completions", payload):
                if line.startswith("data: "):
                    data = line[6:]

                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")

                        print(delta, end="", flush=True)  # вывод прямо во время генерации
                        full_text += delta

                    except Exception:
                        continue

            print()  # перенос строки после завершения
            return full_text

        else:
            data = self._post("/api/chat/completions", payload)
            answer_msg = data["choices"][0]["message"]

            if use_outlet:
                self._post("/api/chat/completed", {
                    "model": self.model,
                    "messages": messages + [answer_msg]
                })

            return answer_msg["content"]

    def chat_simple(
        self,
        prompt: str,
        image: Optional[ImageSource] = None,
        system_prompt: Optional[str] = None,
        use_outlet: bool = False,
        stream: bool = False,
    ):
        content: List[Dict] = [{"type": "text", "text": prompt}]

        if image:
            content.append(self._encode_image(image))

        messages: List[Dict] = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": content
        })

        return self.chat(messages, use_outlet=use_outlet, stream=stream)

    def chat_with_images(
        self,
        prompt: str,
        images: List[ImageSource],
        system_prompt: Optional[str] = None,
        use_outlet: bool = False,
        stream: bool = False,
    ):
        content: List[Dict] = [{"type": "text", "text": prompt}]

        for img in images:
            content.append(self._encode_image(img))

        messages: List[Dict] = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": content
        })

        return self.chat(messages, use_outlet=use_outlet, stream=stream)


if __name__ == "__main__":
    wrapper = Qwen3VLWrapper(
        base_url="http://deepcode.ci.nsu.ru",
        api_key=os.getenv("API_KEY"),
        model="Qwen3-Next-80B-A3B-Instruct"
    )

    print(wrapper.chat_simple("Привет. Как дела?", stream=True))