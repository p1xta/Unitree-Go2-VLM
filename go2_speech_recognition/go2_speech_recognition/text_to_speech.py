import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class TTSConfig:
    model_path: Union[Path, str]
    piper_bin: str = "piper"
    lang: str = "ru"


class PiperTTS:
    """
    Piper TTS wrapper for ROS2 pipelines.
    Outputs ONLY WAV bytes for real-time audio streaming.
    """

    def __init__(self, config: TTSConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config

        self.model_path = Path(config.model_path)
        self.piper_bin = config.piper_bin

        if shutil.which(self.piper_bin) is None:
            self.logger.error(f"Piper executable not found: {self.piper_bin}")

        if not self.model_path.exists():
            self.logger.error(f"TTS model not found: {self.model_path}")

        self.logger.info("PiperTTS initialized")

    def _build_command(self, output_path: Path) -> list[str]:
        return [
            self.piper_bin,
            "--model",
            str(self.model_path),
            "--output_file",
            str(output_path),
        ]

    def synthesize_to_bytes(self, text: str) -> bytes:
        """
        Convert text → WAV bytes (for ROS2 / robot speaker pipeline).
        """

        if not text or not text.strip():
            self.logger.error("Empty text received for synthesis")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = Path(tmp.name)

        try:
            self.logger.info("Synthesizing speech")

            result = subprocess.run(
                self._build_command(temp_path),
                input=text,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                self.logger.error(
                    "Piper synthesis failed: %s",
                    result.stderr.strip() or result.stdout.strip(),
                )

            audio_bytes = temp_path.read_bytes()
            self.logger.info("Synthesis completed")

            return audio_bytes

        except Exception as e:
            self.logger.error(f"TTS error: {e}")
            raise

        finally:
            if temp_path.exists():
                temp_path.unlink()
