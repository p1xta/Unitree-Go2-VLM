import json
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample
from vosk import Model, KaldiRecognizer

from ament_index_python.packages import get_package_share_directory


@dataclass
class STTConfig:
    """
    Configuration for Vosk speech recognition.
    """
    model_path: str = "models/vosk-model-small-ru-0.22"
    sample_rate: int = 16000


class SpeechToText:
    """
    Lightweight Vosk STT wrapper.
    """
    def __init__(self, config: STTConfig = STTConfig()):
        self.logger = logging.getLogger(__name__)
        self.config = config

        try:
            package_share = get_package_share_directory("go2_speech_recognition")

            model_path = Path(package_share) / config.model_path

            if not model_path.exists():
                raise FileNotFoundError(f"Vosk model not found at {model_path}")

            self.model = Model(str(model_path))
            self.recognizer = KaldiRecognizer(self.model, config.sample_rate)
            self.recognizer.SetWords(True)

        except Exception as e:
            self.logger.error(f"Failed to load Vosk model: {e}")
            raise

    def _prepare_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Normalize, convert to mono, and resample.

        Args:
            audio_data (np.ndarray): Raw audio
            sample_rate (int): Original sample rate

        Returns:
            np.ndarray: Prepared mono float32 audio
        """
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        peak = np.max(np.abs(audio_data))
        if peak > 1.0:
            audio_data = audio_data / peak

        if len(audio_data.shape) > 1:
            if audio_data.shape[0] <= 2:
                audio_data = audio_data.mean(axis=0)
            else:
                audio_data = audio_data.mean(axis=1)

        if sample_rate != self.config.sample_rate:
            target_length = int(
                len(audio_data)
                * self.config.sample_rate
                / sample_rate
            )

            audio_data = resample(
                audio_data,
                target_length,
            )

        return audio_data.astype(np.float32)

    def _reset_recognizer(self):
        """
        Reset recognizer state for new utterance.
        """
        self.recognizer = KaldiRecognizer(
            self.model,
            self.config.sample_rate,
        )
        self.recognizer.SetWords(True)

    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ):
        """
        Transcribe audio into text.

        Args:
            audio_data (np.ndarray): Input audio
            sample_rate (int): Input sample rate

        Returns:
            tuple:
                (text: str,
                 confidence: float)
        """
        try:
            prepared_audio = self._prepare_audio(audio_data, sample_rate)
            pcm_audio = (prepared_audio * 32767).astype(np.int16)

            # Reset recognizer per phrase
            self._reset_recognizer()

            self.recognizer.AcceptWaveform(pcm_audio.tobytes())

            result = json.loads(self.recognizer.FinalResult())
            text = result.get("text", "").strip()
            words = result.get("result", [],)

            if words:
                confidences = [word.get("conf", 0.0) for word in words]
                confidence = float(np.mean(confidences))
            else:
                confidence = 0.0

            return text, confidence

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return "", 0.0