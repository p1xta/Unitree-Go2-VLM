import logging
from dataclasses import dataclass

import librosa
import numpy as np
from faster_whisper import WhisperModel


@dataclass
class STTConfig:
    model_size: str = "base"
    device: str = "auto"
    compute_type: str = "default"
    language: str = "ru"
    beam_size: int = 1
    best_of: int = 1
    target_sample_rate: int = 16000


class SpeechToText:
    """
    Lightweight Speech-to-Text wrapper for ROS-based systems.
    Optimized for direct transcription from audio arrays.
    """

    def __init__(self, config: STTConfig = STTConfig()):
        self.logger = logging.getLogger(__name__)
        self.config = config

        try:
            self.model = WhisperModel(
                model_size=config.model_size,
                device=config.device,
                compute_type=config.compute_type,
            )
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _prepare_audio(self, audio_data, sample_rate):
        """Normalize, convert to mono, and resample audio."""
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

        if sample_rate != self.config.target_sample_rate:
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=self.config.target_sample_rate,
            )

        return audio_data.astype(np.float32)

    def _process_segments(self, segments):
        """Aggregate text and estimate confidence."""
        text_parts = []
        confidence_scores = []

        for segment in segments:
            text_parts.append(segment.text)

            if hasattr(segment, "avg_logprob"):
                confidence_scores.append(float(np.exp(segment.avg_logprob)))

        text = " ".join(text_parts).strip()
        confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0

        return text, confidence

    def transcribe(self, audio_data, sample_rate=16000):
        """
        Transcribe numpy audio array into text.

        Returns:
            (text, confidence)
        """
        try:
            prepared_audio = self._prepare_audio(audio_data, sample_rate)

            segments, _ = self.model.transcribe(
                prepared_audio,
                language=(
                    self.config.language if self.config.language != "auto" else None
                ),
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
            )

            return self._process_segments(segments)

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return "", 0.0
