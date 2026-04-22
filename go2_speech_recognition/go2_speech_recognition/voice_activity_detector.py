import numpy as np
import torch
from typing import Callable


SAMPLE_RATE = 16000
WINDOW_SIZE_SAMPLES = 512


class VADetector:
    def __init__(
        self,
        on_speech: Callable[[np.ndarray], None],
        threshold: float = 0.5,
        min_silence_duration_ms: int = 600,
        speech_pad_ms: int = 200,
        max_speech_duration_s: float = 60.0,
    ):
        self.on_speech = on_speech
        self.max_speech_samples = int(max_speech_duration_s * SAMPLE_RATE)

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        _, _, _, VADIterator, _ = utils

        self._vad = VADIterator(
            model,
            threshold=threshold,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        self._history_max = int((speech_pad_ms / 1000) * SAMPLE_RATE) + WINDOW_SIZE_SAMPLES
        self._history = np.zeros(0, dtype=np.float32)
        self._history_base = 0
        self._total_samples = 0

        self._speech_chunks: list[np.ndarray] = []
        self._speech_start_idx: int | None = None
        self._in_speech = False

    def process_chunk(self, audio: np.ndarray) -> None:
        vad_result = self._vad(torch.from_numpy(audio))

        self._history = np.concatenate([self._history, audio])
        if len(self._history) > self._history_max:
            drop = len(self._history) - self._history_max
            self._history = self._history[drop:]
            self._history_base += drop
        self._total_samples += len(audio)

        if vad_result is None:
            if self._in_speech:
                self._speech_chunks.append(audio)
                buffered = self._total_samples - self._speech_start_idx
                if buffered >= self.max_speech_samples:
                    self._emit(self._total_samples)
        elif "start" in vad_result:
            if self._in_speech:
                self._emit(self._total_samples)
            start_idx = vad_result["start"]
            offset = max(0, start_idx - self._history_base)
            self._speech_chunks = [self._history[offset:].copy()]
            self._speech_start_idx = self._history_base + offset
            self._in_speech = True
        elif "end" in vad_result:
            if self._in_speech:
                self._speech_chunks.append(audio)
                self._emit(vad_result["end"])

    def flush(self) -> None:
        if self._in_speech:
            self._emit(self._total_samples)
        self._vad.reset_states()
        self._history = np.zeros(0, dtype=np.float32)
        self._history_base = 0
        self._total_samples = 0

    def _emit(self, end_idx_abs: int) -> None:
        if self._speech_chunks and self._speech_start_idx is not None:
            full = np.concatenate(self._speech_chunks)
            end_cut = end_idx_abs - self._speech_start_idx
            if end_cut > 0:
                self.on_speech(full[:end_cut])
        self._speech_chunks = []
        self._speech_start_idx = None
        self._in_speech = False
