import os

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import torchvision.models as models
from ament_index_python.packages import get_package_share_directory


class WakeWordModel(nn.Module):
    """
    Binary wake word spotting model based on MobileNetV2.
    """
    def __init__(self):
        """
        Initialize MobileNetV2 and replace the classifier head
        for binary classification.
        """
        super().__init__()
        self.model = models.mobilenet_v2()
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, H, W]

        Returns:
            torch.Tensor: Logits of shape [B, 2]
        """
        return self.model(x)
    

class WakeWordInference:
    """
    Inference wrapper for wake word detection.

    Converts raw audio into a mel spectrogram and runs
    a trained model to detect the presence of a wake word.
    """
    def __init__(self, threshold: float = 0.5):
        """
        Initialize the inference pipeline.

        Args:
            model_path (str): Path to the trained model weights (.pt or .pth)
            threshold (float): Decision threshold for wake word detection
        """
        package_share = get_package_share_directory("go2_speech_recognition")
        model_path = os.path.join(package_share, "weights", "best_model.pth")
        
        self.model = WakeWordModel()
        state_dict = torch.load(model_path)
        state_dict = {
            f"model.{k}": v
            for k, v in state_dict.items()
        }

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.threshold = threshold

        self.sample_rate = 16000
        self.window_size = self.sample_rate * 1 # one second chunks 
        self.hop_size = self.sample_rate // 2 # 0.5 seconds overlap

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=64
        )

    def _preprocess_window(self, audio_window: np.ndarray) -> torch.Tensor:
        """
        Convert a single audio window into model input.
        Args:
            audio_window (np.ndarray):
                Mono audio segment of fixed size.
        Returns:
            torch.Tensor:
                Tensor of shape [1, 3, n_mels, time]
        """
        if audio_window.dtype != np.float32:
            audio_window = audio_window.astype(np.float32)

        waveform = torch.from_numpy(audio_window).unsqueeze(0)

        spec = self.mel(waveform)
        spec = torch.log(spec + 1e-9)

        spec = spec.repeat(3, 1, 1).unsqueeze(0)

        return spec
    
    def _prepare_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Normalize input audio format and resample if needed.

        Args:
            audio (np.ndarray):
                Input mono audio.
            sample_rate (int):
                Original sample rate.
        Returns:
            np.ndarray:
                Resampled float32 audio at target sample rate.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        waveform = torch.from_numpy(audio)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        waveform = waveform.unsqueeze(0)

        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.sample_rate
            )

        return waveform.squeeze(0).numpy()
    
    def _generate_windows(self, audio: np.ndarray):
        """
        Generate overlapping windows from audio.

        Args:
            audio (np.ndarray):
                Full audio signal.

        Yields:
            tuple:
                (window_index, audio_window)
        """
        total_length = len(audio)

        if total_length < self.window_size:
            padded = np.zeros(self.window_size, dtype=np.float32)
            padded[:total_length] = audio
            yield 0, padded
            return

        for start in range(0, total_length - self.window_size + 1, self.hop_size):
            end = start + self.window_size
            yield start, audio[start:end]

    def predict(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Predict wake word presence using sliding window inference.

        Args:
            audio (np.ndarray):
                Full mono audio signal.
            sample_rate (int):
                Input sample rate.
        Returns:
            dict:
                {
                    "has_wake_word": bool,
                    "confidence": float,
                    "window_confidences": list[float],
                    "triggered_window_start_sample": int | None
                }
        """
        audio = self._prepare_audio(audio, sample_rate)

        window_confidences = []
        max_confidence = 0.0
        triggered_window_start_sample = None

        with torch.no_grad():
            for idx, window in self._generate_windows(audio):
                spec = self._preprocess_window(window)

                logits = self.model(spec)
                probs = torch.softmax(logits, dim=1)

                confidence = probs[0, 1].item()
                window_confidences.append(confidence)

                if confidence > max_confidence:
                    max_confidence = confidence
                    triggered_window_start_sample = idx 

        return {
            "has_wake_word": max_confidence > self.threshold,
            "confidence": max_confidence,
            "window_confidences": window_confidences,
            "triggered_window_start_sample": triggered_window_start_sample
        }