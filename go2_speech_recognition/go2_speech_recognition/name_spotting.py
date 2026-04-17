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
        state_dict = torch.load(model_path, map_location="cpu")
        state_dict = {
            f"model.{k}": v
            for k, v in state_dict.items()
        }

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.threshold = threshold

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=64
        )

    def _preprocess(self, audio: np.ndarray, sample_rate=16000):
        """
        Convert raw audio into an input tensor for model.

        Args:
            audio (np.ndarray):
                Mono audio signal.

            sample_rate (int, optional):
                Sampling rate of the input audio. Default: 16000

        Returns:
            torch.Tensor:
                Tensor of shape [1, 3, n_mels, time]
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        waveform = torch.from_numpy(audio)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, 16000
            )

        spec = self.mel(waveform)
        spec = torch.log(spec + 1e-9)
        spec = spec.repeat(3, 1, 1)
        spec = spec.unsqueeze(0)

        return spec

    def predict(self, audio: np.ndarray, sample_rate=16000):
        """
        Predict whether the wake word is present in the audio.

        Args:
            audio (np.ndarray):
                Mono audio signal. Shape: [T]

            sample_rate (int, optional):
                Sampling rate of the input audio.

        Returns:
            dict:
                {"has_wake_word": bool,
                 "confidence": float}
        """
        spec = self._preprocess(audio, sample_rate)

        with torch.no_grad():
            logits = self.model(spec)
            probs = torch.softmax(logits, dim=1)

        prob_wake = probs[0, 1].item()

        return {
            "has_wake_word": prob_wake > self.threshold,
            "confidence": prob_wake
        }