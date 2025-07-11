# Copyright 2024 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported to PyTorch by Raynaldi Lalang, 2025.

from dataclasses import dataclass
import numpy as np
import torch
import torchaudio


@dataclass
class SpectrogramConfig:
    """Configuration for spectrogram extraction."""

    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 229
    fmin: float = 30.0
    fmax: float = 8000.0
    center: bool = True
    power: float = 2.0


def split_audio(audio: np.ndarray, segment_frames: int, hop_frames: int) -> np.ndarray:
    """Splits audio into overlapping segments.

    Args:
      audio: np.ndarray, shape [n_samples] or [n_channels, n_samples]
      segment_frames: Number of frames per segment.
      hop_frames: Number of frames to hop between segments.

    Returns:
      np.ndarray of segments, shape [num_segments, ...]
    """
    audio_len = audio.shape[-1]
    segments = []
    for start in range(0, audio_len - segment_frames + 1, hop_frames):
        end = start + segment_frames
        segments.append(audio[..., start:end])
    # If there's leftover audio at the end, pad it and add as last segment
    if (audio_len - segment_frames) % hop_frames != 0:
        last_segment = audio[..., -segment_frames:]
        pad_width = segment_frames - last_segment.shape[-1]
        if pad_width > 0:
            if last_segment.ndim == 1:
                last_segment = np.pad(last_segment, (0, pad_width))
            else:
                last_segment = np.pad(last_segment, ((0, 0), (0, pad_width)))
        segments.append(last_segment)
    return np.stack(segments, axis=0)


def compute_spectrogram(audio: np.ndarray, config: SpectrogramConfig) -> np.ndarray:
    """Computes a log-mel spectrogram.

    Args:
      audio: np.ndarray, shape [n_samples] or [n_channels, n_samples]
      config: SpectrogramConfig

    Returns:
      np.ndarray of log-mel spectrogram, shape [frames, n_mels]
    """
    # Ensure audio is [channels, n_samples]
    if audio.ndim == 1:
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    else:
        audio_tensor = torch.from_numpy(audio).float()

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        f_min=config.fmin,
        f_max=config.fmax,
        power=config.power,
        center=config.center,
    )

    mel_spec = mel_transform(audio_tensor)
    log_mel_spec = torch.log(mel_spec + 1e-6)
    # [n_mels, frames] -> [frames, n_mels]
    log_mel_spec = log_mel_spec.squeeze().transpose(-2, -1)
    return log_mel_spec.cpu().numpy()


def flatten_frames(spectrogram: np.ndarray) -> np.ndarray:
    """Flattens a spectrogram to (frames, features).

    Args:
      spectrogram: np.ndarray, shape [frames, n_mels] or [n_segments, frames, n_mels]

    Returns:
      np.ndarray of shape [frames, n_mels]
    """
    if spectrogram.ndim == 2:
        return spectrogram
    elif spectrogram.ndim == 3:
        n, t, f = spectrogram.shape
        return spectrogram.reshape(n * t, f)
    else:
        raise ValueError(f"Spectrogram must be 2D or 3D, got shape {spectrogram.shape}")


def input_depth(config: SpectrogramConfig) -> int:
    """Returns the number of input features (n_mels) for the spectrogram.

    Args:
      config: SpectrogramConfig

    Returns:
      Number of mel bins (int).
    """
    return config.n_mels


