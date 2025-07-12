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

"""Audio spectrogram functions."""

import dataclasses
import numpy as np
import torch
import torchaudio

# defaults for spectrogram config
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_WIDTH = 128
DEFAULT_NUM_MEL_BINS = 512

# fixed constants; add these to SpectrogramConfig before changing
FFT_SIZE = 2048
MEL_LO_HZ = 20.0


@dataclasses.dataclass
class SpectrogramConfig:
  """Spectrogram configuration parameters."""
  sample_rate: int = DEFAULT_SAMPLE_RATE
  hop_width: int = DEFAULT_HOP_WIDTH
  num_mel_bins: int = DEFAULT_NUM_MEL_BINS

  @property
  def abbrev_str(self):
    s = ''
    if self.sample_rate != DEFAULT_SAMPLE_RATE:
      s += 'sr%d' % self.sample_rate
    if self.hop_width != DEFAULT_HOP_WIDTH:
      s += 'hw%d' % self.hop_width
    if self.num_mel_bins != DEFAULT_NUM_MEL_BINS:
      s += 'mb%d' % self.num_mel_bins
    return s

  @property
  def frames_per_second(self):
    return self.sample_rate / self.hop_width


def split_audio(audio: np.ndarray, segment_frames: int, hop_frames: int) -> np.ndarray:
    """Splits audio into overlapping segments (e.g., for inference or eval).

    Args:
        audio: np.ndarray, shape [n_samples] or [n_channels, n_samples]
        segment_frames: Number of samples per segment (not spec frames)
        hop_frames: Number of samples to hop between segments

    Returns:
        np.ndarray of shape [num_segments, segment_length]
    """
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    n_channels, n_samples = audio.shape
    segments = []

    for start in range(0, n_samples - segment_frames + 1, hop_frames):
        end = start + segment_frames
        segments.append(audio[:, start:end])

    # Add final padded segment if needed
    if (n_samples - segment_frames) % hop_frames != 0:
        last_segment = audio[:, -segment_frames:]
        pad = segment_frames - last_segment.shape[-1]
        if pad > 0:
            last_segment = np.pad(last_segment, ((0, 0), (0, pad)))
        segments.append(last_segment)

    return np.stack(segments, axis=0)


def compute_spectrogram(audio: np.ndarray, config) -> np.ndarray:
    """Computes a log-mel spectrogram matching original MT3.

    Args:
        audio: np.ndarray, shape [n_samples]
        config: SpectrogramConfig

    Returns:
        np.ndarray of shape [frames, num_mel_bins]
    """
    if audio.ndim == 1:
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    else:
        audio_tensor = torch.from_numpy(audio).float()

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=FFT_SIZE,
        hop_length=config.hop_width,
        win_length=FFT_SIZE,
        n_mels=config.num_mel_bins,
        f_min=MEL_LO_HZ,
        power=1.0,
        normalized=False,
        center=True,
        pad_mode="reflect",
    )

    mel_spec = mel_transform(audio_tensor)
    log_mel_spec = torch.log(mel_spec + 1e-6)
    return log_mel_spec.squeeze(0).transpose(0, 1).cpu().numpy()


def flatten_frames(spectrogram: np.ndarray) -> np.ndarray:
    """Flattens a batch of segments to (frames, features)."""
    if spectrogram.ndim == 2:
        return spectrogram
    elif spectrogram.ndim == 3:
        return spectrogram.reshape(-1, spectrogram.shape[-1])
    else:
        raise ValueError(f"Expected 2D or 3D spectrogram, got {spectrogram.shape}")



def input_depth(spectrogram_config):
  return spectrogram_config.num_mel_bins