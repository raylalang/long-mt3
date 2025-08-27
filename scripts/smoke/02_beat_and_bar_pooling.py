import os, sys, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from long_mt3.fusion import BeatPooling, BarPooling

B, T, D = 1, 1024, 128
frames = torch.randn(B, T, D)

# 5 fake beats over T frames
beat_bounds = torch.tensor(
    [[[0, 200], [200, 400], [400, 650], [650, 850], [850, T]]], dtype=torch.long
)

bp = BeatPooling(d_model=D, pos_dim=32, mode="mean")
beat_tokens = bp(frames, beat_bounds)
print("beat_tokens:", tuple(beat_tokens.shape))  # (B, nb, D)

# group beats into bars (2 beats per bar here)
br = BarPooling(beats_per_bar=2, d_model=D, pos_dim=32)
bar_tokens = br(beat_tokens)
print("bar_tokens:", tuple(bar_tokens.shape))
print("[OK]")
