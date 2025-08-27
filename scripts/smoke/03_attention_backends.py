import os, sys, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from long_mt3.attention import build_cross_attention

torch.manual_seed(0)
B, Tq, Tk, D = 1, 256, 64, 128
Q = torch.randn(B, Tq, D)
K = V = torch.randn(B, Tk, D)

for kind in ["vanilla", "performer", "perceiver"]:
    attn = build_cross_attention(kind=kind, d_model=D, nhead=4, dropout=0.1)
    O = attn(Q, K, V)
    print(kind, tuple(O.shape))

print("[OK]")
