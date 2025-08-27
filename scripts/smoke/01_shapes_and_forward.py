# Runs two forwards: (A) UNet+Fusion and (B) Baseline (no UNet, no fusion)
import os, sys, torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
torch.manual_seed(0)

from long_mt3.model import MT3Model


def get_vocab_size():
    try:
        from long_mt3.vocabularies import build_codec, VocabularyConfig

        vocab_config = VocabularyConfig()
        codec = build_codec(vocab_config)

        n = getattr(codec, "num_classes", None)
        return int(n if n is not None else len(codec))
    except Exception as e:
        print(f"[WARN] build_codec failed ({e}); using fallback vocab_size=512")
        return 512


# Choose a consistent feature size; many configs use 256 mel/CQT bins
B, T, f = 1, 1024, 256
input_dim = f
vocab_size = get_vocab_size()

# ---- (A) UNet + Fusion (vanilla) ----
# IMPORTANT: to satisfy Local Time Attention, align channels with its embed dim.
USE_LTA = True

src_spec = torch.randn(B, T, f)
beat_bounds = torch.tensor(
    [[[0, 200], [200, 400], [400, 650], [650, 850], [850, T]]], dtype=torch.long
)

m_unet = MT3Model(
    input_dim=input_dim,
    vocab_size=vocab_size,
    d_model=256,  # keep dims consistent with f and LTA
    nhead=4,  # 256 % 4 == 0
    dim_feedforward=512,
    num_layers=4,
    frontend={
        "type": "unet",
        "in_ch": 1,
        "base": 256 if USE_LTA else 32,  # match MHA embed dim or keep small if LTA off
        "harmonic_attention": True,
        "local_time_attention": USE_LTA,
    },
    fusion={
        "enabled": True,
        "attn_kind": "vanilla",
        "beats_per_bar": 4,
        "pos_dim": 32,
        "pool_mode": "mean",
        "num_pitches": 88,
    },
    tasks={"onset": True, "offset": True, "velocity": False},
)
outA = m_unet(src_spec, beat_bounds=beat_bounds)

print("=== (A) UNet+Fusion ===")
for k, v in outA.items():
    if torch.is_tensor(v):
        print(k, tuple(v.shape), v.dtype)
    elif isinstance(v, dict):
        print(
            k,
            {
                kk: (tuple(vv.shape) if torch.is_tensor(vv) else type(vv))
                for kk, vv in v.items()
            },
        )
    else:
        print(k, type(v))
print(
    "[A OK] finite:",
    all(torch.isfinite(t).all() for t in outA.values() if torch.is_tensor(t)),
)

# ---- (B) Baseline (no UNet, no fusion) ----
src_tokens = torch.randn(B, T, input_dim)
m_base = MT3Model(
    input_dim=input_dim,
    vocab_size=vocab_size,
    d_model=256,
    nhead=4,
    dim_feedforward=512,
    num_layers=4,
    frontend=None,
    fusion={"enabled": False, "num_pitches": 88},
    tasks={"onset": True, "offset": True},
)
outB = m_base(src_tokens)

print("\n=== (B) Baseline (no UNet, no fusion) ===")
for k, v in outB.items():
    if torch.is_tensor(v):
        print(k, tuple(v.shape), v.dtype)
print(
    "[B OK] finite:",
    all(torch.isfinite(t).all() for t in outB.values() if torch.is_tensor(t)),
)
