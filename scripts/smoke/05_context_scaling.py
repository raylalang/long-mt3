# Runs UNet+Fusion (Perceiver) at multiple lengths to ensure no OOM/shape errors
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


USE_LTA = True  # set False if you want smaller UNet base without attention


def run_len(T):
    B, f = 1, 256
    src = torch.randn(B, T, f)
    # evenly spaced fake beats: ~4 beats per 512 frames
    nb = max(2, T // 256)
    starts = torch.linspace(0, T, nb + 1, dtype=torch.long)
    beat_bounds = torch.stack([starts[:-1], starts[1:]], dim=-1).unsqueeze(0)

    m = MT3Model(
        input_dim=f,
        vocab_size=get_vocab_size(),
        d_model=64,
        nhead=2,
        dim_feedforward=128,
        num_layers=2,
        frontend={
            "type": "unet",
            "in_ch": 1,
            "base": 256 if USE_LTA else 32,  # match LTA embed dim if enabled
            "harmonic_attention": True,
            "local_time_attention": USE_LTA,
        },
        fusion={
            "enabled": True,
            "attn_kind": "perceiver",
            "beats_per_bar": 4,
            "pos_dim": 32,
            "pool_mode": "mean",
            "num_pitches": 88,
        },
        tasks={"onset": True, "offset": True},
    )
    y = m(src, beat_bounds=beat_bounds)
    print(f"T={T}", {k: tuple(v.shape) for k, v in y.items() if hasattr(v, "shape")})


for T in [512, 1024, 2048]:
    run_len(T)
print("[OK]")
