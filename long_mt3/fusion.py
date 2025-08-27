from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import build_cross_attention


__all__ = [
    "BeatPooling",
    "BarPooling",
    "GatedResidual",
    "CrossScaleFusion",
]


def _fourier_features(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """
    pos: [B, L] in [0, 1] (normalized positions)
    returns: [B, L, dim]
    """
    # half sin, half cos
    device = pos.device
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), steps=half, device=device)
    )  # [half]
    # [B,L,half]
    ang = pos[..., None] * freqs
    s = torch.sin(ang)
    c = torch.cos(ang)
    out = torch.cat([s, c], dim=-1)
    if out.size(-1) < dim:
        pad = torch.zeros(out.size(0), out.size(1), dim - out.size(-1), device=device)
        out = torch.cat([out, pad], dim=-1)
    return out


class BeatPooling(nn.Module):
    """
    Pool frame embeddings into beat embeddings using simple averaging within [start, end).
    - frame_emb: [B, T, D]
    - beat_bounds: [B, M, 2] (int indices, inclusive start, exclusive end)
    Returns: [B, M, D]
    """

    def __init__(self, d_model: int, pos_dim: int = 32, mode: str = "mean"):
        super().__init__()
        self.mode = mode
        self.pos_dim = pos_dim
        self.proj = nn.Linear(d_model + pos_dim, d_model)

    def forward(
        self, frame_emb: torch.Tensor, beat_bounds: torch.Tensor
    ) -> torch.Tensor:
        B, T, D = frame_emb.shape
        M = beat_bounds.size(1)
        device = frame_emb.device

        out = torch.zeros(B, M, D, device=device, dtype=frame_emb.dtype)
        for b in range(B):
            for m in range(M):
                s = int(beat_bounds[b, m, 0].item())
                e = int(beat_bounds[b, m, 1].item())
                s = max(0, min(s, T - 1))
                e = max(s + 1, min(e, T))
                seg = frame_emb[b, s:e]  # [L, D]
                if self.mode == "mean" or seg.size(0) == 1:
                    out[b, m] = seg.mean(dim=0)
                else:
                    # fallback mean
                    out[b, m] = seg.mean(dim=0)

        # Add a simple global beat index positional signal
        if self.pos_dim > 0:
            idx = torch.arange(M, device=device, dtype=frame_emb.dtype)[None, :].repeat(
                B, 1
            )
            pos = (idx / max(1, M - 1)).clamp(0, 1)  # [B, M] in [0,1]
            ff = _fourier_features(pos, self.pos_dim)  # [B, M, P]
            out = self.proj(torch.cat([out, ff], dim=-1))
        return out


class BarPooling(nn.Module):
    """
    Group beats into bars by averaging every K beats (last bar can be shorter).
      - beat_emb: [B, M, D]
    Returns:
      - bar_emb: [B, Mb, D]
    """

    def __init__(self, beats_per_bar: int, d_model: int, pos_dim: int = 32):
        super().__init__()
        self.beats_per_bar = beats_per_bar
        self.pos_dim = pos_dim
        self.proj = nn.Linear(d_model + pos_dim, d_model)

    def forward(self, beat_emb: torch.Tensor) -> torch.Tensor:
        B, M, D = beat_emb.shape
        K = max(1, self.beats_per_bar)
        Mb = (M + K - 1) // K  # ceil
        device = beat_emb.device

        out = torch.zeros(B, Mb, D, device=device, dtype=beat_emb.dtype)
        for b in range(B):
            for i in range(Mb):
                s = i * K
                e = min((i + 1) * K, M)
                out[b, i] = beat_emb[b, s:e].mean(dim=0)

        if self.pos_dim > 0:
            idx = torch.arange(Mb, device=device, dtype=beat_emb.dtype)[None, :].repeat(
                B, 1
            )
            pos = (idx / max(1, Mb - 1)).clamp(0, 1)  # [B, Mb]
            ff = _fourier_features(pos, self.pos_dim)  # [B, Mb, P]
            out = self.proj(torch.cat([out, ff], dim=-1))
        return out


class GatedResidual(nn.Module):
    """
    y = x + sigmoid(W [x; y]) * y
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([x, y], dim=-1))
        return x + g * y


class CrossScaleFusion(nn.Module):
    """
    Frame <-> Beat <-> Bar fusion:
      1) frame -> beat (cross-attn), residual-gated
      2) beat  -> bar  (cross-attn), residual-gated
      3) bar   -> beat (cross-attn), residual-gated
      4) beat  -> frame (cross-attn), residual-gated

    Inputs:
      - frame_emb: [B, T, D]
      - beat_bounds: [B, M, 2] int start/end indices in frame space

    Returns:
      - frame_refined: [B, T, D]
      - beat_refined:  [B, M, D]
      - bar_fused:     [B, Mb, D]
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dropout: float = 0.1,
        attn_kind: str = "vanilla",
        pos_dim: int = 32,
        beats_per_bar: int = 4,
        pool_mode: str = "mean",
    ):
        super().__init__()
        self.beat_pool = BeatPooling(d_model=d_model, pos_dim=pos_dim, mode=pool_mode)
        self.bar_pool = BarPooling(
            beats_per_bar=beats_per_bar, d_model=d_model, pos_dim=pos_dim
        )

        self.frame_to_beat = build_cross_attention(attn_kind, d_model, nhead, dropout)
        self.beat_to_bar = build_cross_attention(attn_kind, d_model, nhead, dropout)
        self.bar_to_beat = build_cross_attention(attn_kind, d_model, nhead, dropout)
        self.beat_to_frame = build_cross_attention(attn_kind, d_model, nhead, dropout)

        self.res_fb = GatedResidual(d_model)
        self.res_bb = GatedResidual(d_model)
        self.res_bf = GatedResidual(d_model)
        self.res_ff = GatedResidual(d_model)

    def forward(
        self,
        frame_emb: torch.Tensor,  # [B, T, D]
        beat_bounds: torch.Tensor,  # [B, M, 2]
        frame_pad_mask: Optional[torch.Tensor] = None,  # [B, T] True for PAD
    ):
        # Pool to beats and bars
        beat_emb = self.beat_pool(frame_emb, beat_bounds)  # [B, M, D]
        bar_emb = self.bar_pool(beat_emb)  # [B, Mb, D]

        # frame -> beat
        beat_from_frame = self.frame_to_beat(
            q=beat_emb, k=frame_emb, v=frame_emb, key_padding_mask=frame_pad_mask
        )
        beat_fused = self.res_fb(beat_emb, beat_from_frame)

        # beat -> bar
        bar_from_beat = self.beat_to_bar(
            q=bar_emb, k=beat_fused, v=beat_fused, key_padding_mask=None
        )
        bar_fused = self.res_bb(bar_emb, bar_from_beat)

        # bar -> beat
        back_to_beat = self.bar_to_beat(
            q=beat_fused, k=bar_fused, v=bar_fused, key_padding_mask=None
        )
        beat_refined = self.res_bf(beat_fused, back_to_beat)

        # beat -> frame
        frame_from_beat = self.beat_to_frame(
            q=frame_emb, k=beat_refined, v=beat_refined, key_padding_mask=None
        )
        frame_refined = self.res_ff(frame_emb, frame_from_beat)

        return frame_refined, beat_refined, bar_fused
