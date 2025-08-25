from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "VanillaCrossAttention",
    "PerceiverCrossAttention",
    "PerformerCrossAttention",
    "build_cross_attention",
    "HarmonicFrequencyAttention",
    "LocalTimeAttention",
]


def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    # FAVOR+ style positive feature map
    return F.elu(x, alpha=1.0) + 1.0  # strictly positive


class VanillaCrossAttention(nn.Module):
    """
    Wrapper around nn.MultiheadAttention for QxK->V cross-attention.

    Expects batch_first=True tensors:
      q: [B, Lq, D]
      k: [B, Lk, D]
      v: [B, Lk, D]
      key_padding_mask: [B, Lk] (True for PAD positions)
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y, _ = self.attn(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return self.ff(y)


class PerceiverCrossAttention(nn.Module):
    """
    Perceiver-style latent bottleneck:
      1) Latent queries attend to (k, v) -> latent summary
      2) Latent self-attention update
      3) Original queries attend to latents

    Shapes are batch_first:
      q: [B, Lq, D], k,v: [B, Lk, D]
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        num_latents: int = 64,
        latent_layers: int = 1,
    ):
        super().__init__()
        self.latents = nn.Parameter(
            torch.randn(num_latents, d_model) / math.sqrt(d_model)
        )
        self.cross1 = VanillaCrossAttention(d_model, nhead, dropout)  # latents <- (k,v)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.latent_encoder = nn.TransformerEncoder(enc_layer, num_layers=latent_layers)
        self.cross2 = VanillaCrossAttention(d_model, nhead, dropout)  # q <- latents

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = q.size(0)
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, Lz, D]
        lat = self.cross1(lat, k, v, key_padding_mask, attn_mask)  # [B, Lz, D]
        lat = self.latent_encoder(lat)  # [B, Lz, D]
        out = self.cross2(q, lat, lat, None, None)  # [B, Lq, D]
        return out


class PerformerCrossAttention(nn.Module):
    """
    Lightweight multi-head linear (FAVOR-like) cross-attention.
    Complexity O(B * nhead * (Lq + Lk) * d_h) vs O(Lq * Lk).

    NOTE: This is a practical implementation without Flash kernels.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-6

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, D] -> [B, nhead, L, d_head]
        B, L, _ = x.shape
        return x.view(B, L, self.nhead, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, nhead, L, d_head] -> [B, L, D]
        B, n, L, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, n * d)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Lq, _ = q.shape
        Lk = k.size(1)

        q = self._split_heads(self.q_proj(q))  # [B, H, Lq, Dh]
        k = self._split_heads(self.k_proj(k))  # [B, H, Lk, Dh]
        v = self._split_heads(self.v_proj(v))  # [B, H, Lk, Dh]

        # Key padding mask -> zero out masked K/V
        if key_padding_mask is not None:
            # key_padding_mask: [B, Lk] True means PAD
            mask = (~key_padding_mask).float().view(B, 1, Lk, 1)  # 1 for valid
            k = k * mask
            v = v * mask

        # Linear attention with positive features
        qf = _elu_feature_map(q)  # [B, H, Lq, Dh]
        kf = _elu_feature_map(k)  # [B, H, Lk, Dh]

        # Compute KV term: (K^T V) -> [B, H, Dh, Dh]
        kv = torch.einsum("bhlm,bhln->bhmn", kf, v)

        # Denominator: qf * sum_k kf
        k_sum = kf.sum(dim=2)  # [B, H, Dh]
        denom = torch.einsum("bhld,bhd->bhl", qf, k_sum)  # [B, H, Lq]
        denom = denom.unsqueeze(-1) + self.eps

        # Output: (qf @ kv) / denom
        out = torch.einsum("bhld,bhdn->bhln", qf, kv)  # [B, H, Lq, Dh]
        out = out / denom  # broadcast [B, H, Lq, 1]

        out = self._merge_heads(out)  # [B, Lq, D]
        return self.o_proj(self.dropout(out))


def build_cross_attention(
    kind: str, d_model: int, nhead: int, dropout: float
) -> nn.Module:
    name = (kind or "vanilla").lower()
    if name == "vanilla":
        return VanillaCrossAttention(d_model, nhead, dropout)
    if name == "perceiver":
        return PerceiverCrossAttention(d_model, nhead, dropout)
    if name == "performer":
        return PerformerCrossAttention(d_model, nhead, dropout)
    raise ValueError(f"Unknown cross-attention kind: {kind}")


def _harmonic_offsets(
    Q: int = 12, anchors: Tuple[int, ...] = (1, 2, 3), Kmax: int = 8
) -> Tuple[int, ...]:
    offs = {0}
    for n in anchors:
        n = max(1, int(n))
        for k in range(1, Kmax + 1):
            offs.add(int(round(Q * math.log2(k / n))))  # above f0
            offs.add(-int(round(Q * math.log2(k * n))))  # below f0 (aliases)
    return tuple(sorted(offs))


class HarmonicFrequencyAttention(nn.Module):
    """
    Self-attention across FREQUENCY bins (per frame), masked to emphasize harmonic relations.
    Input x: [B, T, F, D]  -> Output: same shape.
    """

    def __init__(
        self,
        freq_bins: int,
        d_model: int,
        nhead: int = 4,
        Q: int = 12,
        anchors: Tuple[int, ...] = (1, 2, 3),
        Kmax: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=dropout
        )
        # Build additive attention mask [F, F]
        M = torch.full((freq_bins, freq_bins), float("-inf"))
        for f0 in range(freq_bins):
            for o in _harmonic_offsets(Q, anchors, Kmax):
                j = f0 + o
                if 0 <= j < freq_bins:
                    M[f0, j] = 0.0
        self.register_buffer("mask", M)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, D = x.shape
        y = x.view(B * T, F, D)
        out, _ = self.attn(y, y, y, attn_mask=self.mask, need_weights=False)
        return out.view(B, T, F, D)


class LocalTimeAttention(nn.Module):
    """
    Local relative-time attention with learnable per-head radius (monotone-ish inductive bias).
    Input/Output: [B, T, D]
    """

    def __init__(
        self, d_model: int, nhead: int = 4, max_radius: int = 16, dropout: float = 0.1
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=dropout
        )
        self.nhead = nhead
        self.max_radius = max_radius
        self.radius = nn.Parameter(
            torch.zeros(nhead)
        )  # sigmoid -> [0, 1] -> [0, max_radius]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        idx = torch.arange(T, device=x.device)
        rel = (idx[None, :] - idx[:, None]).abs().float()  # [T, T]
        # Build per-head 3D mask: [B*nhead, T, T]
        r = (torch.sigmoid(self.radius) * self.max_radius).clamp(min=1.0)  # [H]
        head_masks = []
        for h in range(self.nhead):
            head_masks.append(
                torch.where(rel <= r[h], 0.0, float("-inf")).unsqueeze(0)
            )  # [1, T, T]
        head_mask = torch.cat(head_masks, dim=0)  # [H, T, T]
        attn_mask = head_mask.repeat(B, 1, 1)  # [B*H, T, T]
        out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        return out
