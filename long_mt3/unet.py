import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import HarmonicFrequencyAttention, LocalTimeAttention


def _conv_block(c_in, c_out, p=0.1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
        nn.Dropout(p),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
    )


class UNetEncoder(nn.Module):
    """2D U-Net over (freq,time) spectrograms -> framewise embeddings."""

    def __init__(
        self,
        in_ch=1,
        base=32,
        dropout=0.1,
        d_model=512,
        use_harmonic=False,
        use_local_time=False,
    ):
        super().__init__()
        self.enc1 = _conv_block(in_ch, base, dropout)
        self.enc2 = _conv_block(base, base * 2, dropout)
        self.enc3 = _conv_block(base * 2, base * 4, dropout)
        self.pool = nn.MaxPool2d((2, 2))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(base * 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(base * 8, base * 8, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(base * 8),
            nn.GELU(),
        )
        self.use_harmonic = use_harmonic
        self.use_local_time = use_local_time
        self._hfa = None
        self._lta = LocalTimeAttention(
            d_model=base, nhead=4, max_radius=16, dropout=dropout
        )
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4, dropout)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2, dropout)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base, dropout)
        self.out_proj = nn.Linear(base, d_model)

    def forward(self, spec):  # spec: [B, f, T] or [B, 1, f, T]
        if spec.ndim == 3:
            spec = spec.unsqueeze(1)
        x1 = self.enc1(spec)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        xb = self.bottleneck(self.pool(x3))
        if self.use_harmonic:
            B, C, f, T = xb.shape
            # lazy-init with actual f
            if self._hfa is None:
                self._hfa = HarmonicFrequencyAttention(
                    freq_bins=f,
                    d_model=C,
                    nhead=4,
                    Q=12,
                    anchors=(1, 2, 3),
                    Kmax=8,
                    dropout=0.1,
                ).to(xb.device)
            # [B,C,f,T] -> [B,T,f,C]
            xb_perm = xb.permute(0, 3, 2, 1).contiguous()
            xb_h = self._hfa(xb_perm)  # [B,T,f,C]
            xb = xb_h.permute(0, 3, 2, 1).contiguous()  # back to [B,C,f,T]

        u3 = self.up3(xb)
        if u3.shape[-2:] != x3.shape[-2:]:
            u3 = F.interpolate(
                u3, size=x3.shape[-2:], mode="bilinear", align_corners=False
            )
        y3 = self.dec3(torch.cat([u3, x3], dim=1))

        u2 = self.up2(y3)
        if u2.shape[-2:] != x2.shape[-2:]:
            u2 = F.interpolate(
                u2, size=x2.shape[-2:], mode="bilinear", align_corners=False
            )
        y2 = self.dec2(torch.cat([u2, x2], dim=1))

        u1 = self.up1(y2)
        if u1.shape[-2:] != x1.shape[-2:]:
            u1 = F.interpolate(
                u1, size=x1.shape[-2:], mode="bilinear", align_corners=False
            )
        y1 = self.dec1(torch.cat([u1, x1], dim=1))

        if self.use_local_time:
            # first get [B,T,C] tokens by mean over freq
            y = y1.mean(dim=2).transpose(1, 2).contiguous()  # [B,T,C]
            y = self._lta(y)  # local time attention
        else:
            y = y1.mean(dim=2).transpose(1, 2).contiguous()  # [B,T,C]

        return self.out_proj(y)  # [B, T, d_model]
