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

    def __init__(self, in_ch=1, base=32, dropout=0.1, d_model=512):
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
        self.use_harmonic = True
        self.use_local_time = True
        self._hfa = None
        self._lta = LocalTimeAttention(
            d_model=base * 8, nhead=4, max_radius=16, dropout=dropout
        )
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4, dropout)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2, dropout)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base, dropout)
        self.out_proj = nn.Linear(base, d_model)

    def forward(self, spec):  # spec: [B, F, T] or [B, 1, F, T]
        if spec.ndim == 3:
            spec = spec.unsqueeze(1)
        x1 = self.enc1(spec)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        xb = self.bottleneck(self.pool(x3))
        if self.use_harmonic:
            B, C, F, T = xb.shape
            # lazy-init with actual F
            if self._hfa is None:
                self._hfa = HarmonicFrequencyAttention(
                    freq_bins=F,
                    d_model=C,
                    nhead=4,
                    Q=12,
                    anchors=(1, 2, 3),
                    Kmax=8,
                    dropout=0.1,
                ).to(xb.device)
            # [B,C,F,T] -> [B,T,F,C]
            xb_perm = xb.permute(0, 3, 2, 1).contiguous()
            xb_h = self._hfa(xb_perm)  # [B,T,F,C]
            xb = xb_h.permute(0, 3, 2, 1).contiguous()  # back to [B,C,F,T]
        y3 = self.dec3(torch.cat([self.up3(xb), x3], dim=1))
        y2 = self.dec2(torch.cat([self.up2(y3), x2], dim=1))
        y1 = self.dec1(torch.cat([self.up1(y2), x1], dim=1))
        # collapse freq dim with conv features -> frame embedding per time step

        if self.use_local_time:
            # first get [B,T,C] tokens by mean over freq
            y = y1.mean(dim=2).transpose(1, 2).contiguous()  # [B,T,C]
            y = self._lta(y)  # local time attention
        else:
            y = y1.mean(dim=2).transpose(1, 2).contiguous()  # [B,T,C]

        return self.out_proj(y)  # [B, T, d_model]
