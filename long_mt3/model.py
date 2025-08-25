import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .fusion import CrossScaleFusion
from .losses import (
    FrameBCEHead,
    BeatSnapHead,
    OnsetBCEHead,
    OffsetBCEHead,
    VelocityCEHead,
    monotonic_regular_loss,
    sum_losses,
)
from .unet import UNetEncoder

MAX_LEN = 2048


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)].to(x.device)


class MT3Encoder(nn.Module):
    def __init__(
        self, input_dim, d_model, nhead, dim_feedforward, num_layers, dropout=0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=MAX_LEN)
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        enc = self.encoder_layers(x, src_key_padding_mask=src_key_padding_mask)
        return enc


class MT3Decoder(nn.Module):
    def __init__(
        self, vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout=0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, max_len=MAX_LEN)
        self.decoder_layers = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.out_proj = nn.Linear(d_model, vocab_size)

        # Tie decoder input embedding and output projection weights
        self.out_proj.weight = self.embed.weight
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = self.embed(tgt)
        x = self.pos_decoder(x)
        output = self.decoder_layers(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_proj(output)


class MT3Model(nn.Module):
    def __init__(
        self,
        input_dim,
        vocab_size,
        d_model=512,
        nhead=6,
        dim_feedforward=1024,
        num_layers=8,
        dropout=0.1,
        fusion=None,
        frontend=None,
        tasks=None,
    ):
        super().__init__()
        self.frontend = None
        if frontend and frontend.get("type", "").lower() == "unet":
            self.frontend = UNetEncoder(
                in_ch=frontend.get("in_ch", 1),
                base=frontend.get("base", 32),
                dropout=dropout,
                d_model=d_model,
            )
        self.encoder = MT3Encoder(
            input_dim if self.frontend is None else d_model,
            d_model,
            nhead,
            dim_feedforward,
            num_layers,
            dropout,
        )
        self.decoder = MT3Decoder(
            vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout
        )
        # heads
        self.frame_head = FrameBCEHead(
            d_model, num_pitches=fusion.get("num_pitches", 88)
        )
        self.beat_head = BeatSnapHead(d_model)
        # conditional multi-task heads
        tasks = {} if tasks is None else tasks
        self.onset_head = OnsetBCEHead(d_model) if tasks.get("onset", True) else None
        self.offset_head = OffsetBCEHead(d_model) if tasks.get("offset", True) else None
        self.vel_head = (
            VelocityCEHead(d_model, num_bins=tasks.get("velocity_bins", 32))
            if tasks.get("velocity", False)
            else None
        )

        # fusion
        if fusion and fusion.get("enabled", False):
            self.fusion = CrossScaleFusion(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                attn_kind=fusion.get("attn_kind", "vanilla"),
                beats_per_bar=fusion.get("beats_per_bar", 4),
                pos_dim=fusion.get("pos_dim", 32),
                pool_mode=fusion.get("pool_mode", "mean"),
            )
        else:
            self.fusion = None

    def forward(
        self,
        src,  # [B, T, F] spectrogram if using UNet; else [B, T, d_in]
        src_key_padding_mask=None,  # [B, T]
        beat_bounds=None,  # [B, M, 2] int indices
        targets: dict | None = None,  # optional targets dict
        tgt=None,  # [B, L] decoder input ids (optional)
        tgt_key_padding_mask=None,  # [B, L]
    ):
        # frontend -> encoder
        if getattr(self, "frontend", None) is not None:
            frame_tokens = self.frontend(
                src.transpose(1, 2)
            ).contiguous()  # [B, T, d_model]
        else:
            frame_tokens = src  # [B, T, d_model] for baseline

        memory = self.encoder(
            frame_tokens, src_key_padding_mask=src_key_padding_mask
        )  # [B, T, d_model]

        # optional fusion
        if self.fusion is not None and beat_bounds is not None:
            frame_refined, beat_refined, bar_emb = self.fusion(
                memory, beat_bounds, src_key_padding_mask
            )
        else:
            frame_refined = memory
            beat_refined = None

        logits = None
        if tgt is not None:
            tgt_mask = self.generate_square_subsequent_mask(
                tgt.size(1), device=tgt.device
            )
            logits = self.decoder(
                tgt=tgt,
                memory=frame_refined,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

        # heads -> predictions (and losses if targets)
        out = {}
        losses = {}

        # frame presence
        if targets is not None and "frame" in targets and targets["frame"] is not None:
            l, pred = self.frame_head(frame_refined, targets["frame"])
            losses["frame"] = l
        else:
            pred = self.frame_head(frame_refined, None)
        out["frame_probs"] = pred

        # onset / offset
        if self.onset_head is not None:
            if (
                targets is not None
                and "onset" in targets
                and targets["onset"] is not None
            ):
                l, p = self.onset_head(frame_refined, targets["onset"])
                losses["onset"] = l
            else:
                p = self.onset_head(frame_refined, None)
            out["onset_probs"] = p

        if self.offset_head is not None:
            if (
                targets is not None
                and "offset" in targets
                and targets["offset"] is not None
            ):
                l, p = self.offset_head(frame_refined, targets["offset"])
                losses["offset"] = l
            else:
                p = self.offset_head(frame_refined, None)
            out["offset_probs"] = p

        # velocity (optional)
        if self.vel_head is not None:
            if (
                targets is not None
                and "velocity" in targets
                and targets["velocity"] is not None
            ):
                l, p = self.vel_head(frame_refined, targets["velocity"])
                losses["velocity"] = l
            else:
                p = self.vel_head(frame_refined, None)
            out["velocity_probs"] = p  # [B,T,88,V]

        # beat head + regularizer (only when we had beat tokens)
        if beat_refined is not None:
            if (
                targets is not None
                and "beat_center" in targets
                and targets["beat_center"] is not None
            ):
                l, pred_centers = self.beat_head(beat_refined, targets["beat_center"])
                losses["beat"] = l
                # regularizer on predicted centers
                losses["beat_reg"] = monotonic_regular_loss(pred_centers)
            else:
                pred_centers = self.beat_head(beat_refined, None)
            out["beat_centers"] = pred_centers  # [B, M]

        # return a unified dict
        out["decoder_logits"] = logits
        out["loss_terms"] = losses
        return out
