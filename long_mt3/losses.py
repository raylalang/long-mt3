import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameBCEHead(nn.Module):
    def __init__(self, d_model: int, num_pitches: int = 88):
        super().__init__()
        self.fc = nn.Linear(d_model, num_pitches)

    def forward(self, frame_emb: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.fc(frame_emb)  # [B, T, 88]
        if targets is None:
            return torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="mean"
        )
        return loss, torch.sigmoid(logits)


class BeatSnapHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, beat_emb: torch.Tensor, targets: torch.Tensor | None = None):
        pred = self.fc(beat_emb).squeeze(-1)  # [B, M]
        if targets is None:
            return pred
        loss = F.l1_loss(pred, targets, reduction="mean")
        return loss, pred


class OnsetBCEHead(nn.Module):
    def __init__(self, d_model: int, num_pitches: int = 88):
        super().__init__()
        self.fc = nn.Linear(d_model, num_pitches)

    def forward(self, frame_emb: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.fc(frame_emb)  # [B, T, 88]
        if targets is None:
            return torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="mean"
        )
        return loss, torch.sigmoid(logits)


class OffsetBCEHead(nn.Module):
    def __init__(self, d_model: int, num_pitches: int = 88):
        super().__init__()
        self.fc = nn.Linear(d_model, num_pitches)

    def forward(self, frame_emb: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.fc(frame_emb)  # [B, T, 88]
        if targets is None:
            return torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="mean"
        )
        return loss, torch.sigmoid(logits)


class VelocityCEHead(nn.Module):
    """
    Predict discrete velocity bins (e.g., 32 or 128).
    targets: [B, T, 88] with integer bin ids in [0, V-1], or -1 for ignore.
    """

    def __init__(self, d_model: int, num_bins: int = 32):
        super().__init__()
        self.num_bins = num_bins
        self.fc = nn.Linear(d_model, 88 * num_bins)

    def forward(self, frame_emb: torch.Tensor, targets: torch.Tensor | None = None):
        B, T, D = frame_emb.shape
        logits = self.fc(frame_emb).view(B, T, 88, self.num_bins)  # [B,T,88,V]
        if targets is None:
            probs = torch.softmax(logits, dim=-1)
            return probs
        # mask ignored
        mask = (targets >= 0).float()
        # one-hot with ignore
        oh = torch.zeros_like(logits).scatter_(
            -1, targets.clamp(min=0).unsqueeze(-1), 1.0
        )
        loss = F.cross_entropy(
            logits.view(-1, self.num_bins),
            targets.clamp(min=0).view(-1),
            reduction="none",
        ).view(B, T, 88)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        probs = torch.softmax(logits, dim=-1)
        return loss, probs


def monotonic_regular_loss(
    beat_centers: torch.Tensor,
    weight_mon: float = 0.1,
    weight_reg: float = 0.1,
    min_delta: float = 0.01,
) -> torch.Tensor:
    if beat_centers.ndim != 2:
        raise ValueError(f"beat_centers must be [B, M], got {beat_centers.shape}")
    deltas = beat_centers[:, 1:] - beat_centers[:, :-1]
    mon = torch.relu(min_delta - deltas).mean()
    reg = (deltas - deltas.mean(dim=1, keepdim=True)).abs().mean()
    return weight_mon * mon + weight_reg * reg


def sum_losses(
    loss_dict: dict[str, torch.Tensor], weights: dict[str, float]
) -> torch.Tensor:
    total = 0.0
    for k, v in loss_dict.items():
        w = float(weights.get(k, 1.0))
        total = total + w * v
    return total
