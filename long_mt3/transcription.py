import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .model import MT3Model


class MT3Transcription(pl.LightningModule):
    def __init__(self, model_config, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = MT3Model(**model_config)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # pad token assumed 0

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        src, tgt = batch  # (B, T_spec), (B, T_tgt)
        logits = self(src, tgt[:, :-1])
        loss = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1)
        )
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
