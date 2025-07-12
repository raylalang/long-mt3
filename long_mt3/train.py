import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn

from .model import MT3Model
from .data_pipeline import MT3DataPipeline
from .vocabularies import build_codec, VocabularyConfig
from .contrib.mt3.spectrograms import SpectrogramConfig


class MT3Trainer(pl.LightningModule):
    def __init__(self, model_config, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = MT3Model(**model_config)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # pad token assumed 0

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self(src, tgt[:, :-1])
        loss = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1)
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self(src, tgt[:, :-1])
        loss = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1)
        )
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    vocab_config = VocabularyConfig()
    codec = build_codec(vocab_config)

    spec_config = SpectrogramConfig(**cfg.data.spectrogram_config)
    data_module = MT3DataPipeline(
        manifest_path=cfg.data.manifest_path,
        spectrogram_config=spec_config,
        codec=codec,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        segment_seconds=cfg.data.segment_seconds,
        temperature=cfg.data.temperature
    )

    model_config = {
        "input_dim": spec_config.num_mel_bins,
        "vocab_size": codec.num_classes,
        "n_layers": cfg.model.n_layers,
        "n_heads": cfg.model.n_heads,
        "d_model": cfg.model.d_model,
        "dropout": cfg.model.dropout,
    }

    model = MT3Trainer(
        model_config=model_config, learning_rate=cfg.train.learning_rate
    )
    
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        strategy=cfg.train.strategy,
        callbacks=[checkpoint_callback],
        logger=True
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
