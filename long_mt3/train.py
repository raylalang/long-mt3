import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import torch
import torch.nn as nn
import time

from .model import MT3Model
from .model_v2 import MT3ModelV2
from .data_pipeline import MT3DataPipeline
from .vocabularies import build_codec, VocabularyConfig
from .contrib.mt3.spectrograms import SpectrogramConfig

torch.backends.cudnn.benchmark = True


class MT3Trainer(pl.LightningModule):
    def __init__(self, model_config, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = MT3Model(**model_config)
        # self.model = MT3ModelV2(**model_config)
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
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class EpochTimer(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        duration = time.time() - self.epoch_start_time
        epoch = trainer.current_epoch
        pl_module.log("epoch_duration_sec", duration, prog_bar=True, sync_dist=True, on_epoch=True)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    vocab_config = VocabularyConfig()
    codec = build_codec(vocab_config)
    spec_config = SpectrogramConfig(**cfg.data.spectrogram_config)
    batch_size = cfg.data.batch_size_per_device * len(cfg.train.devices)
    data_module = MT3DataPipeline(
        manifest_path=cfg.data.manifest_path,
        spectrogram_config=spec_config,
        codec=codec,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
        segment_seconds=cfg.data.segment_seconds,
        temperature=cfg.data.temperature
    )

    model_config = {
        "input_dim": spec_config.num_mel_bins,
        "vocab_size": codec.num_classes,
        "d_model": cfg.model.d_model,
        "nhead": cfg.model.nhead,
        "dim_feedforward": cfg.model.dim_feedforward,
        "num_layers": cfg.model.num_layers,
        "dropout": cfg.model.dropout,
    }

    model = MT3Trainer(
        model_config=model_config, learning_rate=cfg.train.learning_rate
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="epoch-{epoch:03d}",
        auto_insert_metric_name=False
    )
    timer_callback = EpochTimer()
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        strategy=cfg.train.strategy,
        callbacks=[checkpoint_callback, timer_callback],
        logger=True,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        enable_model_summary=True
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=cfg.train.ckpt_path)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=".*Tempo, Key or Time signature change events found on non-zero tracks.*"
    )
    main()
