import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import torch.nn as nn
import time

from long_mt3.model import MT3Model
from long_mt3.data_pipeline import MT3DataPipeline
from long_mt3.vocabularies import build_codec, VocabularyConfig, get_special_tokens
from long_mt3.contrib.mt3.spectrograms import SpectrogramConfig

torch.backends.cudnn.benchmark = True


class MT3Trainer(pl.LightningModule):
    def __init__(self, model_config, learning_rate, codec, ignore_program=False, debug=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = MT3Model(**model_config)
        self.codec = codec
        self.ignore_program = ignore_program
        self.debug = debug

        if self.ignore_program:
            vocab_size = codec.num_classes
            weight = torch.ones(vocab_size)
            # Get the index range of program tokens and zero out their loss contribution
            program_start, program_end = self.codec.event_type_range("program")
            weight[program_start:program_end+1] = 0.0  # Hard ignore
        
        pad_token = get_special_tokens()["pad_token"]
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=pad_token)

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self(src, tgt[:, :-1])
        loss = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1)
        )

        if self.debug and batch_idx == 0:
            pred_ids = logits.argmax(dim=-1)  # (B, T)
            tgt_ids = tgt[:, 1:]  # shift target right to match prediction alignment

            decoded_pred = [self.codec.decode_event_index(e.item()) for e in pred_ids[0][:50]]
            decoded_tgt = [self.codec.decode_event_index(e.item()) for e in tgt_ids[0][:50]]

            print(f"[DEBUG] Training Batch {batch_idx}")
            print(f"[DEBUG] Training Prediction: {pred_ids[0][:50]}")
            print(f"[DEBUG] Training Target : {tgt_ids[0][:50]}")
            print(f"[DEBUG] Training Decoded Prediction: {decoded_pred}")
            print(f"[DEBUG] Training Decoded Target : {decoded_tgt}")

        self.log("train_loss", loss)

        return loss


    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self(src, tgt[:, :-1])
        loss = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1)
        )

        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)

        if self.debug or batch_idx == 0:
            pred_ids = logits.argmax(dim=-1)
            tgt_ids = tgt[:, 1:]

            decoded_pred = [self.codec.decode_event_index(e.item()) for e in pred_ids[0][:50]]
            decoded_tgt = [self.codec.decode_event_index(e.item()) for e in tgt_ids[0][:50]]

            print(f"[DEBUG] Validation Batch {batch_idx}")
            print(f"[DEBUG] Validation Prediction: {pred_ids[0][:50]}")
            print(f"[DEBUG] Validation Target    : {tgt_ids[0][:50]}")
            print(f"[DEBUG] Validation Decoded Prediction: {decoded_pred}")
            print(f"[DEBUG] Validation Decoded Target    : {decoded_tgt}")

            # Run autoregressive decode in eval mode (no dropout)
            was_training = self.training
            self.eval()
            with torch.no_grad():
                ar_pred = self.autoregressive_decode(src[0:1])
            if was_training:
                self.train()

            print(f"[DEBUG] Autoregressive Prediction: {ar_pred}")

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def autoregressive_decode(self, src, max_len=2048):
        self.model.eval()  # just in case
        memory = self.model.encoder(src)  # (B, T, D)
        B = src.size(0)
        device = src.device

        pad_token = get_special_tokens()["pad_token"]
        eos_token = get_special_tokens()["eos_token"]

        ys = torch.full((B, 1), pad_token, dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_mask = self.model.generate_square_subsequent_mask(ys.size(1)).to(device)
            logits = self.model.decoder(
                tgt=ys,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
            )  # shape: (B, T, vocab_size)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            ys = torch.cat([ys, next_token], dim=1)

            if torch.all(next_token == eos_token):
                break

        return ys[:, 1:]  # strip initial pad token


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
    datamodule = MT3DataPipeline(
        manifest_path=cfg.data.manifest_path,
        spectrogram_config=spec_config,
        codec=codec,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
        segment_seconds=cfg.data.segment_seconds,
        temperature=cfg.data.temperature,
        ignore_program=cfg.data.ignore_program,
        debug=cfg.data.debug
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
        model_config=model_config, learning_rate=cfg.train.learning_rate,
        codec=codec, ignore_program=cfg.data.ignore_program, debug=cfg.train.debug
    )
    # model = torch.compile(model)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="epoch-{epoch:03d}",
        auto_insert_metric_name=False
    )
    timer_callback = EpochTimer()
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        strategy=cfg.train.strategy,
        callbacks=[checkpoint_callback, timer_callback, early_stop_callback],
        use_distributed_sampler=False,
        logger=not cfg.train.debug,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        enable_model_summary=True
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.train.ckpt_path)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=".*Tempo, Key or Time signature change events found on non-zero tracks.*"
    )
    main()
