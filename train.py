import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import time
import numpy as np

import torch
import torch.nn as nn

from long_mt3.model import MT3Model
from long_mt3.data_pipeline import MT3DataPipeline
from long_mt3.vocabularies import (
    build_codec,
    VocabularyConfig,
    PAD_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    NUM_SPECIAL_TOKENS,
)
from long_mt3.contrib.mt3.spectrograms import SpectrogramConfig
from long_mt3.contrib.mt3.run_length_encoding import merge_run_length_encoded_targets

torch.backends.cudnn.benchmark = True


class MT3Trainer(pl.LightningModule):
    def __init__(self, model_config, learning_rate, codec, debug=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = MT3Model(**model_config)
        self.codec = codec
        self.debug = debug

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    def on_fit_start(self):
        self.print_output_bias()

    def print_output_bias(self):
        bias = self.model.decoder.out_proj.bias
        self.print(
            f"[DEBUG] Output layer bias stats: min={bias.min().item()}, max={bias.max().item()}, mean={bias.mean().item()}"
        )

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        return self.model(
            src,
            tgt,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

    def training_step(self, batch, batch_idx):
        (
            src,
            decoder_input_ids,
            decoder_target_ids,
            src_mask,
            input_mask,
            target_mask,
        ) = batch

        logits = self(
            src,
            decoder_input_ids,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=input_mask,
        )

        loss = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]), decoder_target_ids.reshape(-1)
        )

        pred_ids = logits.argmax(dim=-1)
        tgt_ids = decoder_target_ids

        if self.debug and batch_idx == 0:
            self.print(f"[DEBUG] logits shape: {logits.shape}")
            self.print(f"[DEBUG] decoder_target_ids shape: {tgt_ids.shape}")
            self.print(f"[DEBUG] logits argmax (first sample): {pred_ids[0]}")
            self.print(f"[DEBUG] decoder_target_ids (first sample): {tgt_ids[0]}")
            self.print(
                f"[DEBUG] logits min/max: {logits.min().item()} / {logits.max().item()}"
            )
            self.print(
                f"[DEBUG] decoder_target_ids min/max: {tgt_ids.min().item()} / {tgt_ids.max().item()}"
            )
            self.print(
                f"[DEBUG] Logits first token stats: min={logits[0,0].min().item():.4f}, max={logits[0,0].max().item():.4f}, mean={logits[0,0].mean().item():.4f}"
            )

            debug_range = slice(0, min(200, pred_ids.shape[1], tgt_ids.shape[1]))
            self.print(
                f"[DEBUG] Training Prediction[100:200]: {pred_ids[0][debug_range]}"
            )
            self.print(
                f"[DEBUG] Training Target   [100:200]: {tgt_ids[0][debug_range]}"
            )
            self.print(
                f"[DEBUG] Training Decoded Prediction[100:200]: {self.decode_event_ids(pred_ids[0][debug_range])}"
            )
            self.print(
                f"[DEBUG] Training Decoded Target   [100:200]: {self.decode_event_ids(tgt_ids[0][debug_range])}"
            )
            self.print(
                f"[DEBUG] Unique pred_ids: {torch.unique(pred_ids, return_counts=True)}"
            )
            self.print(
                f"[DEBUG] Unique tgt_ids: {torch.unique(tgt_ids, return_counts=True)}"
            )
            self.print(
                f"[DEBUG] decoder_target_ids shape: {tgt_ids.shape}, predicted shape: {pred_ids.shape}"
            )

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        (
            src,
            decoder_input_ids,
            decoder_target_ids,
            src_mask,
            input_mask,
            target_mask,
        ) = batch

        logits = self(
            src,
            decoder_input_ids,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=input_mask,
        )

        loss = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]), decoder_target_ids.reshape(-1)
        )

        # one big decoded print per epoch
        if (self.debug or batch_idx == 0) and not self.ar_decoded_this_epoch:
            self.ar_decoded_this_epoch = True

            pred_ids = logits.argmax(dim=-1)
            tgt_ids = decoder_target_ids

            debug_range = slice(0, min(200, pred_ids.shape[1], tgt_ids.shape[1]))
            self.print(f"[DEBUG] Validation Batch {batch_idx}")
            self.print(
                f"[DEBUG] Validation Prediction[100:200]: {pred_ids[0][debug_range]}"
            )
            self.print(
                f"[DEBUG] Validation Target   [100:200]: {tgt_ids[0][debug_range]}"
            )
            self.print(
                f"[DEBUG] Validation Decoded Prediction[100:200]: {self.decode_event_ids(pred_ids[0][debug_range])}"
            )
            self.print(
                f"[DEBUG] Validation Decoded Target   [100:200]: {self.decode_event_ids(tgt_ids[0][debug_range])}"
            )

            # Autoregressive decoding probe (eval-only)
            was_training = self.training
            self.eval()
            with torch.no_grad():
                ar_pred = self.autoregressive_decode(src[0:1], src_mask[0:1])
            if was_training:
                self.train()
            self.print(f"[DEBUG] Autoregressive Prediction: {ar_pred}")

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, logger=True
        )
        return loss

    def decode_event_ids(self, event_ids):
        decoded = []
        for e in event_ids:
            idx = e.item()
            if idx == 0:
                decoded.append("PAD")
            elif idx == 1:
                decoded.append("EOS")
            elif idx == 2:
                decoded.append("UNK")
            else:
                decoded.append(self.codec.decode_event_index(idx - NUM_SPECIAL_TOKENS))
        return decoded

    def on_validation_epoch_start(self):
        self.ar_decoded_this_epoch = False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.debug:
            self.print(
                "[DEBUG] Adam optimizer betas:", optimizer.defaults.get("betas", None)
            )
        return optimizer

    def autoregressive_decode(self, src, src_mask=None, max_len=2048):
        self.model.eval()
        device = src.device

        # encode once
        memory = self.model.encoder(src, src_key_padding_mask=src_mask)

        # start with EOS as BOS (your setup)
        ys = torch.full((src.size(0), 1), EOS_TOKEN, dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_mask = self.model.generate_square_subsequent_mask(
                ys.size(1), device=device
            )

            logits = self.model.decoder(
                tgt=ys,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_mask,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if torch.all(next_token == EOS_TOKEN):
                break

        merged = []
        for seq in ys[:, 1:]:  # drop initial EOS
            seq_np = (seq - NUM_SPECIAL_TOKENS).cpu().numpy()
            arr = seq_np[seq_np >= 0]
            if arr.ndim == 1:
                merged_seq = arr
            elif arr.ndim == 2:
                merged_seq = merge_run_length_encoded_targets(arr, self.codec)
            else:
                merged_seq = np.array([], dtype=seq_np.dtype)
            merged.append(torch.tensor(merged_seq, dtype=torch.long))

        return merged


class EpochTimer(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        duration = time.time() - self.epoch_start_time
        pl_module.log(
            "epoch_duration_sec", duration, prog_bar=True, sync_dist=True, on_epoch=True
        )


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    vocab_config = VocabularyConfig()
    codec = build_codec(vocab_config, event_types=cfg.data.event_types)
    spec_config = SpectrogramConfig(**cfg.data.spectrogram_config)

    batch_size = cfg.data.batch_size_per_device
    if isinstance(cfg.train.devices, list) and cfg.train.accelerator == "gpu":
        batch_size *= len(cfg.train.devices)
    datamodule = MT3DataPipeline(
        manifest_path=cfg.data.manifest_path,
        spectrogram_config=spec_config,
        codec=codec,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
        segment_seconds=cfg.data.segment_seconds,
        temperature=cfg.data.temperature,
        debug=cfg.data.debug,
        overfit_one=cfg.data.overfit_one,
    )

    model_config = {
        "input_dim": spec_config.num_mel_bins,
        "vocab_size": codec.num_classes + NUM_SPECIAL_TOKENS,
        "d_model": cfg.model.d_model,
        "nhead": cfg.model.nhead,
        "dim_feedforward": cfg.model.dim_feedforward,
        "num_layers": cfg.model.num_layers,
        "dropout": cfg.model.dropout,
    }

    model = MT3Trainer(
        model_config=model_config,
        learning_rate=cfg.train.learning_rate,
        codec=codec,
        debug=cfg.train.debug,
    )
    # model = torch.compile(model)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="epoch-{epoch:03d}",
        auto_insert_metric_name=False,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.train.early_stop_patience,
        mode="min",
        verbose=True,
    )
    timer_callback = EpochTimer()
    logger = True
    if cfg.train.tb_logger:
        logger = TensorBoardLogger(".", name="lightning_logs")

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        strategy=cfg.train.strategy,
        callbacks=[checkpoint_callback, timer_callback, early_stop_callback],
        use_distributed_sampler=False,
        logger=logger,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_model_summary=True,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.train.ckpt_path)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=".*Tempo, Key or Time signature change events found on non-zero tracks.*",
    )
    main()
