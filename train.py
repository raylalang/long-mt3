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

torch.backends.cudnn.benchmark = True


class MT3Trainer(pl.LightningModule):
    def __init__(self, model_config, learning_rate, codec, debug=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = MT3Model(**model_config)
        self.codec = codec
        self.debug = debug

        # Precompute vocab-id sets per event type and a small transition table
        def _ids(event_type: str) -> set[int]:
            lo, hi = self.codec.event_type_range(event_type)
            return set(range(lo + NUM_SPECIAL_TOKENS, hi + NUM_SPECIAL_TOKENS + 1))

        self.SHIFT_IDS = _ids("shift")
        self.PITCH_IDS = _ids("pitch")
        self.VEL_IDS = _ids("velocity")
        self.PROG_IDS = _ids("program")
        self.DRUM_IDS = _ids("drum")
        try:
            self.TIE_IDS = _ids("tie")
        except Exception:
            self.TIE_IDS = set()

        self.ALL_EVENT_IDS = (
            self.SHIFT_IDS
            | self.PITCH_IDS
            | self.VEL_IDS
            | self.PROG_IDS
            | self.DRUM_IDS
        )
        self.ALLOW = {
            "program": self.VEL_IDS,
            "velocity": self.PITCH_IDS | self.DRUM_IDS,
            "pitch": self.PROG_IDS | self.VEL_IDS | self.SHIFT_IDS,  # removed TIE_IDS
            "drum": self.PROG_IDS | self.VEL_IDS | self.SHIFT_IDS,  # removed TIE_IDS
            "shift": self.PROG_IDS | self.VEL_IDS | self.PITCH_IDS | self.DRUM_IDS,
            "tie": self.PROG_IDS | self.VEL_IDS | self.PITCH_IDS | self.DRUM_IDS,
            None: self.ALL_EVENT_IDS,
        }

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
            self.print(
                f"[DEBUG] Autoregressive Prediction: {self.decode_event_ids(torch.tensor(ar_pred[0][debug_range]))}"
            )

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

    def autoregressive_decode(
        self,
        src: torch.Tensor,  # [1, S, F] spectrogram
        src_mask: torch.Tensor = None,
        max_len: int = 2048,
        prefix_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Greedy decode for a single example. Supports an optional decoder token prefix
        (e.g., the MT3 tie-section). No caching (O(T^2)) but fine for eval.
        Returns: LongTensor of shape [1, L] with decoded token ids (including EOS).
        """
        assert (
            src.dim() == 3 and src.size(0) == 1
        ), "Expect src of shape [1, S, F] for eval."
        device = src.device

        def _forward(src_tensor: torch.Tensor, tgt_in: torch.Tensor):
            out = None
            if hasattr(self, "forward"):
                try:
                    out = self.forward(src_tensor, tgt_in)
                except TypeError:
                    out = None
            if out is None and hasattr(self, "model"):
                out = self.model(src_tensor, tgt_in)
            if out is None and hasattr(self, "net"):
                out = self.net(src_tensor, tgt_in)
            # Unwrap dicts like {"logits": ...}
            if isinstance(out, dict):
                out = out.get("logits", out.get("logit", out))
            return out  # expected shape [B=1, T, V]

        ids: list[int] = list(prefix_ids or [])
        assert len(ids) > 0, "prefix_ids must be provided (tie-section) for decoding."

        with torch.inference_mode():
            for _ in range(max_len - len(ids)):
                tgt_in = torch.tensor([ids], dtype=torch.long, device=device)  # [1, T]
                logits = _forward(src, tgt_in)  # [1, T, V]
                if logits is None:
                    raise RuntimeError(
                        "Decoder forward() returned None, check model call sites."
                    )
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                # Take last step’s logits and pick argmax
                next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                ids.append(next_id)

                if next_id == EOS_TOKEN:
                    break

        # Return as a batch of size 1
        return torch.tensor([ids], dtype=torch.long, device=device)


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
    timer_callback = EpochTimer()
    callbacks = [checkpoint_callback, timer_callback]

    if cfg.train.early_stop_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=cfg.train.early_stop_patience,
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    logger = True
    if cfg.train.tb_logger:
        logger = TensorBoardLogger(".", name="lightning_logs")

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        strategy=cfg.train.strategy,
        callbacks=callbacks,
        use_distributed_sampler=False,
        logger=logger,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        log_every_n_steps=10 if not cfg.train.debug else 1,
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
