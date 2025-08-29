import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import time
import numpy as np
from collections import Counter

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
    def __init__(self, model_config, codec, learning_rate,
                 label_smoothing=0.0, allow_drums=False,
                 allowed_programs=None, shift_loss_weight=0.1, debug=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = MT3Model(**model_config)
        self.codec = codec
        self.debug = debug

        # Precompute vocab-id sets per event type and a small transition table
        def _ids(event_type: str) -> set[int]:
            try:
                lo, hi = self.codec.event_type_range(event_type)
            except ValueError:
                return set()
            # vocab ids are offset by NUM_SPECIAL_TOKENS
            return set(range(lo + NUM_SPECIAL_TOKENS, hi + NUM_SPECIAL_TOKENS + 1))

        def _has(event_type: str) -> bool:
            try:
                _ = self.codec.event_type_range(event_type)
                return True
            except ValueError:
                return False
        
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
        if (not self.hparams.allow_drums) and _has("drum"):
            # hard-disable DRUM ids everywhere
            self.DRUM_IDS = set()
       
        # Optional program whitelist (absolute vocab ids)
        self.allowed_program_ids = set(allowed_programs or [])
        if len(self.allowed_program_ids) > 0 and len(self.PROG_IDS) > 0:
            # keep only whitelisted program ids; forbid the rest
            self.PROG_FORBID = sorted(list(self.PROG_IDS - self.allowed_program_ids))
        else:
            self.PROG_FORBID = []

        # Build transition table
        self.ALLOW = {
            "program": self.VEL_IDS,
            "velocity": self.PITCH_IDS | self.DRUM_IDS,
            "pitch": self.PROG_IDS | self.VEL_IDS | self.SHIFT_IDS,
            "drum": self.PROG_IDS | self.VEL_IDS | self.SHIFT_IDS,
            "shift": self.PROG_IDS | self.VEL_IDS | self.PITCH_IDS | self.DRUM_IDS,
            "tie": self.PROG_IDS | self.VEL_IDS | self.PITCH_IDS | self.DRUM_IDS,
            None: self.ALL_EVENT_IDS,
        }
        if not self.hparams.allow_drums:
            # strip drums from every transition
            for k in list(self.ALLOW.keys()):
                self.ALLOW[k] = self.ALLOW[k] - self.DRUM_IDS

        min_idx, max_idx = self.codec.event_type_range("shift")
        vocab_dim = self.model.decoder.out_proj.out_features
        mask = torch.zeros(vocab_dim, dtype=torch.bool, device=self.device)
        shift_ids = torch.arange(min_idx, max_idx + 1, device=self.device) + NUM_SPECIAL_TOKENS
        mask[shift_ids] = True
        self.register_buffer("shift_id_mask", mask, persistent=False)

        self.shift_loss_weight = float(self.hparams.shift_loss_weight)

        # Precompute train-time forbidden token indices (logit mask)
        forbid_ids: list[int] = []
        if (not self.hparams.allow_drums) and len(self.DRUM_IDS) > 0:
            forbid_ids.extend(sorted(list(self.DRUM_IDS)))
        if len(self.PROG_FORBID) > 0:
            forbid_ids.extend(self.PROG_FORBID)
        self.register_buffer(
            "forbid_idx",
            torch.tensor(forbid_ids, dtype=torch.long),
            persistent=False,
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=label_smoothing)

    def on_fit_start(self):
        self.print_output_bias()

    def print_output_bias(self):
        bias = self.model.decoder.out_proj.bias
        self.print(
            f"[DEBUG] Output layer bias stats: min={bias.min().item()}, max={bias.max().item()}, mean={bias.mean().item()}"
        )

    def forward(self, batch):
        out = self.model(
            src=batch["spec"],
            src_key_padding_mask=batch["spec_mask"],
            beat_bounds=batch.get("beat_bounds"),
            targets=batch.get("targets"),
            tgt=batch.get("decoder_input_ids"),
            tgt_key_padding_mask=batch.get("in_mask"),
        )
        return out

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        aux_losses = out["loss_terms"]
        logits = out["decoder_logits"]

        loss = 0.0
        if logits is not None:
            # loss = self.loss_fn(
            #     logits.reshape(-1, logits.shape[-1]),
            #     batch["decoder_target_ids"].reshape(-1),
            # )
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets = batch["decoder_target_ids"]        # [B, T]
            ce = nn.functional.cross_entropy(
                logits_flat, targets.reshape(-1),
                ignore_index=PAD_TOKEN, reduction="none",
                label_smoothing=self.hparams.label_smoothing,
            ).view_as(targets)
            # keep tokens up to and including the first EOS per sequence
            eos_id = EOS_TOKEN
            cumsums = (targets == eos_id).cumsum(dim=1)
            keep = (cumsums <= 1) & (targets != PAD_TOKEN)
            if self.forbid_idx.numel() > 0:
                logits_flat.index_fill_(1, self.forbid_idx, float("-inf"))
            weights = torch.ones_like(ce)
            weights = torch.where(self.shift_id_mask[targets], self.shift_loss_weight * weights, weights)
            loss = ( (ce * weights)[keep] ).sum() / (weights[keep].sum().clamp_min(1))

        loss_weights = {
            "frame": 1.0,
            "onset": 1.0,
            "offset": 1.0,
            "velocity": 0.5,
            "beat": 0.2,
            "beat_reg": 0.1,
        }
        for name, l in aux_losses.items():
            w = loss_weights.get(name, 1.0)
            loss = loss + w * l
            self.log(f"train_{name}", l, on_epoch=True, prog_bar=False, sync_dist=True)

        pred_ids = logits.argmax(dim=-1) if logits is not None else None
        tgt_ids = batch["decoder_target_ids"]

        if self.debug and batch_idx == 0 and logits is not None:
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
                f"[DEBUG] Logits first token stats: min={logits[0,0].min().item():.4f}, max={logits[0,0].max().item():.4f}, mean={logits[0,0].mean().item():.4f}",
            )
            self.print(
                f"[DEBUG] Most common decoder_target_ids: {Counter(tgt_ids).most_common(20)}"
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
        out = self.forward(batch)
        aux_losses = out["loss_terms"]
        logits = out["decoder_logits"]

        loss = 0.0
        if logits is not None:
            # loss = self.loss_fn(
            #     logits.reshape(-1, logits.shape[-1]),
            #     batch["decoder_target_ids"].reshape(-1),
            # )
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets = batch["decoder_target_ids"]        # [B, T]
            ce = nn.functional.cross_entropy(
                logits_flat, targets.reshape(-1),
                ignore_index=PAD_TOKEN, reduction="none",
                label_smoothing=self.hparams.label_smoothing,
            ).view_as(targets)
            # keep tokens up to and including the first EOS per sequence
            eos_id = EOS_TOKEN
            cumsums = (targets == eos_id).cumsum(dim=1)
            keep = (cumsums <= 1) & (targets != PAD_TOKEN)
            if self.forbid_idx.numel() > 0:
                logits_flat.index_fill_(1, self.forbid_idx, float("-inf"))
            weights = torch.ones_like(ce)
            weights = torch.where(self.shift_id_mask[targets], self.shift_loss_weight * weights, weights)
            loss = ( (ce * weights)[keep] ).sum() / (weights[keep].sum().clamp_min(1))
        
        for name, l in aux_losses.items():
            loss = loss + l
            self.log(f"val_{name}", l, on_epoch=True, prog_bar=False, sync_dist=True)

        if (self.debug or batch_idx == 0) and not self.ar_decoded_this_epoch:
            self.ar_decoded_this_epoch = True

            pred_ids = logits.argmax(dim=-1)
            tgt_ids = batch["decoder_target_ids"]

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

            # Autoregressive decoding probe (eval-only) — run only if prefix exists
            prefix = batch.get("decoder_input_ids")
            if prefix is not None and prefix.size(0) > 0 and prefix.size(1) > 0:
                full = prefix[0].tolist()
                prefix_ids = full[: max(1, min(6, len(full)))]
                was_training = self.training
                self.eval()
                with torch.no_grad():
                    ar_pred = self.autoregressive_decode(
                        batch["spec"][0:1],
                        (
                            batch.get("spec_mask", None)[0:1]
                            if batch.get("spec_mask") is not None
                            else None
                        ),
                        prefix_ids=prefix_ids,
                    )
                if was_training:
                    self.train()
                self.print(
                    f"[DEBUG] Autoregressive Prediction: {self.decode_event_ids(torch.tensor(ar_pred[0][debug_range]))}"
                )
            else:
                self.print("[DEBUG] Skipping AR probe: no decoder prefix in batch.")

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
        src: torch.Tensor,  # [1, S, f] spectrogram
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
        ), "Expect src of shape [1, S, f] for eval."
        device = src.device

        def _forward(src_tensor, tgt_in):
            B, S, f = src_tensor.shape
            # Build uniform beat grid as fallback
            M = 32
            step = max(1, S // M)
            starts = torch.arange(0, S, step, device=src_tensor.device)
            ends = torch.clamp(starts + step, max=S)
            beat_bounds = torch.stack([starts, ends], dim=-1)
            if beat_bounds.size(0) == 0:
                beat_bounds = torch.tensor([[0, max(1, S)]], device=src_tensor.device)
            beat_bounds = beat_bounds.unsqueeze(0)  # [1, M', 2]

            # frontend -> encoder
            if getattr(self.model, "frontend", None) is not None:
                feat = self.model.frontend(src_tensor.transpose(1, 2))
            else:
                feat = src_tensor

            memory = self.model.encoder(feat, src_key_padding_mask=src_mask)

            # fusion if available
            if getattr(self.model, "fusion", None) is not None:
                memory, _, _ = self.model.fusion(
                    memory, beat_bounds, frame_pad_mask=src_mask
                )

            # decoder step
            tgt_mask = self.model.generate_square_subsequent_mask(
                tgt_in.size(1), device=src_tensor.device
            )
            logits = self.model.decoder(
                tgt=tgt_in,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
            )
            return logits

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
                # Take last step’s logits and constrain by ALLOW table before argmax
                last_logits = logits[:, -1, :].squeeze(0)  # [V]

                # Determine last non-special event type from generated ids
                last_type = None
                for tok in reversed(ids):
                    if tok >= NUM_SPECIAL_TOKENS:
                        ev = self.codec.decode_event_index(tok - NUM_SPECIAL_TOKENS)
                        last_type = getattr(ev, "type", None)
                        break

                # Allowed ids: always include EOS, then apply transition table
                allowed_ids = set([EOS_TOKEN])
                allowed_ids |= self.ALLOW.get(last_type, self.ALL_EVENT_IDS)

                # Mask logits outside allowed set
                mask = torch.full_like(last_logits, float("-inf"))
                # Convert set -> list for advanced indexing
                mask[list(allowed_ids)] = 0.0
                constrained = last_logits + mask

                # Greedy pick under constraints
                next_id = int(torch.argmax(constrained, dim=-1).item())
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
        "frontend": cfg.model.get("frontend", {}),
        "fusion": cfg.model.get("fusion", {}),
    }

    model = MT3Trainer(
        model_config=model_config,
        codec=codec,
        learning_rate=cfg.train.learning_rate,
        label_smoothing=cfg.train.get("label_smoothing", 0.0),
        allow_drums=cfg.train.get("allow_drums", False),
        allowed_programs=cfg.train.get("allowed_programs", None),
        shift_loss_weight=cfg.train.get("shift_loss_weight", 0.1),
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
        gradient_clip_val=cfg.train.get("gradient_clip_val", 0.0),
        enable_model_summary=True,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.train.ckpt_path)
    print("Training complete!")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=".*Tempo, Key or Time signature change events found on non-zero tracks.*",
    )
    main()
