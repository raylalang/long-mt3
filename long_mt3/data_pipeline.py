import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
from collections import defaultdict
from .dataset import MT3Dataset, MT3TemperatureSampler


class MT3DataPipeline(pl.LightningDataModule):
    def __init__(
        self,
        manifest_path,
        spectrogram_config,
        codec,
        batch_size=4,
        num_workers=4,
        segment_seconds=10.0,
        temperature=1.0,
        debug=False,
        overfit_one=False,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.spectrogram_config = spectrogram_config
        self.codec = codec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.segment_seconds = segment_seconds
        self.temperature = temperature
        self.dataset = {}
        self.debug = debug
        self.overfit_one = overfit_one

    def setup(self, stage=None):
        with open(self.manifest_path) as f:
            manifest = json.load(f)

        for split in ["train", "validation"]:
            dataset_map = defaultdict(list)
            for sample in manifest["manifest"][split]:
                dataset_name = sample["dataset"]
                dataset_map[dataset_name].append(sample)

            if self.overfit_one and split == "train":
                first = manifest["manifest"]["train"][0]
                name = first["dataset"]
                self.dataset[split] = MT3Dataset(
                    data_list=[first],
                    spectrogram_config=self.spectrogram_config,
                    codec=self.codec,
                    segment_seconds=self.segment_seconds,
                    split=split,
                    debug=self.debug,
                )
                print(
                    f"[OVERFIT] Using a fixed single training sample from dataset '{name}'."
                )
            else:
                self.dataset[split] = MT3TemperatureSampler(
                    dataset_map=dataset_map,
                    spectrogram_config=self.spectrogram_config,
                    codec=self.codec,
                    segment_seconds=self.segment_seconds,
                    temperature=self.temperature,
                    split=split,
                    debug=self.debug,
                )
                if self.debug:
                    max_debug_samples = 8
                    original_len = len(self.dataset[split])
                    self.dataset[split] = torch.utils.data.Subset(
                        self.dataset[split], range(min(max_debug_samples, original_len))
                    )
                    print(
                        f"[DEBUG] Truncated {split} dataset from {original_len} to {len(self.dataset[split])} samples."
                    )

            print(f"Loaded {len(self.dataset[split])} samples for {split} split.")

    def collate_fn(self, batch):
        specs = [b["src"] for b in batch]
        in_ids = [b["decoder_input_ids"] for b in batch]
        tgt_ids = [b["decoder_target_ids"] for b in batch]

        spec_lens = [s.shape[0] for s in specs]
        max_spec = max(spec_lens)
        feat_dim = specs[0].shape[1]

        in_lens = [t.shape[0] for t in in_ids]
        tgt_lens = [t.shape[0] for t in tgt_ids]
        max_len = max(max(in_lens), max(tgt_lens))

        B = len(batch)
        padded_specs = torch.zeros(B, max_spec, feat_dim)
        spec_masks = torch.ones(B, max_spec, dtype=torch.bool)

        padded_in = torch.full((B, max_len), fill_value=0, dtype=torch.long)
        padded_tgt = torch.full((B, max_len), fill_value=0, dtype=torch.long)
        in_masks = torch.ones(B, max_len, dtype=torch.bool)
        tgt_masks = torch.ones(B, max_len, dtype=torch.bool)

        # frame labels [T, 88] -> pad to max_spec
        fl_list = [b.get("frame_labels") for b in batch]
        has_fl = all(x is not None for x in fl_list)
        padded_fl = None
        if has_fl:
            num_pitches = fl_list[0].shape[-1]
            padded_fl = torch.zeros(B, max_spec, num_pitches)
        # beats: variable M
        bb_list = [b.get("beat_bounds") for b in batch]
        bt_list = [b.get("beat_targets") for b in batch]
        has_beats = all(x is not None for x in bb_list) and all(
            x is not None for x in bt_list
        )
        if has_beats:
            Ms = [x.shape[0] for x in bb_list]
            max_M = max(Ms)
            padded_bb = torch.full((B, max_M, 2), fill_value=-1, dtype=torch.long)
            padded_bt = torch.zeros(B, max_M, dtype=torch.float32)
        else:
            padded_bb = None
            padded_bt = None

        on_list = [b.get("onset_labels") for b in batch]
        off_list = [b.get("offset_labels") for b in batch]
        has_on = all(x is not None for x in on_list)
        has_off = all(x is not None for x in off_list)
        if has_on and has_off:
            padded_on = torch.zeros(B, max_spec, on_list[0].shape[-1])
            padded_off = torch.zeros(B, max_spec, off_list[0].shape[-1])
        else:
            padded_on, padded_off = None, None

        # velocity bins [T, 88] (long; -1 means ignore) -> pad to max_spec
        vb_list = [b.get("velocity_bins") for b in batch]
        has_vb = all(x is not None for x in vb_list)
        if has_vb:
            padded_vb = torch.full(
                (B, max_spec, vb_list[0].shape[-1]), fill_value=-1, dtype=torch.long
            )
        else:
            padded_vb = None

        for i in range(B):
            Ls, Li, Lt = spec_lens[i], in_lens[i], tgt_lens[i]
            padded_specs[i, :Ls] = specs[i]
            spec_masks[i, :Ls] = False
            padded_in[i, :Li] = in_ids[i]
            in_masks[i, :Li] = False
            padded_tgt[i, :Lt] = tgt_ids[i]
            tgt_masks[i, :Lt] = False

            if has_fl:
                fl = fl_list[i]
                padded_fl[i, : fl.shape[0]] = fl
            if has_beats:
                bb = bb_list[i]
                bt = bt_list[i]
                m = bb.shape[0]
                padded_bb[i, :m] = bb
                padded_bt[i, :m] = bt
            if has_on:
                on = on_list[i]
                padded_on[i, : on.shape[0]] = on
            if has_off:
                off = off_list[i]
                padded_off[i, : off.shape[0]] = off
            if has_vb:
                vb = vb_list[i]
                padded_vb[i, : vb.shape[0]] = vb

        out = {
            "spec": padded_specs,
            "decoder_input_ids": padded_in,
            "decoder_target_ids": padded_tgt,
            "spec_mask": spec_masks,
            "in_mask": in_masks,
            "tgt_mask": tgt_masks,
            "beat_bounds": padded_bb,  # [B, M*, 2] with -1/-1 padding
            "frame_labels": padded_fl,  # [B, T*, 88]
            "beat_targets": padded_bt,  # [B, M*]
            "targets": {
                "frame": padded_fl,  # keep existing usage
                "onset": padded_on,  # None if not built
                "offset": padded_off,  # None if not built
                "velocity": padded_vb,  # None if not built
                "beat_center": padded_bt,  # reuse beat targets
            },
        }
        return out

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            sampler=None,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            pin_memory=True,
        )
