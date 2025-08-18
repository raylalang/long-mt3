import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
from collections import defaultdict
from .dataset import MT3TemperatureSampler


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
            self.dataset[split] = MT3TemperatureSampler(
                dataset_map=dataset_map,
                spectrogram_config=self.spectrogram_config,
                codec=self.codec,
                segment_seconds=self.segment_seconds,
                temperature=self.temperature,
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

            if self.overfit_one and split == "train":
                original_len = len(self.dataset[split])
                self.dataset[split] = torch.utils.data.Subset(self.dataset[split], [0])
                print(
                    f"[OVERFIT] Truncated {split} dataset from {original_len} to 1 sample."
                )

            print(f"Loaded {len(self.dataset[split])} samples for {split} split.")

    def collate_fn(self, batch):
        specs, in_ids, tgt_ids = zip(*batch)

        spec_lens = [s.shape[0] for s in specs]
        max_spec = max(spec_lens)
        feat_dim = specs[0].shape[1]

        in_lens = [t.shape[0] for t in in_ids]
        tgt_lens = [t.shape[0] for t in tgt_ids]
        max_len = max(max(in_lens), max(tgt_lens))

        padded_specs = torch.zeros(len(batch), max_spec, feat_dim)
        spec_masks = torch.ones(len(batch), max_spec, dtype=torch.bool)

        padded_in = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
        padded_tgt = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
        in_masks = torch.ones(len(batch), max_len, dtype=torch.bool)
        tgt_masks = torch.ones(len(batch), max_len, dtype=torch.bool)

        for i in range(len(batch)):
            Ls, Li, Lt = spec_lens[i], in_lens[i], tgt_lens[i]
            padded_specs[i, :Ls] = specs[i]
            spec_masks[i, :Ls] = False

            padded_in[i, :Li] = in_ids[i]
            in_masks[i, :Li] = False

            padded_tgt[i, :Lt] = tgt_ids[i]
            tgt_masks[i, :Lt] = False

        return padded_specs, padded_in, padded_tgt, spec_masks, in_masks, tgt_masks

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
