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
            )

    def collate_fn(self, batch):
        specs, tokens = zip(*batch)
        spec_lens = [s.shape[0] for s in specs]
        token_lens = [t.shape[0] for t in tokens]
        max_spec = max(spec_lens)
        max_token = max(token_lens)

        padded_specs = torch.zeros(len(batch), max_spec, specs[0].shape[1])
        padded_tokens = torch.full(
            (len(batch), max_token), fill_value=0, dtype=torch.long
        )

        for i in range(len(batch)):
            padded_specs[i, :spec_lens[i]] = specs[i]
            padded_tokens[i, :token_lens[i]] = tokens[i]

        return padded_specs, padded_tokens

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            prefetch_factor=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            prefetch_factor=4
        )
