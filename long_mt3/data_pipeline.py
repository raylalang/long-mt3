import torch
from torch.utils.data import DataLoader
from .dataset import MT3Dataset
import pytorch_lightning as pl


class MT3DataPipeline(pl.LightningDataModule):
    def __init__(
        self, data_list_path, spectrogram_config, codec, batch_size=4, num_workers=4
    ):
        super().__init__()
        self.data_list_path = data_list_path
        self.spectrogram_config = spectrogram_config
        self.codec = codec
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        import json

        with open(self.data_list_path) as f:
            data_list = [json.loads(line) for line in f]
        self.dataset = MT3Dataset(data_list, self.spectrogram_config, self.codec)

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
            padded_specs[i, : spec_lens[i]] = specs[i]
            padded_tokens[i, : token_lens[i]] = tokens[i]
        return padded_specs, padded_tokens

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
