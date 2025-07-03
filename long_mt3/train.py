import argparse
import json
import glob
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from .dataset import MT3Dataset
from .model import MT3Model
from .contrib.spectrograms import SpectrogramConfig
from .vocabularies import build_codec, VocabularyConfig


def load_manifest(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def collate_fn(batch):
    specs, tokens = zip(*batch)
    spec_lens = [s.shape[0] for s in specs]
    token_lens = [t.shape[0] for t in tokens]

    max_spec = max(spec_lens)
    max_token = max(token_lens)

    padded_specs = torch.zeros(len(batch), max_spec, specs[0].shape[1])
    padded_tokens = torch.full((len(batch), max_token), fill_value=0, dtype=torch.long)

    for i in range(len(batch)):
        padded_specs[i, : spec_lens[i]] = specs[i]
        padded_tokens[i, : token_lens[i]] = tokens[i]

    return padded_specs, padded_tokens


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.manifest == "debug":
        folders = sorted(glob.glob("manifests/debug_*"))
        assert folders, "No debug_* folders found."
        manifest_path = os.path.join(folders[-1], "train.jsonl")
    else:
        manifest_path = args.manifest

    manifest = load_manifest(manifest_path)
    codec = build_codec(VocabularyConfig())
    spec_config = SpectrogramConfig()

    dataset = MT3Dataset(manifest, spec_config, codec)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    model = MT3Model(
        input_dim=spec_config.n_mels,
        vocab_size=codec.num_classes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for specs, tokens in tqdm(loader):
            specs = specs.to(device)
            tokens = tokens.to(device)

            optimizer.zero_grad()
            output = model(specs, tokens[:, :-1])
            loss = criterion(
                output.reshape(-1, output.shape[-1]),
                tokens[:, 1:].reshape(-1),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="debug")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)

    args = parser.parse_args()
    main(args)
