import os
import json
from pathlib import Path
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import random


def parse_maestro(root):
    # Structure: {year}/{name}.{wav,mid}
    items = []
    for midi_path in root.glob("*/*.mid"):
        mix_audio_path = midi_path.with_suffix(".wav")
        if mix_audio_path.exists():
            items.append(
                {
                    "dataset": "maestro",
                    "mix_audio_path": str(mix_audio_path.resolve()),
                    "midi_path": str(midi_path.resolve()),
                }
            )
    return items


def parse_urmp(root):
    # Structure: {name}.{AuMix*.wav, *converted.mid}
    items = []
    for mix_audio_path in root.glob("*.AuMix*.wav"):
        name = mix_audio_path.stem.split(".")[0]
        midi_matches = list(root.glob(f"{name}.*converted.mid"))
        if midi_matches:
            items.append(
                {
                    "dataset": "urmp",
                    "mix_audio_path": str(mix_audio_path.resolve()),
                    "midi_path": str(midi_matches[0].resolve()),
                }
            )
    return items


def parse_musicnet(root):
    # Structure: {train, test}_{data, labels_midi}/{name}.{wav,mid}
    result = {"train": [], "test": []}
    for split in ["train", "test"]:
        data_dir = root / f"{split}_data"
        label_dir = root / f"{split}_labels_midi"
        if not data_dir.exists() or not label_dir.exists():
            continue
        for mix_audio_path in data_dir.glob("*.wav"):
            stem = mix_audio_path.stem
            midi_path = label_dir / f"{stem}.mid"
            if midi_path.exists():
                result[split].append(
                    {
                        "dataset": "musicnet",
                        "mix_audio_path": str(mix_audio_path.resolve()),
                        "midi_path": str(midi_path.resolve()),
                    }
                )
    return result


def parse_slakh2100(root):
    # Structure: {train, validation, test}/{name}/{mix.wav, all_src.mid}
    result = {"train": [], "validation": [], "test": []}
    for split in result.keys():
        split_dir = root / split
        if not split_dir.exists():
            continue
        for song_dir in split_dir.iterdir():
            if not song_dir.is_dir():
                continue
            midi_path = song_dir / "all_src.mid"
            mix_audio_path = song_dir / "mix.wav"
            if mix_audio_path.exists() and midi_path.exists():
                canonical = "validation" if split == "validation" else split
                result[canonical].append(
                    {
                        "dataset": "slakh2100",
                        "mix_audio_path": str(mix_audio_path.resolve()),
                        "midi_path": str(midi_path.resolve()),
                    }
                )
    return result


def parse_guitarset(root):
    # Structure:    audio_mono-pickup_mix/{name}_mix.wav
    #               annotation/{name}.mid
    items = []
    audio_dir = root / "audio_mono-pickup_mix"
    midi_dir = root / "annotation"

    for mix_audio_path in audio_dir.glob("*_mix.wav"):
        name = mix_audio_path.stem.replace("_mix", "")
        midi_path = midi_dir / f"{name}.mid"
        if midi_path.exists():
            items.append(
                {
                    "dataset": "guitarset",
                    "mix_audio_path": str(mix_audio_path.resolve()),
                    "midi_path": str(midi_path.resolve()),
                }
            )
    return items


def parse_babyslakh(root):
    # Structure: {name}.{mix.wav, all_src.mid}
    items = []

    for song_dir in root.iterdir():
        if not song_dir.is_dir():
            continue
        midi_path = song_dir / "all_src.mid"
        mix_audio_path = song_dir / "mix.wav"
        if mix_audio_path.exists() and midi_path.exists():
            items.append(
                {
                    "dataset": "babyslakh",
                    "mix_audio_path": str(mix_audio_path.resolve()),
                    "midi_path": str(midi_path.resolve()),
                }
            )
    return items


def split_items(items, ratios, seed):
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    return {
        "train": items[:n_train],
        "validation": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


parsers = {
    "maestro": parse_maestro,
    "urmp": parse_urmp,
    "musicnet": parse_musicnet,
    "slakh2100": parse_slakh2100,
    "guitarset": parse_guitarset,
    "babyslakh": parse_babyslakh,
}


def build_manifest(cfg):
    manifest = {"train": [], "validation": [], "test": []}

    # Aggregate datasets by usage (train/validation/test)
    all_datasets = set()
    for split in ["train", "validation", "test"]:
        all_datasets.update(cfg.datasets.get(split, []))

    for dataset_name in all_datasets:
        if dataset_name not in parsers:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        root = Path(cfg.paths[dataset_name])
        parsed = parsers[dataset_name](root)

        if isinstance(parsed, list):  # unsplit dataset — do full split once
            if dataset_name not in cfg.split:
                raise ValueError(
                    f"Missing split ratio for unsplit dataset '{dataset_name}'"
                )
            ratios = cfg.split[dataset_name]
            split_result = split_items(parsed, ratios, cfg.seed)
            for split in ["train", "validation", "test"]:
                if dataset_name in cfg.datasets.get(split, []):
                    manifest[split].extend(split_result[split])
        elif isinstance(parsed, dict):  # pre-split dataset
            for split in ["train", "validation", "test"]:
                if dataset_name in cfg.datasets.get(split, []):
                    if split in parsed:
                        manifest[split].extend(parsed[split])
                    else:
                        print(
                            f"Skipping {dataset_name} for '{split}' split (not available)"
                        )

    return manifest


@hydra.main(config_path="configs", config_name="build_manifest", version_base=None)
def main(cfg: DictConfig):
    manifest = build_manifest(cfg)

    # Include config and datetime in output
    output = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "manifest": manifest,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("manifests") / f"manifest_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Manifest written to: {output_path}")
    for split in cfg.datasets:
        print(f"  {split}: {len(manifest[split])} items")


if __name__ == "__main__":
    main()
