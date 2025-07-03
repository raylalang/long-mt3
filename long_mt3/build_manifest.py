from datetime import datetime
import os
import glob
import json
import random
import argparse


def find_tracks(root):
    return sorted(
        [os.path.join(root, d) for d in os.listdir(root) if d.startswith("Track")]
    )


def make_examples(track_path):
    examples = []

    # Mixture-level example
    mix_wav = os.path.join(track_path, "mix.wav")
    mix_mid = os.path.join(track_path, "all_src.mid")
    if os.path.isfile(mix_wav) and os.path.isfile(mix_mid):
        examples.append(
            {
                "audio_path": mix_wav,
                "midi_path": mix_mid,
                "track": os.path.basename(track_path),
                "type": "mixture",
            }
        )

    # Stem-level examples
    stems_dir = os.path.join(track_path, "stems")
    midi_dir = os.path.join(track_path, "MIDI")
    if os.path.isdir(stems_dir) and os.path.isdir(midi_dir):
        stem_wavs = sorted(glob.glob(os.path.join(stems_dir, "S*.wav")))
        for wav_path in stem_wavs:
            basename = os.path.splitext(os.path.basename(wav_path))[0]
            midi_path = os.path.join(midi_dir, f"{basename}.mid")
            if os.path.isfile(midi_path):
                examples.append(
                    {
                        "audio_path": wav_path,
                        "midi_path": midi_path,
                        "track": os.path.basename(track_path),
                        "stem": basename,
                        "type": "stem",
                    }
                )
    return examples


def split_examples(examples, ratios, seed=420):
    random.Random(seed).shuffle(examples)
    n = len(examples)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return (
        examples[:n_train],
        examples[n_train : n_train + n_val],
        examples[n_train + n_val :],
    )


def write_jsonl(filename, examples):
    with open(filename, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} entries to {filename}")


def build_manifest(data_root, output_dir, ratios, seed):
    all_examples = []
    for track_path in find_tracks(data_root):
        all_examples.extend(make_examples(track_path))

    if "debug" in output_dir:
        print(f"Debug mode: using only the first 10 examples from {len(all_examples)}")
        all_examples = all_examples[:10]

    train, val, test = split_examples(all_examples, ratios, seed)

    os.makedirs(output_dir, exist_ok=True)
    write_jsonl(os.path.join(output_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(output_dir, "val.jsonl"), val)
    write_jsonl(os.path.join(output_dir, "test.jsonl"), test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = os.path.join("manifests", f"{args.output_prefix}_{timestamp}")

    build_manifest(args.data_root, manifest_dir, args.ratios, args.seed)

    config_path = os.path.join(manifest_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(
            {"data_root": args.data_root, "ratios": args.ratios, "seed": args.seed},
            f,
            indent=2,
        )
