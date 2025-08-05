import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import torchaudio
from tqdm import tqdm
import pandas as pd
import yaml
import os
import json
import note_seq
import random

from long_mt3.model import MT3Model
from long_mt3.vocabularies import build_codec, VocabularyConfig, get_special_tokens
from long_mt3.contrib.mt3.note_sequences import NoteEncodingWithTiesSpec
from long_mt3.contrib.mt3.metrics_utils import event_predictions_to_ns, frame_metrics, get_prettymidi_pianoroll
from long_mt3.contrib.mt3.spectrograms import SpectrogramConfig, compute_spectrogram, split_audio

from train import MT3Trainer

torch.backends.cudnn.benchmark = True

def load_audio(path, sr=16000):
    audio, orig_sr = torchaudio.load(path)
    if orig_sr != sr:
        audio = torchaudio.functional.resample(audio, orig_sr, sr)
    return audio.mean(0)


def segment_audio(features, segment_frames, hop_width=256, sample_rate=16000):
    segments = []
    for i in range(0, features.shape[0], segment_frames):
        chunk = features[i:i+segment_frames]
        if chunk.shape[0] < segment_frames:
            pad = segment_frames - chunk.shape[0]
            chunk = np.pad(chunk, ((0, pad), (0, 0)))
        start_time = i * hop_width / sample_rate
        segments.append((start_time, chunk))
    return segments


def run_model(model, segments, device, codec, max_len=1024, batch_size=1, verbose=False):
    model.eval()
    predictions = []
    special_tokens = get_special_tokens()
    eos_token = special_tokens["eos_token"]

    progress = tqdm(total=len(segments), desc="Decoding segments", disable=not verbose)

    for i in range(0, len(segments), batch_size):
        segment_chunk = segments[i:i + batch_size]

        src_batch = torch.stack([
            torch.tensor(feat, dtype=torch.float32) for _, feat in segment_chunk
        ])

        try:
            src_batch = src_batch.to(device)
            memory_batch = model.encoder(src_batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise
            else:
                raise

        B = src_batch.size(0)
        tgt = torch.full((B, 1), eos_token, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        out_tokens = [[] for _ in range(B)]

        for step in range(max_len):
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
            try:
                logits = model.decoder(tgt, memory_batch, tgt_mask=tgt_mask)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    raise
                else:
                    raise

            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            # if verbose:
            #     print(f"Step {step:03d}: next_token = {next_token.tolist()}")

            for j in range(B):
                if not finished[j]:
                    token = next_token[j].item()
                    if token == eos_token:
                        finished[j] = True
                    else:
                        out_tokens[j].append(token)

            if finished.all():
                break

            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

        for (start_time, _), tokens in zip(segment_chunk, out_tokens):
            predictions.append({
                "start_time": start_time,
                "est_tokens": tokens,
                "raw_inputs": None
            })
            if verbose:
                print("Final predicted token sequence:")
                for i, t in enumerate(tokens):
                    print(f"{i:02d}: {t} → {codec.decode_event_index(t)}")


        torch.cuda.empty_cache()
        progress.update(B)

    progress.close()
    return predictions


def evaluate(example_id, prediction, midi_path, codec, encoding_spec):
    ref_ns = note_seq.midi_file_to_note_sequence(midi_path)
    result = event_predictions_to_ns(prediction, codec, encoding_spec)
    est_ns = result["est_ns"]

    est_roll = get_prettymidi_pianoroll(est_ns, fps=100, is_drum=False)
    ref_roll = get_prettymidi_pianoroll(ref_ns, fps=100, is_drum=False)

    p, r, f1 = frame_metrics(ref_roll, est_roll, velocity_threshold=1)
    return {"precision": p, "recall": r, "f1": f1}, est_ns


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    if cfg.eval.accelerator == "gpu" and isinstance(cfg.eval.devices, list):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in cfg.eval.devices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(cfg.data.manifest_path) as f:
        manifest = json.load(f)["manifest"]

    spec_config = SpectrogramConfig(**cfg.data.spectrogram_config)
    vocab_config = VocabularyConfig()
    codec = build_codec(vocab_config)
    encoding_spec = NoteEncodingWithTiesSpec

    hparams_path = os.path.abspath(cfg.eval.hparams_yaml)
    checkpoint_path = os.path.abspath(cfg.eval.checkpoint)

    hparams_version = hparams_path.split(os.sep)
    ckpt_version = checkpoint_path.split(os.sep)

    try:
        hparams_version_idx = hparams_version.index("lightning_logs") + 1
        ckpt_version_idx = ckpt_version.index("lightning_logs") + 1
        hparams_version_tag = hparams_version[hparams_version_idx]
        ckpt_version_tag = ckpt_version[ckpt_version_idx]
    except (ValueError, IndexError):
        raise ValueError("Unable to extract version tag from paths.")

    if hparams_version_tag != ckpt_version_tag:
        raise ValueError(
            f"Version mismatch: hparams={hparams_version_tag}, checkpoint={ckpt_version_tag}"
        )

    with open(hparams_path, 'r') as f:
        hparams = yaml.safe_load(f)
    model_config = hparams["model_config"]

    trainer_module = MT3Trainer(model_config=model_config, learning_rate=0)
    ckpt = torch.load(checkpoint_path, map_location=device)
    trainer_module.load_state_dict(ckpt["state_dict"])
    model = trainer_module.model
    model.to(device)
    model = torch.compile(model)

    checkpoint_tag = os.path.splitext(os.path.basename(checkpoint_path))[0]
    version_dir = os.path.join("lightning_logs", hparams_version_tag)
    is_eval_one = getattr(cfg.eval, "eval_one", False)
    save_midi = getattr(cfg.eval, "save_midi", False)
    output_dir = os.path.join(version_dir, "eval_one" if is_eval_one else "eval")
    os.makedirs(output_dir, exist_ok=True)


    hop_width = cfg.data.spectrogram_config.hop_width
    segment_frames = int(cfg.data.segment_seconds * cfg.data.spectrogram_config.sample_rate / hop_width)

    if getattr(cfg.eval, "eval_one", False):
        # ex = manifest["test"][0]
        ex = random.choice(manifest["test"])
        audio = load_audio(ex["mix_audio_path"])
        features = compute_spectrogram(audio.numpy(), spec_config)
        segments = segment_audio(features, segment_frames)
        prediction = run_model(model, segments, device, codec, max_len=1024, batch_size=cfg.eval.segment_batch_size, verbose=True)
        metrics, est_ns = evaluate(ex.get("unique_id", ex["midi_path"]), prediction, ex["midi_path"], codec, encoding_spec)
        metrics.update({
            "id": ex.get("unique_id", ex["midi_path"]),
            "dataset": ex["dataset"]
        })
        df = pd.DataFrame([metrics])
        df.to_csv(os.path.join(output_dir, f"eval_one_metrics_{checkpoint_tag}.csv"), index=False)

        midi_filename = os.path.join(output_dir, f"pred_{checkpoint_tag}_{os.path.basename(ex['midi_path']).replace('.mid', '')}.mid")
        note_seq.sequence_proto_to_midi_file(est_ns, midi_filename)

        print(f"\nEval one summary: {metrics}")
        print(f"Saved predicted MIDI to {midi_filename}")
    else:
        results = []
        for ex in tqdm(manifest["test"]):
            audio = load_audio(ex["mix_audio_path"])
            features = compute_spectrogram(audio.numpy(), spec_config)
            segments = segment_audio(features, segment_frames)
            prediction = run_model(model, segments, device, codec, max_len=1024, batch_size=cfg.eval.segment_batch_size)
            metrics, est_ns = evaluate(ex.get("unique_id", ex["midi_path"]), prediction, ex["midi_path"], codec, encoding_spec)
            metrics.update({
                "id": ex.get("unique_id", ex["midi_path"]),
                "dataset": ex["dataset"]
            })
            results.append(metrics)
            if save_midi:
                midi_filename = os.path.join(output_dir, f"pred_{checkpoint_tag}_{os.path.basename(ex['midi_path']).replace('.mid', '')}.mid")
                note_seq.sequence_proto_to_midi_file(est_ns, midi_filename)

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, f"results_{checkpoint_tag}.csv"), index=False)

        summary = df.groupby("dataset")[["f1", "precision", "recall"]].mean().reset_index()
        summary.to_csv(os.path.join(output_dir, f"summary_{checkpoint_tag}.csv"), index=False)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
