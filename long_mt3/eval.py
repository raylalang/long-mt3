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

from .model import MT3Model
from .vocabularies import build_codec, VocabularyConfig
from .contrib.mt3.note_sequences import NoteEncodingWithTiesSpec
from .contrib.mt3.metrics_utils import event_predictions_to_ns, frame_metrics, get_prettymidi_pianoroll
from .contrib.mt3.spectrograms import SpectrogramConfig, compute_spectrogram, split_audio

from .train import MT3Trainer

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


def generate_square_subsequent_mask(sz):
    return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)


def run_model(model, segments, device, bos_token=1, eos_token=2, max_len=1024, batch_size=1, verbose=False):
    model.eval()
    predictions = []

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
        tgt = torch.full((B, 1), bos_token, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        out_tokens = [[] for _ in range(B)]

        for step in range(max_len):
            tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)
            try:
                logits = model.decoder(tgt, memory_batch, tgt_mask=tgt_mask)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("⚠️ CUDA OOM during decoder pass. Try reducing batch size or max_len.")
                    torch.cuda.empty_cache()
                    raise
                else:
                    raise

            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            # if verbose:
            #     eos_count = (next_token == eos_token).sum().item()
            #     print(f"Step {step}: EOS predicted in {eos_count}/{B} sequences")

            for j in range(B):
                if not finished[j]:
                    token = next_token[j].item()
                    if token == eos_token:
                        finished[j] = True
                        progress.update(1)
                    else:
                        out_tokens[j].append(token)

            if finished.all():
                break

            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

        unfinished = (~finished).nonzero(as_tuple=False).squeeze(-1).tolist()
        if unfinished:
            if verbose:
                print(f"⚠️ {len(unfinished)} sequences reached max_len without eos_token.")
            progress.update(len(unfinished))

        for (start_time, _), tokens in zip(segment_chunk, out_tokens):
            predictions.append({
                "start_time": start_time,
                "est_tokens": tokens,
                "raw_inputs": None
            })

        torch.cuda.empty_cache()

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

    hparams_path = cfg.eval.hparams_yaml
    with open(hparams_path, 'r') as f:
        hparams = yaml.safe_load(f)
    model_config = hparams["model_config"]

    trainer_module = MT3Trainer(model_config=model_config, learning_rate=0)
    ckpt = torch.load(cfg.eval.checkpoint, map_location=device)
    trainer_module.load_state_dict(ckpt["state_dict"])
    model = trainer_module.model
    model.to(device)
    model = torch.compile(model)

    hop_width = cfg.data.spectrogram_config.hop_width
    segment_frames = int(cfg.data.segment_seconds * cfg.data.spectrogram_config.sample_rate / hop_width)

    if getattr(cfg.eval, "eval_one", False):
        ex = manifest["test"][0]
        audio = load_audio(ex["mix_audio_path"])
        features = compute_spectrogram(audio.numpy(), spec_config)
        segments = segment_audio(features, segment_frames)
        prediction = run_model(model, segments, device, bos_token=1, eos_token=2, max_len=1024, batch_size=cfg.eval.segment_batch_size, verbose=True)
        metrics, est_ns = evaluate(ex.get("unique_id", ex["midi_path"]), prediction, ex["midi_path"], codec, encoding_spec)
        metrics.update({
            "id": ex.get("unique_id", ex["midi_path"]),
            "dataset": ex["dataset"]
        })
        df = pd.DataFrame([metrics])
        df.to_csv("eval_one_metrics.csv", index=False)

        midi_filename = f"pred_{os.path.basename(ex['midi_path']).replace('.mid', '')}.mid"
        note_seq.sequence_proto_to_midi_file(est_ns, midi_filename)

        print(f"\nEval one summary: {metrics}")
        print(f"Saved predicted MIDI to {midi_filename}")
    else:
        results = []
        for ex in tqdm(manifest["test"]):
            audio = load_audio(ex["mix_audio_path"])
            features = compute_spectrogram(audio.numpy(), spec_config)
            segments = segment_audio(features, segment_frames)
            prediction = run_model(model, segments, device, bos_token=1, eos_token=2, max_len=1024, batch_size=cfg.eval.segment_batch_size)
            metrics, _ = evaluate(ex.get("unique_id", ex["midi_path"]), prediction, ex["midi_path"], codec, encoding_spec)
            metrics.update({
                "id": ex.get("unique_id", ex["midi_path"]),
                "dataset": ex["dataset"]
            })
            results.append(metrics)

        df = pd.DataFrame(results)
        df.to_csv("results.csv", index=False)

        summary = df.groupby("dataset")[["f1", "precision", "recall"]].mean().reset_index()
        summary.to_csv("summary.csv", index=False)

        print("\nPer-dataset summary:")
        print(summary.to_string(index=False))

        print("\nAll results:")
        for r in results:
            print(f"{r['id']} ({r['dataset']}): F1={r['f1']:.3f}, P={r['precision']:.3f}, R={r['recall']:.3f}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
