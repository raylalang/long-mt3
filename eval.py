import os
import json
import random
from typing import List, Dict, Any, Sequence
from tqdm import tqdm
from collections import Counter

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import pandas as pd
import torchaudio
import note_seq

from long_mt3.vocabularies import VocabularyConfig, build_codec, NUM_SPECIAL_TOKENS
from long_mt3.contrib.mt3.spectrograms import compute_spectrogram, SpectrogramConfig
from long_mt3.contrib.mt3.metrics_utils import get_prettymidi_pianoroll, frame_metrics
from long_mt3.contrib.mt3.note_sequences import (
    NoteEncodingWithTiesSpec,
    NoteEventData,
    NoteEncodingState,
    note_event_data_to_events,
)
from long_mt3.contrib.mt3.run_length_encoding import (
    decode_events,
    encode_and_index_events,
    run_length_encode_shifts_fn,
)
from long_mt3.dataset import MAX_LEN
from long_mt3.vocabularies import EOS_TOKEN, NUM_SPECIAL_TOKENS

from train import (
    MT3Trainer,
)


def _load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)

    if "manifest" in data:
        return data["manifest"]
    return data


def _load_audio_mono(
    path: str, target_sr: int, start_time: float = 0.0, end_time: float = -1.0
) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    audio = wav.squeeze(0)  # [T]

    # Apply crop in seconds
    start_frame = max(0, int(start_time * target_sr))
    if end_time > 0:
        end_frame = min(audio.numel(), int(end_time * target_sr))
    else:
        end_frame = audio.numel()
    if end_frame < start_frame:
        end_frame = start_frame  # empty window → empty slice
    audio = audio[start_frame:end_frame]

    return audio


def _segments_from_spec(spec: np.ndarray, segment_frames: int) -> List[np.ndarray]:
    segs: List[np.ndarray] = []
    for i in range(0, spec.shape[0], segment_frames):
        chunk = spec[i : i + segment_frames]
        if chunk.shape[0] < segment_frames:
            pad = segment_frames - chunk.shape[0]
            chunk = np.pad(chunk, ((0, pad), (0, 0)), mode="constant")
        segs.append(chunk)
    return segs


def _predict_tokens_for_segments(
    model: MT3Trainer,
    segments: List[np.ndarray],
    device: torch.device,
    max_len: int,
    batch_size: int,
) -> List[List[int]]:
    preds: List[List[int]] = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(segments), batch_size), desc="Decoding segments"):
            batch = segments[i : i + batch_size]
            src = torch.tensor(
                np.stack(batch, axis=0), dtype=torch.float32, device=device
            )
            pred_ids = model.autoregressive_decode(src, src_mask=None, max_len=max_len)
            for row in pred_ids:
                toks = row.tolist()
                if 1 in toks:
                    toks = toks[: toks.index(1) + 1]
                preds.append(toks)
    return preds


def _encode_ns_to_tokens(
    ns: note_seq.NoteSequence, codec, window_seconds: float
) -> list[int]:
    # Build per-note events (mirror long_mt3/dataset.py:get_event_times_and_values)
    event_times = [note.start_time for note in ns.notes]
    event_values = [
        NoteEventData(
            pitch=note.pitch,
            velocity=note.velocity,
            program=note.program,
            is_drum=note.is_drum,
        )
        for note in ns.notes
    ]

    # Frame grid for the whole window (mirror dataset)
    steps_per_second = codec.steps_per_second
    num_steps = int(window_seconds * steps_per_second)
    frame_times = [i / steps_per_second for i in range(num_steps)]

    # Encode events with the same state/spec as training
    state = NoteEncodingState()
    events, _, _, _, _ = encode_and_index_events(
        state=state,
        event_times=event_times,
        event_values=event_values,
        encode_event_fn=note_event_data_to_events,
        codec=codec,
        frame_times=frame_times,
    )

    # Run-length encode shifts (exactly like dataset)
    rle_fn = run_length_encode_shifts_fn(codec)
    features = {"targets": events}
    features = rle_fn(features)
    events = features["targets"]

    # Clip and append EOS, then shift by NUM_SPECIAL_TOKENS (exactly like dataset)
    if len(events) >= MAX_LEN:
        events = events[: MAX_LEN - 1]
    base_ids = [int(e) + NUM_SPECIAL_TOKENS for e in events]
    tokens = base_ids + [EOS_TOKEN]
    return tokens


def _decode_tokens_to_ns(all_tokens, codec) -> note_seq.NoteSequence:
    # Filter to real event ids (strip special-token offset)
    event_tokens = [
        t - NUM_SPECIAL_TOKENS for t in all_tokens if t >= NUM_SPECIAL_TOKENS
    ]
    if not event_tokens:
        return note_seq.NoteSequence()

    state = NoteEncodingWithTiesSpec.init_decoding_state_fn()
    NoteEncodingWithTiesSpec.begin_decoding_segment_fn(state)
    decode_events(
        state=state,
        tokens=np.asarray(event_tokens, dtype=np.int32),
        start_time=0.0,
        max_time=None,
        codec=codec,
        decode_event_fn=NoteEncodingWithTiesSpec.decode_event_fn,
    )
    res = NoteEncodingWithTiesSpec.flush_decoding_state_fn(state)

    # Different versions return different shapes, normalize to NoteSequence
    if isinstance(res, note_seq.NoteSequence):
        return res
    if hasattr(res, "note_sequence"):
        return res.note_sequence
    if isinstance(res, dict) and "note_sequence" in res:
        return res["note_sequence"]

    # Last resort: try to coerce, otherwise raise a helpful error
    raise TypeError(f"Unexpected flush_decoding_state_fn() return type: {type(res)}")


def _coerce_devices(d: Any) -> Sequence[int] | int | None:
    # Accept int, list/tuple of ints, or comma-separated string "0,1"
    if d is None:
        return None
    if isinstance(d, int):
        return d
    if isinstance(d, (list, tuple)):
        return list(map(int, d))
    if isinstance(d, str):
        parts = [p.strip() for p in d.split(",") if p.strip() != ""]
        if len(parts) == 1:
            try:
                return int(parts[0])
            except ValueError:
                return None
        return [int(p) for p in parts]
    return None


def _resolve_device(accelerator: str, devices_spec: Any):
    devices = _coerce_devices(devices_spec)

    # CUDA / GPU
    if accelerator in ("gpu", "cuda") and torch.cuda.is_available():
        # Respect CUDA_VISIBLE_DEVICES: indices are relative to visible set
        visible = torch.cuda.device_count()
        if visible == 0:
            return torch.device("cpu")

        # int -> number of devices requested (Trainer semantics)
        if isinstance(devices, int):
            if devices <= 0:
                return torch.device("cuda:0")
            if devices == 1:
                return torch.device("cuda:0")
            print(
                f"[WARN] eval.py is single-process, requested {devices} GPUs. Using cuda:0."
            )
            return torch.device("cuda:0")

        # list/tuple -> explicit GPU indices
        if isinstance(devices, (list, tuple)) and len(devices) > 0:
            idx = int(devices[0])
            if idx < 0 or idx >= visible:
                raise ValueError(
                    f"Requested cuda:{idx}, but only {visible} visible device(s)."
                )
            return torch.device(f"cuda:{idx}")

        # default
        return torch.device("cuda:0")

    # Apple Silicon
    if accelerator == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    # CPU fallback
    return torch.device("cpu")


def _evaluate_example(
    example: Dict[str, Any],
    model: MT3Trainer,
    codec,
    spec_cfg: SpectrogramConfig,
    segment_seconds: float,
    segment_batch_size: int,
    max_decode_len: int,
    device: torch.device,
    start_time: float = 0.0,
    end_time: float = -1.0,
    save_midi_dir: str = None,
    eval_one: bool = False,
) -> Dict[str, Any]:
    # Timebase & segmentation
    sr = spec_cfg.sample_rate
    hop = spec_cfg.hop_width
    fps = float(sr) / float(hop)
    seg_frames = int(segment_seconds * fps)

    # Audio -> spectrogram (cropped)
    audio = _load_audio_mono(
        example["mix_audio_path"], sr, start_time=start_time, end_time=end_time
    )
    spec = compute_spectrogram(audio.numpy(), spec_cfg)  # [S, F]

    # Segment-level AR decode
    segments = _segments_from_spec(spec, seg_frames)
    seg_preds = _predict_tokens_for_segments(
        model, segments, device, max_len=max_decode_len, batch_size=segment_batch_size
    )

    # Concatenate segment tokens (drop duplicated EOS between segments)
    flat_tokens: List[int] = []
    for s, toks in enumerate(seg_preds):
        if s > 0 and flat_tokens and flat_tokens[-1] == 1 and toks and toks[0] == 1:
            toks = toks[1:]
        flat_tokens.extend(toks)

    # Predicted NoteSequence (for metrics/MIDI export)
    est_ns = _decode_tokens_to_ns(flat_tokens, codec)

    # Save predicted MIDI
    if save_midi_dir:
        os.makedirs(save_midi_dir, exist_ok=True)
        mid_base = (
            example.get("unique_id")
            or os.path.basename(example.get("midi_path", "pred.mid")).rsplit(".", 1)[0]
        )
        mid_name = os.path.join(save_midi_dir, f"pred_{mid_base}.mid")
        note_seq.sequence_proto_to_midi_file(est_ns, mid_name)

    # Ground-truth crop to SAME window and rebase to 0 for metrics & tokens
    gt_full = note_seq.midi_file_to_note_sequence(example["midi_path"])
    gt_crop = _crop_ns_to_window(gt_full, start_time, end_time)

    # Encode GT tokens exactly like training targets (window_seconds from audio)
    window_seconds = float(audio.numel()) / float(sr)
    gt_tokens = _encode_ns_to_tokens(gt_crop, codec, window_seconds=window_seconds)

    # Debug snippets: compare Pred vs GT (head & tail)
    if eval_one:
        print("[DIAG] pred types:", _type_counts(flat_tokens, codec))
        print("[DIAG] gt types:", _type_counts(gt_tokens, codec))

        _debug_token_snippets(
            "Eval Pred (head)", flat_tokens, codec, start=0, length=32
        )
        _debug_token_snippets(
            "Eval Pred (tail)",
            flat_tokens,
            codec,
            start=max(0, len(flat_tokens) - 32),
            length=32,
        )
        _debug_token_snippets("Eval GT (head)", gt_tokens, codec, start=0, length=32)
        _debug_token_snippets(
            "Eval GT (tail)",
            gt_tokens,
            codec,
            start=max(0, len(gt_tokens) - 32),
            length=32,
        )

    # Frame metrics on aligned window
    is_drum = False
    ref_roll = get_prettymidi_pianoroll(gt_crop, fps=fps, is_drum=is_drum)
    est_roll = get_prettymidi_pianoroll(est_ns, fps=fps, is_drum=is_drum)
    p, r, f1 = frame_metrics(ref_roll, est_roll, velocity_threshold=1)

    row = {
        "id": example.get("unique_id", example.get("midi_path", "unknown")),
        "dataset": example.get("dataset", "unknown"),
        "num_tokens": len(flat_tokens),
        "frame_precision": float(p),
        "frame_recall": float(r),
        "frame_f1": float(f1),
    }
    return row


def _crop_ns_to_window(
    ns: note_seq.NoteSequence, start_time: float, end_time: float
) -> note_seq.NoteSequence:
    window_end = ns.total_time if end_time < 0 else end_time
    sub = note_seq.extract_subsequence(ns, start_time, window_end)

    window_len = max(0.0, window_end - start_time)
    EPS = 1e-6

    # Clamp to [start_time, window_end] first, then rebase to 0
    for n in sub.notes:
        s = n.start_time
        e = n.end_time
        if s < start_time:
            s = start_time
        if e < start_time:
            e = start_time
        if e > window_end:
            e = window_end
        if s > window_end:
            s = window_end
        if e < s + EPS:
            e = s + EPS

        n.start_time = s - start_time
        n.end_time = e - start_time

    return sub


def _debug_token_snippets(
    name: str, ids: list[int], codec, start: int = 0, length: int = 32
):
    s = start
    e = min(len(ids), start + length)
    if s >= e:
        print(f"[DEBUG] {name} tokens: <empty>")
        return
    ids_snip = ids[s:e]
    dec_snip = []
    for t in ids_snip:
        if t >= NUM_SPECIAL_TOKENS:
            ev = codec.decode_event_index(t - NUM_SPECIAL_TOKENS)
            dec_snip.append(str(ev))
        else:
            dec_snip.append(f"<SPECIAL:{t}>")
    print(f"[DEBUG] {name} IDs[{s}:{e}]: {ids_snip}")
    print(f"[DEBUG] {name} Decoded[{s}:{e}]: {dec_snip}")


def _type_counts(ids: list[int], codec) -> dict[str, int]:
    c = Counter()
    for t in ids:
        if t < NUM_SPECIAL_TOKENS:
            c["SPECIAL"] += 1
        else:
            ev = codec.decode_event_index(t - NUM_SPECIAL_TOKENS)
            c[ev.type] += 1
    return dict(c)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Build codec & spectrogram config to match training
    vocab_cfg = VocabularyConfig()
    codec = build_codec(vocab_cfg, event_types=cfg.data.event_types)
    spec_cfg = SpectrogramConfig(**cfg.data.spectrogram_config)

    # Load Lightning checkpoint
    ckpt_path = cfg.eval.checkpoint
    assert os.path.isfile(ckpt_path), f"Missing checkpoint: {ckpt_path}"

    device = _resolve_device(
        accelerator=cfg.eval.accelerator,
        devices_spec=cfg.eval.devices,
    )
    print(f"[INFO] Using device: {device}")

    # Recreate model from checkpoint, pass runtime codec
    model: MT3Trainer = MT3Trainer.load_from_checkpoint(ckpt_path, codec=codec)
    model.to(device)
    model.eval()

    # Resolve version dir for outputs
    version_dir = os.path.abspath(os.path.join(os.path.dirname(ckpt_path), ".."))
    out_dir = os.path.join(version_dir, "eval_one" if cfg.eval.eval_one else "eval")
    os.makedirs(out_dir, exist_ok=True)

    # Load manifest and choose split
    manifest = _load_manifest(cfg.data.manifest_path)
    if not cfg.data.overfit_one:
        test_split = (
            manifest.get("test")
            or manifest.get("validation")
            or manifest.get("val")
            or []
        )
    else:
        test_split = [manifest["train"][0]]
        print(
            f"[OVERFIT] Using a fixed single training sample for evaluation: {test_split[0]}"
        )
    assert len(test_split) > 0, "No 'test' (or 'validation') split found in manifest."

    # Either eval one example or the whole set
    if cfg.eval.eval_one:
        ex = test_split[0]
        row = _evaluate_example(
            ex,
            model=model,
            codec=codec,
            spec_cfg=spec_cfg,
            segment_seconds=cfg.data.segment_seconds,
            segment_batch_size=cfg.eval.segment_batch_size,
            max_decode_len=2048,
            device=device,
            start_time=cfg.eval.start_time,
            end_time=cfg.eval.end_time,
            save_midi_dir=os.path.join(out_dir, "mid"),
            eval_one=cfg.eval.eval_one,
        )
        df = pd.DataFrame([row])
        df.to_csv(os.path.join(out_dir, "results_eval_one.csv"), index=False)
        print(f"Wrote {len(df)} example to {out_dir}")
        return

    # Full-set evaluation
    rows: List[Dict[str, Any]] = []
    for ex in test_split:
        rows.append(
            _evaluate_example(
                ex,
                model=model,
                codec=codec,
                spec_cfg=spec_cfg,
                segment_seconds=cfg.data.segment_seconds,
                segment_batch_size=cfg.eval.segment_batch_size,
                max_decode_len=2048,
                device=device,
                start_time=cfg.eval.start_time,
                end_time=cfg.eval.end_time,
                save_midi_dir=os.path.join(out_dir, "mid"),
                eval_one=cfg.eval.eval_one,
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    summary = df.groupby("dataset", as_index=False)["num_tokens"].mean()
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"Wrote {len(df)} examples to {out_dir}")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
