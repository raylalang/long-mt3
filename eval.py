import os
import json
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
    note_encoding_state_to_events,
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
    start_frame = max(0, int(start_time * target_sr))
    if end_time > 0:
        end_frame = min(audio.numel(), int(end_time * target_sr))
    else:
        end_frame = audio.numel()
    if end_frame < start_frame:
        end_frame = start_frame
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


def _decode_tokens_to_ns(all_tokens, codec) -> note_seq.NoteSequence:
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
    return res["sequence"]


def _encode_ns_to_tokens(
    ns: note_seq.NoteSequence, codec, window_seconds: float
) -> list[int]:
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
    steps_per_second = codec.steps_per_second
    num_steps = int(window_seconds * steps_per_second)
    frame_times = [i / steps_per_second for i in range(num_steps)]
    state = NoteEncodingState()
    NoteEncodingWithTiesSpec.begin_encoding_segment_fn(state)
    rle_fn = run_length_encode_shifts_fn(codec)
    features = dict(
        codec=codec,
        onsets=event_times,
        event_values=event_values,
        frame_times=frame_times,
    )
    features = rle_fn(features)
    events = features["targets"]
    if len(events) >= MAX_LEN:
        events = events[: MAX_LEN - 1]
    base_ids = [int(e) + NUM_SPECIAL_TOKENS for e in events]
    tokens = base_ids + [EOS_TOKEN]
    return tokens


def _tie_prefix_from_prev_ns(
    prev_ns: note_seq.NoteSequence, segment_seconds: float, codec
) -> list[int]:
    boundary = segment_seconds
    active = []
    for n in prev_ns.notes:
        if getattr(n, "is_drum", False):
            continue
        if n.start_time <= boundary and n.end_time > boundary:
            active.append(n)
    state = NoteEncodingState()
    NoteEncodingWithTiesSpec.begin_encoding_segment_fn(state)
    for n in active:
        ev = NoteEventData(
            pitch=n.pitch,
            velocity=0,
            program=n.program,
            is_drum=n.is_drum,
        )
        # Spec helper to register tied events for this segment
        NoteEncodingWithTiesSpec.add_tied_event_fn(state, ev)
    tie_events = note_encoding_state_to_events(state)
    tie_ids = [int(codec.encode_event(e)) + NUM_SPECIAL_TOKENS for e in tie_events]
    return tie_ids


def _decode_one_with_prefix(
    model: MT3Trainer,
    spec_chunk: np.ndarray,
    device: torch.device,
    max_len: int,
    prefix_ids: list[int],
) -> list[int]:
    x = torch.tensor(spec_chunk[None, ...], dtype=torch.float32, device=device)
    with torch.inference_mode():
        out = model.autoregressive_decode(
            x, src_mask=None, max_len=max_len, prefix_ids=prefix_ids
        )
        toks = out[0].tolist() if isinstance(out, torch.Tensor) else out[0]
    if 1 in toks:
        toks = toks[: toks.index(1) + 1]
    return toks


def _sequential_decode_with_ties(
    model: MT3Trainer,
    segments: Sequence[np.ndarray],
    codec,
    device: torch.device,
    max_len: int,
    segment_seconds: float,
) -> list[list[int]]:
    preds: list[list[int]] = []
    prev_ns: note_seq.NoteSequence = note_seq.NoteSequence()
    for si, seg in enumerate(segments):
        prefix = _tie_prefix_from_prev_ns(prev_ns, segment_seconds, codec)
        toks = _decode_one_with_prefix(
            model=model,
            spec_chunk=seg,
            device=device,
            max_len=max_len,
            prefix_ids=prefix,
        )
        preds.append(toks)
        seg_ns = _decode_tokens_to_ns(toks, codec)
        prev_ns = seg_ns
    return preds


def _coerce_devices(d):
    if d is None:
        return None
    if isinstance(d, int):
        return d
    if isinstance(d, (list, tuple)):
        return [int(x) for x in d]
    if isinstance(d, dict):
        return d
    if isinstance(d, str):
        if d.strip().lower() == "auto":
            return "auto"
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
    if accelerator in ("gpu", "cuda") and torch.cuda.is_available():
        visible = torch.cuda.device_count()
        if visible == 0:
            return torch.device("cpu")
        if isinstance(devices, int):
            if devices <= 0:
                return torch.device("cuda:0")
            if devices == 1:
                return torch.device("cuda:0")
            return torch.device("cuda:0")
        if isinstance(devices, (list, tuple)) and len(devices) > 0:
            return torch.device(
                f"cuda:{devices[0] if isinstance(devices[0], int) else 0}"
            )
        return torch.device("cuda:0")
    if accelerator in ("mps",) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ns_to_note_arrays(ns, pitch_min=None, pitch_max=None):
    on, off, pc = [], [], []
    for n in ns.notes:
        if getattr(n, "is_drum", False):
            continue
        if pitch_min is not None and n.pitch < pitch_min:
            continue
        if pitch_max is not None and n.pitch > pitch_max:
            continue
        s = max(0.0, float(n.start_time))
        e = max(s, float(n.end_time))
        on.append(s)
        off.append(e)
        pc.append(int(n.pitch))
    return np.asarray(on), np.asarray(off), np.asarray(pc)


def _greedy_match(
    ref_on,
    ref_off,
    ref_p,
    est_on,
    est_off,
    est_p,
    onset_tol,
    offset_tol,
    offset_ratio,
    need_off,
):
    ref_used = np.zeros(len(ref_on), dtype=bool)
    est_used = np.zeros(len(est_on), dtype=bool)
    tp = 0
    for j in np.argsort(est_on):
        cand = np.where(
            (~ref_used)
            & (ref_p == est_p[j])
            & (np.abs(ref_on - est_on[j]) <= onset_tol)
        )[0]
        if cand.size == 0:
            continue
        i = cand[np.argmin(np.abs(ref_on[cand] - est_on[j]))]
        if need_off:
            ref_dur = max(1e-6, ref_off[i] - ref_on[i])
            off_tol = max(offset_tol, offset_ratio * ref_dur)
            if np.abs(ref_off[i] - est_off[j]) > off_tol:
                continue
        ref_used[i] = True
        est_used[j] = True
        tp += 1
    fp = int((~est_used).sum())
    fn = int((~ref_used).sum())
    return tp, fp, fn


def onset_f1(ref_ns, est_ns, onset_tolerance=0.05):
    r_on, r_off, r_p = _ns_to_note_arrays(ref_ns)
    e_on, e_off, e_p = _ns_to_note_arrays(est_ns)
    if len(r_on) == len(e_on) == 0:
        return 1.0, 1.0, 1.0
    if len(e_on) == 0:
        return 0.0, 0.0, 0.0
    try:
        import mir_eval

        ref = np.stack([r_on, r_off, r_p], 1)
        est = np.stack([e_on, e_off, e_p], 1)
        P, R, F = mir_eval.transcription.precision_recall_f1_overlap(
            ref, est, onset_tolerance=onset_tolerance, offset_ratio=None
        )
        return float(P), float(R), float(F)
    except Exception:
        tp, fp, fn = _greedy_match(
            r_on, r_off, r_p, e_on, e_off, e_p, onset_tolerance, 0.0, 0.0, False
        )
        P = tp / (tp + fp) if (tp + fp) else 0.0
        R = tp / (tp + fn) if (tp + fn) else 0.0
        F = 2 * P * R / (P + R) if (P + R) else 0.0
        return P, R, F


def onset_offset_f1(
    ref_ns,
    est_ns,
    onset_tolerance=0.05,
    offset_tolerance=0.05,
    offset_tolerance_ratio=0.2,
):
    r_on, r_off, r_p = _ns_to_note_arrays(ref_ns)
    e_on, e_off, e_p = _ns_to_note_arrays(est_ns)
    if len(r_on) == len(e_on) == 0:
        return 1.0, 1.0, 1.0
    if len(e_on) == 0:
        return 0.0, 0.0, 0.0
    try:
        import mir_eval

        ref = np.stack([r_on, r_off, r_p], 1)
        est = np.stack([e_on, e_off, e_p], 1)
        P, R, F = mir_eval.transcription.precision_recall_f1_overlap(
            ref,
            est,
            onset_tolerance=onset_tolerance,
            offset_ratio=offset_tolerance_ratio,
        )
        return float(P), float(R), float(F)
    except Exception:
        tp, fp, fn = _greedy_match(
            r_on,
            r_off,
            r_p,
            e_on,
            e_off,
            e_p,
            onset_tolerance,
            offset_tolerance,
            offset_tolerance_ratio,
            True,
        )
        P = tp / (tp + fp) if (tp + fp) else 0.0
        R = tp / (tp + fn) if (tp + fn) else 0.0
        F = 2 * P * R / (P + R) if (P + R) else 0.0
        return P, R, F


def _crop_ns_to_window(
    ns: note_seq.NoteSequence, start_time: float, end_time: float
) -> note_seq.NoteSequence:
    window_end = ns.total_time if end_time < 0 else end_time
    sub = note_seq.extract_subsequence(ns, start_time, window_end)
    window_len = max(0.0, window_end - start_time)
    EPS = 1e-6
    rebased = note_seq.NoteSequence()
    rebased.ticks_per_quarter = sub.ticks_per_quarter
    for n in sub.notes:
        m = rebased.notes.add()
        m.CopyFrom(n)
        m.start_time = max(0.0, n.start_time - start_time)
        m.end_time = max(m.start_time, n.end_time - start_time)
    rebased.total_time = max(
        window_len - EPS, max((n.end_time for n in rebased.notes), default=0.0)
    )
    return rebased


def _debug_token_snippets(name: str, ids: list[int], codec, start: int, length: int):
    s = start
    e = min(len(ids), start + length)
    if s >= e:
        print(f"[DEBUG] {name}: empty slice")
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
    sr = spec_cfg.sample_rate
    hop = spec_cfg.hop_width
    fps = float(sr) / float(hop)
    seg_frames = int(segment_seconds * fps)

    audio = _load_audio_mono(
        example["mix_audio_path"], sr, start_time=start_time, end_time=end_time
    )
    spec = compute_spectrogram(audio.numpy(), spec_cfg)

    segs = _segments_from_spec(spec, seg_frames)

    pred_token_segments = _sequential_decode_with_ties(
        model=model,
        segments=segs,
        codec=codec,
        device=device,
        max_len=max_decode_len,
        segment_seconds=segment_seconds,
    )

    flat_tokens: List[int] = []
    for seg_tokens in pred_token_segments:
        if flat_tokens and seg_tokens and flat_tokens[-1] == EOS_TOKEN:
            seg_tokens = [t for t in seg_tokens if t != EOS_TOKEN]
        flat_tokens.extend(seg_tokens)
    if not flat_tokens or flat_tokens[-1] != EOS_TOKEN:
        flat_tokens.append(EOS_TOKEN)

    est_ns = _decode_tokens_to_ns(flat_tokens, codec)

    if save_midi_dir:
        os.makedirs(save_midi_dir, exist_ok=True)
        mid_base = (
            example.get("unique_id")
            or os.path.basename(example.get("midi_path", "pred.mid")).rsplit(".", 1)[0]
        )
        mid_name = os.path.join(save_midi_dir, f"pred_{mid_base}.mid")
        note_seq.sequence_proto_to_midi_file(est_ns, mid_name)

    gt_full = note_seq.midi_file_to_note_sequence(example["midi_path"])
    gt_crop = _crop_ns_to_window(gt_full, start_time, end_time)

    window_seconds = float(audio.numel()) / float(sr)
    gt_tokens = _encode_ns_to_tokens(gt_crop, codec, window_seconds=window_seconds)

    if eval_one:
        print("[DIAG] pred types:", _type_counts(flat_tokens, codec))
        print("[DIAG] gt types:", _type_counts(gt_tokens, codec))
        _debug_token_snippets("Eval Pred (head)", flat_tokens, codec, 0, 32)
        _debug_token_snippets(
            "Eval Pred (tail)", flat_tokens, codec, max(0, len(flat_tokens) - 32), 32
        )
        _debug_token_snippets("Eval GT   (head)", gt_tokens, codec, 0, 32)
        _debug_token_snippets(
            "Eval GT   (tail)", gt_tokens, codec, max(0, len(gt_tokens) - 32), 32
        )

    is_drum = False
    ref_roll = get_prettymidi_pianoroll(gt_crop, fps=fps, is_drum=is_drum)
    est_roll = get_prettymidi_pianoroll(est_ns, fps=fps, is_drum=is_drum)
    p, r, f1 = frame_metrics(ref_roll, est_roll, velocity_threshold=1)
    onset_p, onset_r, onset_f = onset_f1(gt_crop, est_ns, onset_tolerance=0.050)
    oo_p, oo_r, oo_f = onset_offset_f1(
        gt_crop,
        est_ns,
        onset_tolerance=0.050,
        offset_tolerance=0.050,
        offset_tolerance_ratio=0.20,
    )

    row = {
        "id": example.get("unique_id", example.get("midi_path", "unknown")),
        "dataset": example.get("dataset", "unknown"),
        "num_tokens": len(flat_tokens),
        "frame_precision": float(p),
        "frame_recall": float(r),
        "frame_f1": float(f1),
        "onset_precision": float(onset_p),
        "onset_recall": float(onset_r),
        "onset_f1": float(onset_f),
        "onset_offset_precision": float(oo_p),
        "onset_offset_recall": float(oo_r),
        "onset_offset_f1": float(oo_f),
    }
    return row


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    vocab_cfg = VocabularyConfig()
    codec = build_codec(vocab_cfg, event_types=cfg.data.event_types)
    spec_cfg = SpectrogramConfig(**cfg.data.spectrogram_config)

    ckpt_path = cfg.train.resume_from_checkpoint
    assert ckpt_path and os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
    device = _resolve_device(cfg.train.accelerator, cfg.train.devices)

    model = MT3Trainer.load_from_checkpoint(
        ckpt_path, map_location=device, strict=False
    )
    model.eval()
    model.to(device)

    version_dir = os.path.abspath(os.path.join(os.path.dirname(ckpt_path), ".."))
    out_dir = os.path.join(version_dir, "eval_one" if cfg.eval.eval_one else "eval")
    os.makedirs(out_dir, exist_ok=True)

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
        df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
        print(df.head())
        print(f"Wrote 1 example to {out_dir}")
        return

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
