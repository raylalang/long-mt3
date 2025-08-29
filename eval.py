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


def _decode_tokens_to_ns(all_tokens, codec, start_time: float = 0.0) -> note_seq.NoteSequence:
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
        start_time=start_time,
        max_time=None,
        codec=codec,
        decode_event_fn=NoteEncodingWithTiesSpec.decode_event_fn,
    )
    res = NoteEncodingWithTiesSpec.flush_decoding_state_fn(state)
    return res


def _encode_ns_to_tokens(
    ns: note_seq.NoteSequence, codec, window_seconds: float
) -> list[int]:
    # Gather note events (times + values)
    event_times = [float(note.start_time) for note in ns.notes]
    event_values = [
        NoteEventData(
            pitch=int(note.pitch),
            velocity=int(note.velocity),
            program=int(getattr(note, "program", 0) or 0),
            is_drum=bool(getattr(note, "is_drum", False)),
        )
        for note in ns.notes
    ]

    # Frame times for the window, matching the codec's step rate
    steps_per_second = float(codec.steps_per_second)
    num_steps = int(round(window_seconds * steps_per_second))
    # ensure at least one frame so we don’t pass an empty list
    num_steps = max(1, num_steps)
    frame_times = [i / steps_per_second for i in range(num_steps)]

    # Initialize encoding state
    state = NoteEncodingWithTiesSpec.init_encoding_state_fn()

    # Encode + index
    events, event_start_indices, event_end_indices, state_events, state_event_indices = (
        encode_and_index_events(
            state=state,
            event_times=event_times,
            event_values=event_values,
            encode_event_fn=NoteEncodingWithTiesSpec.encode_event_fn,
            codec=codec,
            frame_times=frame_times,
            encoding_state_to_events_fn=NoteEncodingWithTiesSpec.encoding_state_to_events_fn,
        )
    )

    # Run-length encode shifts
    rle = run_length_encode_shifts_fn(codec)
    features = {"targets": events}
    features = rle(features)
    events = features["targets"]

    # Cap length and append EOS
    if len(events) >= MAX_LEN:
        events = events[: MAX_LEN - 1]
    base_ids = [int(e) + NUM_SPECIAL_TOKENS for e in events]
    tokens = base_ids + [EOS_TOKEN]
    return tokens


def _tie_prefix_from_prev_ns(
    prev_ns: note_seq.NoteSequence, segment_seconds: float, codec
) -> list[int]:
    boundary = float(segment_seconds)   # ← this is the correct boundary
    active = []
    for n in prev_ns.notes:
        if getattr(n, "is_drum", False):
            continue
        if n.start_time <= boundary and n.end_time > boundary:
            active.append(n)
    state = NoteEncodingState()
    for n in active:
        if getattr(n, "is_drum", False):
            continue
        prog = int(getattr(n, "program", 0) or 0)
        state.active_pitches[(int(n.pitch), prog)] = 1
    tie_events = note_encoding_state_to_events(state)
    tie_ids = [int(codec.encode_event(ev)) + NUM_SPECIAL_TOKENS for ev in tie_events]
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
    if EOS_TOKEN in toks:
        toks = toks[: toks.index(EOS_TOKEN) + 1]
    return toks


def _sequential_decode_with_ties(
    model: MT3Trainer,
    segments: Sequence[np.ndarray],
    codec,
    device: torch.device,
    max_len: int,
    segment_seconds: float,
    eval_one: bool = False
) -> list[list[int]]:
    preds: list[list[int]] = []
    prev_ns: note_seq.NoteSequence = note_seq.NoteSequence()
    if eval_one:
        pbar = tqdm(total=len(segments), desc="segments", dynamic_ncols=True, leave=False)
    for si, seg in enumerate(segments):
        prefix = _tie_prefix_from_prev_ns(prev_ns, segment_seconds, codec)
        toks = _decode_one_with_prefix(
            model=model,
            spec_chunk=seg,
            device=device,
            max_len=max_len,
            prefix_ids=prefix,
        )
        if EOS_TOKEN not in toks and len(toks) >= max_len:
            print(f"[WARN] no EOS, hit max_len on segment {si} (len={len(toks)})", flush=True)

        preds.append(toks)
        seg_ns = _decode_tokens_to_ns(toks, codec)
        prev_ns = seg_ns
        if eval_one:
            pbar.update(1)
    if eval_one:
        pbar.close()

    total_len = sum(len(t) for t in preds)
    eos_hits = sum(1 for t in preds if EOS_TOKEN in t)
    max_hit  = sum(1 for t in preds if (EOS_TOKEN not in t and len(t) >= max_len))
    avg_len  = total_len / max(1, len(preds))
    print(f"[DEBUG] seg_summary: n={len(preds)} avg_len={avg_len:.1f} eos%={100.0*eos_hits/max(1,len(preds)):.1f} maxcap%={100.0*max_hit/max(1,len(preds)):.1f}", flush=True)

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


def _segments_tokens_to_ns(
    token_segments: Sequence[list[int]],
    segment_seconds: float,
    codec,
) -> note_seq.NoteSequence:
    combined = note_seq.NoteSequence()
    prev_ns = note_seq.NoteSequence() 

    for si, seg_tokens in enumerate(token_segments):
        # Build tie prefix from the previous segment's LOCAL timeline.
        # Boundary is always the segment length in that local frame.
        tie_ids = _tie_prefix_from_prev_ns(prev_ns, segment_seconds, codec) if si > 0 else []

        # Decode this segment LOCALLY at t=0 so ties are computed correctly.
        toks_for_decode = tie_ids + seg_tokens
        seg_local = _decode_tokens_to_ns(toks_for_decode, codec, start_time=0.0)

        # When merging, place it at its ABSOLUTE offset.
        abs_start = si * float(segment_seconds)
        for n in seg_local.notes:
            m = combined.notes.add()
            m.CopyFrom(n)
            m.start_time = n.start_time + abs_start
            m.end_time   = n.end_time   + abs_start
        combined.total_time = max(combined.total_time, seg_local.total_time + abs_start)

        # For the next iteration, keep the LOCAL (t=0) decode as the previous segment.
        prev_ns = seg_local

    return combined


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
        P, R, f = mir_eval.transcription.precision_recall_f1_overlap(
            ref, est, onset_tolerance=onset_tolerance, offset_ratio=None
        )
        return float(P), float(R), float(f)
    except Exception:
        print("Error occurred while doing mir_eval, using fallback method _greedy_match")
        tp, fp, fn = _greedy_match(
            r_on, r_off, r_p, e_on, e_off, e_p, onset_tolerance, 0.0, 0.0, False
        )
        P = tp / (tp + fp) if (tp + fp) else 0.0
        R = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * P * R / (P + R) if (P + R) else 0.0
        return P, R, f


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
        P, R, f = mir_eval.transcription.precision_recall_f1_overlap(
            ref,
            est,
            onset_tolerance=onset_tolerance,
            offset_ratio=offset_tolerance_ratio,
        )
        return float(P), float(R), float(f)
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
        f = 2 * P * R / (P + R) if (P + R) else 0.0
        return P, R, f


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

def _shift_vals(ids, codec):
    vals = []
    for t in ids:
        if t >= NUM_SPECIAL_TOKENS:
            ev = codec.decode_event_index(t - NUM_SPECIAL_TOKENS)
            if ev.type == "shift":
                vals.append(int(ev.value))
    if not vals:
        return 0, 0.0, 0.0
    import numpy as np
    return len(vals), float(np.mean(vals)), float(np.median(vals))

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

    print("[DEBUG] sr", sr,
        "hop", hop,
        "fps", fps,
        "steps/s", codec.steps_per_second,
        "segment_seconds", segment_seconds,
        "seg_frames", seg_frames,
        "spec_T", spec.shape[0],
        flush=True)

    pred_token_segments = _sequential_decode_with_ties(
        model=model,
        segments=segs,
        codec=codec,
        device=device,
        max_len=max_decode_len,
        segment_seconds=segment_seconds,
        eval_one=eval_one
    )


    est_ns = _segments_tokens_to_ns(
        pred_token_segments, segment_seconds=segment_seconds, codec=codec
    )
    gt_full = note_seq.midi_file_to_note_sequence(example["midi_path"])
    gt_crop = _crop_ns_to_window(gt_full, start_time, end_time)
    window_seconds = float(audio.numel()) / float(sr)
    gt_tokens = _encode_ns_to_tokens(gt_crop, codec, window_seconds=window_seconds)
    print("[DEBUG] est_total_time_before_crop", float(est_ns.total_time), flush=True)
    est_ns = _crop_ns_to_window(est_ns, 0.0, window_seconds)

    print("[DEBUG] gt_notes", len(gt_crop.notes),
      "est_notes", len(est_ns.notes),
      "window_seconds", window_seconds,
      flush=True)
    print("[DEBUG] gt first onsets:", sorted([float(n.start_time) for n in gt_crop.notes])[:10], flush=True)
    print("[DEBUG] est first onsets:", sorted([float(n.start_time) for n in est_ns.notes])[:10], flush=True)

    # auxiliary loss evaluation on the evaluation window
    aux_logs = {}
    with torch.no_grad():
        # encoder embeddings for the full window (chunked to respect PE length)
        spec_tensor = torch.tensor(spec[None, ...], dtype=torch.float32, device=device)  # [1, T, F]
        feat = (
            model.model.frontend(spec_tensor)
            if getattr(model.model, "frontend", None) is not None
            else spec_tensor
        )  # [1, T, F’]

        # figure out the max length the encoder's pos-encoder supports
        pe = getattr(getattr(model.model, "encoder", None), "pos_encoder", None)
        pe_max_len = None
        if pe is not None:
            if hasattr(pe, "pe"):                   # sinusoidal buffer
                pe_max_len = int(pe.pe.size(1))     # e.g., 2048
            elif hasattr(pe, "max_len"):            # future-proof
                pe_max_len = int(pe.max_len)

        if pe_max_len is None or feat.size(1) <= pe_max_len:
            # no need to chunk
            memory = model.model.encoder(feat)  # [1, T, D]
        else:
            # chunk along time, encode each slice, then concat
            T = feat.size(1)
            chunks = []
            for s in range(0, T, pe_max_len):
                e = min(T, s + pe_max_len)
                chunks.append(model.model.encoder(feat[:, s:e]))  # [1, e-s, D]
            memory = torch.cat(chunks, dim=1)  # [1, T, D]

        # build frame labels from GT crop at eval FPS to match spectrogram T
        sr = spec_cfg.sample_rate
        hop = spec_cfg.hop_width
        fps_eval = float(sr) / float(hop)

        def _frame_labels_eval(ns_local):
            T = spec.shape[0]
            y = torch.zeros(T, 88, dtype=torch.float32, device=device)
            for n in ns_local.notes:
                p = int(n.pitch) - 21
                if 0 <= p < 88:
                    s = int(max(0, round(n.start_time * fps_eval)))
                    e = int(min(T, round(n.end_time * fps_eval)))
                    if e > s:
                        y[s:e, p] = 1.0
            return y

        fl_eval = _frame_labels_eval(gt_crop)  # [T, 88]

        beat_bounds_eval = None
        beat_targets_eval = None
        if getattr(model.model, "fusion", None) is not None:
            # derive beats from GT window
            def _estimate_qpm_eval(ns_local, default_qpm=120.0):
                if hasattr(ns_local, "tempos") and len(ns_local.tempos) > 0 and ns_local.tempos[0].qpm > 0:
                    return float(ns_local.tempos[0].qpm)
                return float(default_qpm)

            qpm = _estimate_qpm_eval(gt_crop)
            beat_period = 60.0 / qpm
            window_seconds = float(spec.shape[0]) / fps_eval
            M = max(1, int(round(window_seconds / beat_period)))
            starts = [i * beat_period for i in range(M)]
            ends = [min(window_seconds, (i + 1) * beat_period) for i in range(M)]
            bounds = torch.zeros(1, M, 2, dtype=torch.long, device=device)
            centers, durs = [], []
            for i in range(M):
                s = int(max(0, round(starts[i] * fps_eval)))
                e = int(min(spec.shape[0], round(ends[i] * fps_eval)))
                if e <= s:
                    e = min(spec.shape[0], s + 1)
                bounds[0, i, 0] = s
                bounds[0, i, 1] = e
                centers.append((starts[i] + ends[i]) * 0.5)
                durs.append(max(1e-6, ends[i] - starts[i]))
            onsets = [float(n.start_time) for n in gt_crop.notes]
            targets = torch.zeros(1, M, dtype=torch.float32, device=device)
            for i in range(M):
                cs = centers[i]
                dur = durs[i]
                local = [t for t in onsets if starts[i] <= t < ends[i]]
                if local:
                    vals = [max(-0.5, min(0.5, (t - cs) / dur)) for t in local]
                    targets[0, i] = float(np.mean(vals))
            beat_bounds_eval = bounds
            beat_targets_eval = targets

            # fusion forward to get beat embeddings (memory already full-length)
            memory, beat_emb, _ = model.model.fusion(
                frame_emb=memory,
                beat_bounds=bounds[0],
                beats_per_bar=model.model.beats_per_bar,
                frame_mask=None,
            )

        # heads and losses
        if getattr(model.model, "frame_head", None) is not None:
            frame_logits = model.model.frame_head(memory)  # [1, T, 88]
            if fl_eval is not None:
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    frame_logits, fl_eval.unsqueeze(0)
                )
                aux_logs["eval_frame_bce"] = float(bce.item())
            aux_logs["frame_pred_shape"] = tuple(frame_logits.shape)

        if (
            beat_bounds_eval is not None
            and getattr(model.model, "beat_head", None) is not None
        ):
            beat_pred = model.model.beat_head(beat_emb)  # [1, M, 1] or [1, M]
            if beat_pred.dim() == 3 and beat_pred.size(-1) == 1:
                beat_pred = beat_pred.squeeze(-1)
            if beat_targets_eval is not None:
                l1 = torch.nn.functional.l1_loss(beat_pred, beat_targets_eval)
                aux_logs["eval_beat_l1"] = float(l1.item())
            aux_logs["beat_pred_shape"] = tuple(beat_pred.shape)


    if save_midi_dir:
        os.makedirs(save_midi_dir, exist_ok=True)
        mid_base = (
            example.get("unique_id")
            or os.path.basename(example.get("midi_path", "pred.mid")).rsplit(".", 1)[0]
        )
        mid_name = os.path.join(save_midi_dir, f"pred_{mid_base}.mid")
        note_seq.sequence_proto_to_midi_file(est_ns, mid_name)

    if eval_one:
        flat_tokens: List[int] = []
        for seg_tokens in pred_token_segments:
            # avoid stacking multiple EOS when segments are concatenated
            if flat_tokens and seg_tokens and flat_tokens[-1] == EOS_TOKEN:
                seg_tokens = [t for t in seg_tokens if t != EOS_TOKEN]
            flat_tokens.extend(seg_tokens)
        if not flat_tokens or flat_tokens[-1] != EOS_TOKEN:
            flat_tokens.append(EOS_TOKEN)
            
        print("[DEBUG] pred types:", _type_counts(flat_tokens, codec))
        print("[DEBUG] gt types:", _type_counts(gt_tokens, codec))
        _debug_token_snippets("[DEBUG] Eval Pred (head)", flat_tokens, codec, 0, 32)
        _debug_token_snippets(
            "[DEBUG] Eval Pred (tail)", flat_tokens, codec, max(0, len(flat_tokens) - 32), 32
        )
        _debug_token_snippets("[DEBUG] Eval GT (head)", gt_tokens, codec, 0, 32)
        _debug_token_snippets(
            "[DEBUG] Eval GT (tail)", gt_tokens, codec, max(0, len(gt_tokens) - 32), 32
        )

        pred_shift_n, pred_shift_mean, pred_shift_med = _shift_vals(flat_tokens, codec)
        gt_shift_n,   gt_shift_mean,   gt_shift_med   = _shift_vals(gt_tokens, codec)
        print(f"[DEBUG] shifts pred n/mean/median: {pred_shift_n}/{pred_shift_mean:.2f}/{pred_shift_med:.2f}", flush=True)
        print(f"[DEBUG] shifts  gt  n/mean/median: {gt_shift_n}/{gt_shift_mean:.2f}/{gt_shift_med:.2f}", flush=True)

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
        "dataset": example.get("dataset", "unknown"),
        "mix_audio_path": example.get("mix_audio_path", ""),
        "midi_path": example.get("midi_path", ""),
        "num_tokens": sum(len(s) for s in pred_token_segments),
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
    row.update(aux_logs)
    return row


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # vocab_cfg = VocabularyConfig()
    # codec = build_codec(vocab_cfg, event_types=cfg.data.event_types)
    spec_cfg = SpectrogramConfig(**cfg.data.spectrogram_config)

    checkpoint = cfg.eval.checkpoint
    assert checkpoint and os.path.exists(checkpoint), f"Missing checkpoint: {checkpoint}"
    device = _resolve_device(cfg.eval.accelerator, cfg.eval.devices)

    model = MT3Trainer.load_from_checkpoint(
        checkpoint, map_location=device, strict=False
    )
    model.eval()
    model.to(device)

    codec = model.codec

    try:
        shift_vals = []
        for i in range(codec.num_classes):
            ev = codec.decode_event_index(i)
            if getattr(ev, "type", None) == "shift":
                shift_vals.append(int(ev.value))
        if shift_vals:
            mn, mx = min(shift_vals), max(shift_vals)
            print(
                f"[DEBUG] shift_value_range: min={mn} max={mx} count={len(shift_vals)} "
                f"| steps_per_second={codec.steps_per_second} "
                f"| seconds_range=[{mn/codec.steps_per_second:.3f}, {mx/codec.steps_per_second:.3f}]",
                flush=True,
            )
        else:
            print("[DEBUG] shift_value_range: no 'shift' events found in codec", flush=True)
    except Exception as e:
        print(f"[DEBUG] shift_value_range: failed to enumerate ({e})", flush=True)

    v_proj = model.model.decoder.out_proj.out_features
    expected = codec.num_classes + NUM_SPECIAL_TOKENS
    print(f"[DEBUG] decoder_out={v_proj}, codec.num_classes={codec.num_classes}, specials={NUM_SPECIAL_TOKENS}")
    assert v_proj == expected, f"Vocab mismatch: decoder_out={v_proj} vs codec_size={expected}"

    version_dir = os.path.abspath(os.path.join(os.path.dirname(checkpoint), ".."))
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

    dynamic_max_len = int(5 * codec.steps_per_second * float(cfg.data.segment_seconds)) + 64
    print(f"[DEBUG] dynamic_max_len={dynamic_max_len}", flush=True)

    if cfg.eval.eval_one:
        ex = test_split[0]
        print("[EVAL-ONE] example:")
        print("audio:", ex.get("mix_audio_path", "NA"))
        print("midi :", ex.get("midi_path", "NA"))
        
        row = _evaluate_example(
            ex,
            model=model,
            codec=codec,
            spec_cfg=spec_cfg,
            segment_seconds=cfg.data.segment_seconds,
            segment_batch_size=cfg.eval.segment_batch_size,
            max_decode_len=dynamic_max_len,
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
    pbar = tqdm(total=len(test_split), desc="test samples", dynamic_ncols=True, leave=False)
    for ex in test_split:
        rows.append(
            _evaluate_example(
                ex,
                model=model,
                codec=codec,
                spec_cfg=spec_cfg,
                segment_seconds=cfg.data.segment_seconds,
                segment_batch_size=cfg.eval.segment_batch_size,
                max_decode_len=dynamic_max_len,
                device=device,
                start_time=cfg.eval.start_time,
                end_time=cfg.eval.end_time,
                save_midi_dir=os.path.join(out_dir, "mid"),
                eval_one=cfg.eval.eval_one,
            )
        )
        pbar.update(1)
    pbar.close()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    summary = df.groupby("dataset", as_index=False)["num_tokens"].mean()
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"Wrote {len(df)} examples to {out_dir}")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
