import random
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info
import torch.nn.functional as F
import note_seq

from .vocabularies import PAD_TOKEN, EOS_TOKEN, UNK_TOKEN, NUM_SPECIAL_TOKENS
from .contrib.mt3.spectrograms import compute_spectrogram
from .contrib.mt3.run_length_encoding import (
    encode_and_index_events,
    run_length_encode_shifts_fn,
    note_encoding_state_to_events,
    note_sequence_to_onsets_and_offsets_and_programs,
)
from .contrib.mt3.event_codec import Event
from .contrib.mt3.note_sequences import (
    NoteEncodingState,
    NoteEventData,
    note_event_data_to_events,
    validate_note_sequence,
    trim_overlapping_notes,
)

MAX_LEN = 2048
PITCH_MIN = 21
PITCH_MAX = 108
NUM_PITCHES = PITCH_MAX - PITCH_MIN + 1


def _estimate_qpm(ns: note_seq.NoteSequence, default_qpm: float = 120.0) -> float:
    if hasattr(ns, "tempos") and len(ns.tempos) > 0 and ns.tempos[0].qpm > 0:
        return float(ns.tempos[0].qpm)
    return float(default_qpm)


def _build_frame_labels(
    ns: note_seq.NoteSequence, frames: int, fps: float
) -> torch.Tensor:
    y = torch.zeros(frames, NUM_PITCHES, dtype=torch.float32)
    for n in ns.notes:
        p = int(n.pitch) - PITCH_MIN
        if p < 0 or p >= NUM_PITCHES:
            continue
        s = int(max(0, round(n.start_time * fps)))
        e = int(min(frames, round(n.end_time * fps)))
        if e > s:
            y[s:e, p] = 1.0
    return y


def _build_onset_offset_labels(
    ns: note_seq.NoteSequence, frames: int, fps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    onset = torch.zeros(frames, NUM_PITCHES, dtype=torch.float32)
    offset = torch.zeros(frames, NUM_PITCHES, dtype=torch.float32)
    for n in ns.notes:
        p = int(n.pitch) - PITCH_MIN
        if p < 0 or p >= NUM_PITCHES:
            continue
        s = int(max(0, round(n.start_time * fps)))
        e = int(min(frames, round(n.end_time * fps)))
        if s < frames:
            onset[s, p] = 1.0
        if e - 1 >= 0:
            offset[max(0, e - 1), p] = 1.0
    return onset, offset


def _build_velocity_bins(
    ns: note_seq.NoteSequence, frames: int, fps: float, num_bins: int = 32
) -> torch.Tensor:
    """
    Discretize MIDI velocities at note onsets into [0, num_bins-1]; -1 elsewhere.
    """
    vel = torch.full((frames, NUM_PITCHES), -1, dtype=torch.long)
    for n in ns.notes:
        p = int(n.pitch) - PITCH_MIN
        if p < 0 or p >= NUM_PITCHES:
            continue
        s = int(max(0, round(n.start_time * fps)))
        if s >= frames:
            continue
        bin_idx = int(round((n.velocity / 127.0) * (num_bins - 1)))
        bin_idx = max(0, min(num_bins - 1, bin_idx))
        vel[s, p] = bin_idx
    return vel


def _build_beats_and_targets(
    ns: note_seq.NoteSequence,
    segment_seconds: float,
    frames: int,
    fps: float,
    default_beats: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    qpm = _estimate_qpm(ns)
    beat_period = 60.0 / qpm
    num_beats_est = max(1, int(round(segment_seconds / beat_period)))
    # cap to a reasonable range; pad/collate will handle variable length
    M = max(1, min(num_beats_est, 64))

    # beat boundaries in seconds, rebased to [0, segment_seconds)
    starts = [i * beat_period for i in range(M)]
    ends = [min(segment_seconds, (i + 1) * beat_period) for i in range(M)]
    # if tempos are missing, ensure coverage of the window
    if M == 1 and segment_seconds > beat_period * 1.5:
        M = default_beats
        starts = [i * (segment_seconds / M) for i in range(M)]
        ends = [min(segment_seconds, (i + 1) * (segment_seconds / M)) for i in range(M)]

    # map to frame indices
    bounds = torch.zeros(M, 2, dtype=torch.long)
    for i in range(M):
        s = int(max(0, round(starts[i] * fps)))
        e = int(min(frames, round(ends[i] * fps)))
        if e <= s:
            e = min(frames, s + 1)
        bounds[i, 0] = s
        bounds[i, 1] = e

    # beat targets: normalized signed offset of note onsets from beat center in [-0.5, 0.5]
    # mean across onsets within each beat; empty beats -> 0.0
    centers = [(starts[i] + ends[i]) * 0.5 for i in range(M)]
    durations = [max(1e-6, (ends[i] - starts[i])) for i in range(M)]
    onsets = [float(n.start_time) for n in ns.notes]
    targets = torch.zeros(M, dtype=torch.float32)
    for i in range(M):
        cs = centers[i]
        dur = durations[i]
        # collect onsets inside this beat
        local = [t for t in onsets if starts[i] <= t < ends[i]]
        if not local:
            targets[i] = 0.0
        else:
            # signed distance to center normalized by beat duration
            vals = [max(-0.5, min(0.5, (t - cs) / dur)) for t in local]
            targets[i] = float(np.mean(vals)) if len(vals) > 0 else 0.0
    return bounds, targets


class MT3Dataset(Dataset):
    """
    Yields (spectrogram, decoder_input_ids, decoder_target_ids).
    """

    def __init__(
        self,
        data_list,
        spectrogram_config,
        codec,
        segment_seconds,
        split=None,
        debug=False,
    ):
        self.data_list = data_list
        self.spectrogram_config = spectrogram_config
        self.codec = codec
        self.segment_seconds = segment_seconds
        self.cache = OrderedDict()
        self.max_cache_size = 100
        self.split = split
        self.debug = debug

        frames_per_second = (
            spectrogram_config.sample_rate / spectrogram_config.hop_width
        )
        self.segment_frames = int(segment_seconds * frames_per_second)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        worker_info = get_worker_info()
        if worker_info is not None:
            if not hasattr(self, "_worker_cache"):
                self._worker_cache = OrderedDict()
            cache = self._worker_cache
        else:
            cache = self.cache

        if idx in cache:
            spec, ns = cache[idx]
            print("[DEBUG] Cache hit!")
        else:
            sample = self.data_list[idx]
            audio = self.load_audio(sample["mix_audio_path"])
            spec = compute_spectrogram(audio, self.spectrogram_config)
            spec = torch.from_numpy(spec).float()
            ns = note_seq.midi_file_to_note_sequence(sample["midi_path"])
            validate_note_sequence(ns)
            ns = trim_overlapping_notes(ns)
            cache[idx] = (spec, ns)
            if len(cache) > self.max_cache_size:
                cache.popitem(last=False)

            if self.debug:
                print(
                    f"[DEBUG] Dataset ({self.split}) Sample audio path: {sample['mix_audio_path']}"
                )
                print(
                    f"[DEBUG] Dataset ({self.split}) Sample MIDI path: {sample['midi_path']}"
                )
                print(f"[DEBUG] Dataset ({self.split}) Sample spec shape: {spec.shape}")
                print(
                    f"[DEBUG] NoteSequence ({self.split}): {len(ns.notes)} notes, total_time={ns.total_time:.2f}s"
                )

        n_frames = spec.shape[0]

        # crop/pad to fixed segment
        if n_frames >= self.segment_frames:
            # compute/reuse a deterministic start that hits a note
            if not hasattr(self, "_fixed_starts"):
                self._fixed_starts = {}
            if idx in self._fixed_starts:
                start_frame = self._fixed_starts[idx]
            else:
                # choose a note onset near the median to be stable
                onsets = [n.start_time for n in ns.notes]
                if onsets:
                    frames_per_second = (
                        self.spectrogram_config.sample_rate
                        / self.spectrogram_config.hop_width
                    )
                    onset = float(np.median(onsets))
                    start_frame = (
                        int(onset * frames_per_second) - self.segment_frames // 4
                    )
                    start_frame = max(
                        0, min(start_frame, n_frames - self.segment_frames)
                    )
                else:
                    start_frame = 0  # no notes, fall back to start
                self._fixed_starts[idx] = start_frame
            end_frame = start_frame + self.segment_frames
            spec = spec[start_frame:end_frame]
        else:
            start_frame = 0
            end_frame = n_frames
            pad_len = self.segment_frames - n_frames
            spec = F.pad(spec, (0, 0, 0, pad_len))

        # seconds for event slicing
        frames_per_second = (
            self.spectrogram_config.sample_rate / self.spectrogram_config.hop_width
        )
        start_time = start_frame / frames_per_second
        end_time = end_frame / frames_per_second

        if self.debug and idx == 0:
            print(
                f"[DEBUG] Using {self.segment_frames} frames for {self.split}, starting from {start_time:.2f}s to {end_time:.2f}s"
            )

        # build events
        event_times, event_values = self.get_event_times_and_values(
            ns, start_time, end_time
        )
        frame_times = [
            i / self.codec.steps_per_second
            for i in range(int((end_time - start_time) * self.codec.steps_per_second))
        ]
        state = NoteEncodingState()

        events, _, _, state_events, _ = encode_and_index_events(
            state=state,
            event_times=event_times,
            event_values=event_values,
            encode_event_fn=note_event_data_to_events,
            codec=self.codec,
            frame_times=frame_times,
            encoding_state_to_events_fn=note_encoding_state_to_events,
        )

        # run-length encode shifts
        rle_fn = run_length_encode_shifts_fn(self.codec)
        features = {"targets": events}
        features = rle_fn(features)
        events = features["targets"]

        # clip + add EOS
        if len(events) >= MAX_LEN:
            events = events[: MAX_LEN - 1]

        # Build a single EOS-terminated sequence, then shift for input/target.
        base_ids = torch.tensor(events, dtype=torch.long) + NUM_SPECIAL_TOKENS
        seq = torch.cat([base_ids, torch.tensor([EOS_TOKEN], dtype=torch.long)])

        decoder_input_ids = seq[:-1]
        decoder_target_ids = seq[1:]

        if self.debug and idx == 0:

            def decode_seq(seq):
                out = []
                for e in seq[:50]:
                    idx = e.item()
                    if idx == 0:
                        out.append("PAD")
                    elif idx == 1:
                        out.append("EOS")
                    elif idx == 2:
                        out.append("UNK")
                    else:
                        out.append(
                            self.codec.decode_event_index(idx - NUM_SPECIAL_TOKENS)
                        )
                return out

            print(
                f"[DEBUG] Decoder ({self.split}) Input tokens: {decode_seq(decoder_input_ids)}"
            )
            print(
                f"[DEBUG] Decoder ({self.split}) Target tokens: {decode_seq(decoder_target_ids)}"
            )
            print(
                f"[DEBUG] Dataset ({self.split}) spec stats - mean: {spec.mean().item():.4f}, std: {spec.std().item():.4f}, min: {spec.min().item():.4f}, max: {spec.max().item():.4f}"
            )
            print(f"[DEBUG] Dataset ({self.split}) First input spec: {spec[0][:5]}")
            print(f"[DEBUG] Decoder ({self.split}) Input tensor: {decoder_input_ids}")
            print(f"[DEBUG] Decoder ({self.split}) Target tensor: {decoder_target_ids}")

            for name, tensor in [
                ("Input", decoder_input_ids),
                ("Target", decoder_target_ids),
            ]:
                if (tensor[:-1] == PAD_TOKEN).any() or (tensor[:-1] == UNK_TOKEN).any():
                    print(
                        f"[DEBUG][FATAL] PAD or UNK inside decoder_{name.lower()}_ids before last position!",
                        tensor,
                    )
                if (tensor[:-1] == EOS_TOKEN).any():
                    print(
                        f"[DEBUG][FATAL] EOS inside decoder_{name.lower()}_ids before last position!",
                        tensor,
                    )

        # labels and beat targets
        frame_labels = _build_frame_labels(ns, self.segment_frames, frames_per_second)

        # Framewise onset/offset/velocity labels aligned to the same window
        onset_labels, offset_labels = _build_onset_offset_labels(
            ns=ns, frames=self.segment_frames, fps=frames_per_second
        )
        velocity_bins = _build_velocity_bins(
            ns=ns, frames=self.segment_frames, fps=frames_per_second, num_bins=32
        )

        # ns here is already cropped and rebased to the window above
        beat_bounds, beat_targets = _build_beats_and_targets(
            ns=ns,
            segment_seconds=(end_time - start_time),
            frames=self.segment_frames,
            fps=frames_per_second,
            default_beats=16,
        )

        return {
            "spec": spec,  # [T, F]
            "decoder_input_ids": decoder_input_ids,  # [L]
            "decoder_target_ids": decoder_target_ids,  # [L]
            "beat_bounds": beat_bounds,  # [M, 2] in frame indices
            "frame_labels": frame_labels,  # [T, 88]
            "beat_targets": beat_targets,  # [M]
            "onset_labels": onset_labels,  # [T, 88] float {0,1}
            "offset_labels": offset_labels,  # [T, 88] float {0,1}
            "velocity_bins": velocity_bins,  # [T, 88] long in [-1..V-1]
        }

    def load_audio(self, path):
        import torchaudio

        waveform, sr = torchaudio.load(path)
        if sr != self.spectrogram_config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.spectrogram_config.sample_rate
            )
        return waveform.numpy().squeeze()

    def get_event_times_and_values(self, ns, start_time, end_time):
        """
        Extract BOTH onsets and offsets (+programs/drums) for the cropped window.
        Times are rebased so the segment starts at 0.0s.
        """
        # Crop to [start_time, end_time) then rebase to 0 for encoding.
        seg = note_seq.NoteSequence()
        seg.ticks_per_quarter = ns.ticks_per_quarter
        for n in ns.notes:
            if n.end_time > start_time and n.start_time < end_time:
                m = seg.notes.add()
                m.CopyFrom(n)
                m.start_time = max(0.0, n.start_time - start_time)
                m.end_time = max(0.0, n.end_time - start_time)
        # Build onsetoffset events (with programs / drums)
        event_times, event_values = note_sequence_to_onsets_and_offsets_and_programs(
            seg
        )
        return event_times, event_values


class MT3TemperatureSampler(Dataset):
    def __init__(
        self,
        dataset_map,
        spectrogram_config,
        codec,
        segment_seconds=10,
        temperature=0.3,
        split=None,
        debug=False,
    ):
        self.datasets = {}
        self.lengths = []

        for name, samples in dataset_map.items():
            ds = MT3Dataset(
                samples,
                spectrogram_config,
                codec,
                segment_seconds,
                split=split,
                debug=debug,
            )
            self.datasets[name] = ds
            self.lengths.append(len(ds))

        self.dataset_names = list(self.datasets.keys())
        self.split = split

        sizes = torch.tensor(self.lengths, dtype=torch.float)
        probs = sizes**temperature
        self.probs = probs / probs.sum()

    def __len__(self):
        return sum(len(ds) for ds in self.datasets.values())

    def __getitem__(self, _):
        dataset_name = random.choices(self.dataset_names, weights=self.probs, k=1)[0]
        ds = self.datasets[dataset_name]
        idx = random.randint(0, len(ds) - 1)
        return ds[idx]
