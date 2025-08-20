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

        return spec, decoder_input_ids, decoder_target_ids

    # --- helpers -------------------------------------------------------------

    def load_audio(self, path):
        import torchaudio

        waveform, sr = torchaudio.load(path)
        if sr != self.spectrogram_config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.spectrogram_config.sample_rate
            )
        return waveform.numpy().squeeze()

    def get_event_times_and_values(self, ns, start_time, end_time):
        filtered_notes = [
            note
            for note in ns.notes
            if note.end_time > start_time and note.start_time < end_time
        ]
        ns_segment = note_seq.NoteSequence()
        ns_segment.ticks_per_quarter = ns.ticks_per_quarter
        for note in filtered_notes:
            new_note = ns_segment.notes.add()
            new_note.CopyFrom(note)
            new_note.start_time = max(0.0, note.start_time - start_time)
            new_note.end_time = max(0.0, note.end_time - start_time)

        event_times = [note.start_time for note in ns_segment.notes]
        event_values = [
            NoteEventData(
                pitch=note.pitch,
                velocity=note.velocity,
                program=note.program,
                is_drum=note.is_drum,
            )
            for note in ns_segment.notes
        ]
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
