import torch
from torch.utils.data import Dataset
import random
import numpy as np
import note_seq
from .contrib.mt3.spectrograms import compute_spectrogram
from .contrib.mt3.run_length_encoding import encode_and_index_events
from .contrib.mt3.event_codec import Event
from .contrib.mt3 import note_sequences

MAX_LEN = 1024

class MT3Dataset(Dataset):
    """
    PyTorch Dataset for MT3.
    Each sample yields (spectrogram, target_event_ids).
    """

    def __init__(self, data_list, spectrogram_config, codec, segment_frames):
        """
        Args:
            data_list: List of dicts, each with 'audio_path' and 'event_path' keys
            spectrogram_config: SpectrogramConfig dataclass
            codec: Vocabulary or codec object for encoding events
            segment_frames: chunking into fixed-length segments.
        """
        self.data_list = data_list
        self.spectrogram_config = spectrogram_config
        self.codec = codec
        self.segment_frames = segment_frames


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        audio = self.load_audio(sample["mix_audio_path"])

        # Compute spectrogram
        spec = compute_spectrogram(audio, self.spectrogram_config)
        n_frames = spec.shape[0]

        # Determine crop window
        if n_frames >= self.segment_frames:
            start_frame = np.random.randint(0, n_frames - self.segment_frames + 1)
            end_frame = start_frame + self.segment_frames
            spec = spec[start_frame:end_frame]
        else:
            start_frame = 0
            end_frame = n_frames
            pad_len = self.segment_frames - n_frames
            spec = np.pad(spec, ((0, pad_len), (0, 0)))

        spec = torch.from_numpy(spec).float()

        # Compute time window in seconds
        frames_per_second = self.spectrogram_config.sample_rate / self.spectrogram_config.hop_width
        start_time = start_frame / frames_per_second
        end_time = end_frame / frames_per_second

        # Load + trim events
        events = self.load_events(sample["midi_path"], start_time, end_time)
        event_ids = torch.tensor(events, dtype=torch.long)[:MAX_LEN]

        return spec, event_ids

    def load_audio(self, path):
        import torchaudio

        waveform, sr = torchaudio.load(path)
        if sr != self.spectrogram_config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.spectrogram_config.sample_rate
            )
        return waveform.numpy().squeeze()

    def load_events(self, midi_path, start_time, end_time):
        ns = note_seq.midi_file_to_note_sequence(midi_path)
        note_sequences.validate_note_sequence(ns)
        ns = note_sequences.trim_overlapping_notes(ns)

        # Filter notes based on start_time and end_time
        filtered_notes = [
            note for note in ns.notes
            if start_time <= note.start_time < end_time
        ]
        
        # Replace notes safely
        del ns.notes[:]  # Clears all notes
        for note in filtered_notes:
            ns.notes.add().CopyFrom(note)

        # Prepare encoding
        ns_total_time = end_time - start_time
        event_times = [note.start_time - start_time for note in ns.notes]
        event_values = [(note.pitch, note.velocity) for note in ns.notes]

        def encode_fn(state, value, codec):
            pitch, velocity = value
            return [Event("pitch", pitch), Event("velocity", velocity)]

        frame_times = [
            i / self.codec.steps_per_second
            for i in range(int(ns_total_time * self.codec.steps_per_second))
        ]

        events, *_ = encode_and_index_events(
            state=None,
            event_times=event_times,
            event_values=event_values,
            encode_event_fn=encode_fn,
            codec=self.codec,
            frame_times=frame_times,
        )

        return events




class MT3TemperatureSampler(Dataset):
    def __init__(self, dataset_map, spectrogram_config, codec, segment_seconds=10, temperature=0.3):
        self.datasets = {}
        self.lengths = []

        frames_per_second = spectrogram_config.sample_rate / spectrogram_config.hop_width
        segment_frames = int(segment_seconds * frames_per_second)

        for name, samples in dataset_map.items():
            ds = MT3Dataset(samples, spectrogram_config, codec, segment_frames)
            self.datasets[name] = ds
            self.lengths.append(len(ds))

        self.dataset_names = list(self.datasets.keys())

        # compute temperature-scaled probabilities
        sizes = torch.tensor(self.lengths, dtype=torch.float)
        probs = sizes ** temperature
        self.probs = probs / probs.sum()

    def __len__(self):
        return sum(len(ds) for ds in self.datasets.values())

    def __getitem__(self, _):
        dataset_name = random.choices(self.dataset_names, weights=self.probs, k=1)[0]
        ds = self.datasets[dataset_name]
        idx = random.randint(0, len(ds) - 1)
        return ds[idx]

