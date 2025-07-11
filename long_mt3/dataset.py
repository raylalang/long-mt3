import torch
from torch.utils.data import Dataset
import note_seq
from .spectrograms import compute_spectrogram
from .run_length_encoding import encode_and_index_events
from .event_codec import Event
from . import note_sequences


class MT3Dataset(Dataset):
    """
    PyTorch Dataset for MT3.
    Each sample yields (spectrogram, target_event_ids).
    """

    def __init__(self, data_list, spectrogram_config, codec, segment_frames=None):
        """
        Args:
            data_list: List of dicts, each with 'audio_path' and 'event_path' keys
            spectrogram_config: Your SpectrogramConfig dataclass
            codec: Your Vocabulary or codec object for encoding events
            segment_frames: Optional, if you want to chunk audio
        """
        self.data_list = data_list
        self.spectrogram_config = spectrogram_config
        self.codec = codec
        self.segment_frames = segment_frames

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        audio = self.load_audio(sample["audio_path"])
        events = self.load_events(sample["midi_path"])

        spec = compute_spectrogram(audio, self.spectrogram_config)
        if self.segment_frames is not None and spec.shape[0] > self.segment_frames:
            spec = spec[: self.segment_frames, :]
        spec = torch.from_numpy(spec).float()

        event_ids = torch.tensor(events, dtype=torch.long)

        return spec, event_ids

    def load_audio(self, path):
        import torchaudio

        waveform, sr = torchaudio.load(path)
        if sr != self.spectrogram_config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.spectrogram_config.sample_rate
            )
        return waveform.numpy().squeeze()

    def load_events(self, midi_path):
        ns = note_seq.midi_file_to_note_sequence(midi_path)
        ns.notes.sort(key=lambda note: note.start_time)
        note_sequences.validate_note_sequence(ns)
        ns = note_sequences.trim_overlapping_notes(ns)

        event_times = [note.start_time for note in ns.notes]
        event_values = [(note.pitch, note.velocity) for note in ns.notes]

        def encode_fn(state, value, codec):
            pitch, velocity = value
            return [Event("pitch", pitch), Event("velocity", velocity)]

        frame_times = [
            i / self.codec.steps_per_second
            for i in range(int(ns.total_time * self.codec.steps_per_second))
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

