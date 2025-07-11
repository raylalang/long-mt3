from dataclasses import dataclass
import note_seq
from .contrib.mt3 import event_codec

DEFAULT_STEPS_PER_SECOND = 100
DEFAULT_MAX_SHIFT_SECONDS = 10
DEFAULT_NUM_VELOCITY_BINS = 127


@dataclass
class VocabularyConfig:
    steps_per_second: int = DEFAULT_STEPS_PER_SECOND
    max_shift_seconds: int = DEFAULT_MAX_SHIFT_SECONDS
    num_velocity_bins: int = DEFAULT_NUM_VELOCITY_BINS


def build_codec(config: VocabularyConfig) -> event_codec.Codec:
    event_ranges = [
        event_codec.EventRange(
            "pitch", note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH
        ),
        event_codec.EventRange("velocity", 0, config.num_velocity_bins),
        event_codec.EventRange(
            "program", note_seq.MIN_MIDI_PROGRAM, note_seq.MAX_MIDI_PROGRAM
        ),
    ]

    codec = event_codec.Codec(
        max_shift_steps=config.steps_per_second * config.max_shift_seconds,
        steps_per_second=config.steps_per_second,
        event_ranges=event_ranges,
    )
    return codec
