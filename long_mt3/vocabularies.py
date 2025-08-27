from dataclasses import dataclass
import math
import note_seq
from typing import Optional, List

from .contrib.mt3 import event_codec

# Constants for special tokens
PAD_TOKEN = 0
EOS_TOKEN = 1
UNK_TOKEN = 2
NUM_SPECIAL_TOKENS = 3

# Defaults
DEFAULT_STEPS_PER_SECOND = 100
DEFAULT_MAX_SHIFT_SECONDS = 10
DEFAULT_NUM_VELOCITY_BINS = 127


@dataclass
class VocabularyConfig:
    steps_per_second: int = DEFAULT_STEPS_PER_SECOND
    max_shift_seconds: int = DEFAULT_MAX_SHIFT_SECONDS
    num_velocity_bins: int = DEFAULT_NUM_VELOCITY_BINS


def build_codec(
    config: VocabularyConfig, event_types: Optional[List[str]] = None
) -> event_codec.Codec:
    """
    Build the event codec, optionally selecting which event types to include.

    Args:
        config: VocabularyConfig with settings like steps_per_second and num_velocity_bins.
        event_types: List of event types to include in the vocabulary.
                     Valid options: "program", "pitch", "velocity", "drum", "tie", "velocity".

    Returns:
        event_codec.Codec instance defining the vocabulary.
    """
    if event_types is None:
        event_types = ["pitch", "program", "drum", "tie", "velocity"]

    event_ranges = [
        event_codec.EventRange(
            "pitch", note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH
        ),
        event_codec.EventRange("velocity", 0, config.num_velocity_bins),
        event_codec.EventRange("tie", 0, 0),
        event_codec.EventRange(
            "program", note_seq.MIN_MIDI_PROGRAM, note_seq.MAX_MIDI_PROGRAM
        ),
        event_codec.EventRange(
            "drum", note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH
        ),
    ]

    return event_codec.Codec(
        max_shift_steps=int(config.steps_per_second * config.max_shift_seconds),
        steps_per_second=config.steps_per_second,
        event_ranges=event_ranges,
    )


def num_velocity_bins_from_codec(codec: event_codec.Codec) -> int:
    lo, hi = codec.event_type_range("velocity")
    return hi - lo


def velocity_to_bin(velocity: int, num_velocity_bins: int) -> int:
    if velocity == 0:
        return 0
    return math.ceil(num_velocity_bins * velocity / note_seq.MAX_MIDI_VELOCITY)


def bin_to_velocity(velocity_bin: int, num_velocity_bins: int) -> int:
    if velocity_bin == 0:
        return 0
    return int(note_seq.MAX_MIDI_VELOCITY * velocity_bin / num_velocity_bins)


def get_special_tokens() -> dict:
    return {
        "pad_token": PAD_TOKEN,
        "eos_token": EOS_TOKEN,
        "unk_token": UNK_TOKEN,
        "num_special_tokens": NUM_SPECIAL_TOKENS,
    }
