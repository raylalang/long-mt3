# Copyright 2024 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported to PyTorch by Raynaldi Lalang, 2025.

import dataclasses
from typing import (
    Any,
    Callable,
    Mapping,
    MutableMapping,
    Tuple,
    Optional,
    Sequence,
    TypeVar,
    List,
)
import numpy as np

from . import event_codec

Event = event_codec.Event

# Type aliases
EventData = Any
EncodingState = Any
DecodingState = Any
DecodeResult = Any

T = TypeVar("T", bound=EventData)
ES = TypeVar("ES", bound=EncodingState)
DS = TypeVar("DS", bound=DecodingState)


@dataclasses.dataclass
class EventEncodingSpec:
    """Spec for encoding events."""

    init_encoding_state_fn: Callable[[], EncodingState]
    encode_event_fn: Callable[
        [EncodingState, EventData, event_codec.Codec], Sequence[event_codec.Event]
    ]
    encoding_state_to_events_fn: Optional[
        Callable[[EncodingState], Sequence[event_codec.Event]]
    ]
    init_decoding_state_fn: Callable[[], DecodingState]
    begin_decoding_segment_fn: Callable[[DecodingState], None]
    decode_event_fn: Callable[
        [DecodingState, float, event_codec.Event, event_codec.Codec], None
    ]
    flush_decoding_state_fn: Callable[[DecodingState], DecodeResult]


def encode_and_index_events(
    state: ES,
    event_times: Sequence[float],
    event_values: Sequence[T],
    encode_event_fn: Callable[[ES, T, event_codec.Codec], Sequence[event_codec.Event]],
    codec: event_codec.Codec,
    frame_times: Sequence[float],
    encoding_state_to_events_fn: Optional[
        Callable[[ES], Sequence[event_codec.Event]]
    ] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Faithful numpy/PyTorch port of MT3 encode_and_index_events."""

    indices = np.argsort(event_times, kind="stable")
    event_steps = [round(event_times[i] * codec.steps_per_second) for i in indices]
    event_values = [event_values[i] for i in indices]

    events = []
    state_events = []
    event_start_indices = []
    state_event_indices = []

    cur_step = 0
    cur_event_idx = 0
    cur_state_event_idx = 0

    def fill_event_start_indices_to_cur_step():
        while (
            len(event_start_indices) < len(frame_times)
            and frame_times[len(event_start_indices)]
            < cur_step / codec.steps_per_second
        ):
            event_start_indices.append(cur_event_idx)
            state_event_indices.append(cur_state_event_idx)

    for event_step, event_value in zip(event_steps, event_values):
        while event_step > cur_step:
            events.append(codec.encode_event(Event(type="shift", value=1)))
            cur_step += 1
            fill_event_start_indices_to_cur_step()
            cur_event_idx = len(events)
            cur_state_event_idx = len(state_events)
        if encoding_state_to_events_fn:
            for e in encoding_state_to_events_fn(state):
                state_events.append(codec.encode_event(e))
        for e in encode_event_fn(state, event_value, codec):
            events.append(codec.encode_event(e))

    while cur_step / codec.steps_per_second <= frame_times[-1]:
        events.append(codec.encode_event(Event(type="shift", value=1)))
        cur_step += 1
        fill_event_start_indices_to_cur_step()
        cur_event_idx = len(events)

    event_end_indices = event_start_indices[1:] + [len(events)]

    events = np.array(events, dtype=np.int32)
    state_events = np.array(state_events, dtype=np.int32)
    event_start_indices = np.array(event_start_indices, dtype=np.int32)
    event_end_indices = np.array(event_end_indices, dtype=np.int32)
    state_event_indices = np.array(state_event_indices, dtype=np.int32)

    return (
        events,
        event_start_indices,
        event_end_indices,
        state_events,
        state_event_indices,
    )


def extract_target_sequence_with_indices(
    features: Mapping[str, Any], state_events_end_token: Optional[int] = None
) -> Mapping[str, Any]:
    """Extract target sequence corresponding to audio token segment."""
    target_start_idx = features["input_event_start_indices"][0]
    target_end_idx = features["input_event_end_indices"][-1]
    features["targets"] = features["targets"][target_start_idx:target_end_idx]

    if state_events_end_token is not None:
        state_event_start_idx = features["input_state_event_indices"][0]
        state_event_end_idx = state_event_start_idx + 1
        state_events = features["state_events"]
        while state_events[state_event_end_idx - 1] != state_events_end_token:
            state_event_end_idx += 1
        features["targets"] = np.concatenate(
            [
                state_events[state_event_start_idx:state_event_end_idx],
                features["targets"],
            ],
            axis=0,
        )
    return features


def remove_redundant_state_changes_fn(
    codec: event_codec.Codec,
    feature_key: str = "targets",
    state_change_event_types: Sequence[str] = (),
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Return preprocessing function that removes redundant state change events."""

    state_change_event_ranges = [
        codec.event_type_range(event_type) for event_type in state_change_event_types
    ]

    def remove_redundant_state_changes(
        features: MutableMapping[str, Any],
    ) -> Mapping[str, Any]:
        """Remove redundant tokens e.g. duplicate velocity changes from sequence."""
        events = features[feature_key]
        current_state = [None] * len(state_change_event_ranges)
        output = []
        for event in events:
            is_redundant = False
            for i, (min_index, max_index) in enumerate(state_change_event_ranges):
                if min_index <= event <= max_index:
                    if current_state[i] == event:
                        is_redundant = True
                    current_state[i] = event
            if not is_redundant:
                output.append(event)
        features[feature_key] = np.array(output, dtype=np.int32)
        return features

    return remove_redundant_state_changes


def run_length_encode_shifts_fn(
    codec: event_codec.Codec, feature_key: str = "targets"
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Return a function that run-length encodes shifts for a given codec."""

    def run_length_encode_shifts(
        features: MutableMapping[str, Any],
    ) -> Mapping[str, Any]:
        events = features[feature_key]
        output = []
        i = 0
        N = len(events)
        while i < N:
            event = codec.decode_event_index(events[i])
            if event.type == "shift":
                shift_steps = event.value
                j = i + 1
                while j < N:
                    next_event = codec.decode_event_index(events[j])
                    if next_event.type == "shift":
                        shift_steps += next_event.value
                        j += 1
                    else:
                        break
                while shift_steps > 0:
                    out_steps = min(codec.max_shift_steps, shift_steps)
                    output.append(
                        codec.encode_event(Event(type="shift", value=out_steps))
                    )
                    shift_steps -= out_steps
                i = j
            else:
                output.append(events[i])
                i += 1
        features[feature_key] = np.array(output, dtype=np.int32)
        return features

    return run_length_encode_shifts


def merge_run_length_encoded_targets(
    targets: np.ndarray, codec: event_codec.Codec
) -> np.ndarray:
    """Merge multiple tracks of target events into a single stream."""

    num_tracks = targets.shape[0]
    targets_length = targets.shape[1]

    current_step = 0
    current_offsets = np.zeros(num_tracks, dtype=np.int32)
    output = []
    done = False

    while not done:
        next_step = codec.max_shift_steps + 1
        next_track = -1
        for i in range(num_tracks):
            if (
                current_offsets[i] == targets_length
                or targets[i][current_offsets[i]] == 0
            ):
                continue
            if not codec.is_shift_event_index(targets[i][current_offsets[i]]):
                next_step = 0
                next_track = i
            elif targets[i][current_offsets[i]] < next_step:
                next_step = targets[i][current_offsets[i]]
                next_track = i

        if next_track == -1:
            done = True
            break

        if next_step == current_step and next_step > 0:
            start_offset = current_offsets[next_track] + 1
        else:
            start_offset = current_offsets[next_track]

        end_offset = start_offset + 1
        while end_offset < targets_length and not codec.is_shift_event_index(
            targets[next_track][end_offset]
        ):
            end_offset += 1
        output.extend(targets[next_track][start_offset:end_offset])

        current_step = next_step
        current_offsets[next_track] = end_offset

    return np.array(output, dtype=np.int32)


def decode_events(
    state: DS,
    tokens: np.ndarray,
    start_time: float,
    max_time: Optional[float],
    codec: event_codec.Codec,
    decode_event_fn: Callable[[DS, float, event_codec.Event, event_codec.Codec], None],
) -> Tuple[int, int]:
    """Decode a series of tokens, maintaining a decoding state object."""

    invalid_events = 0
    dropped_events = 0
    cur_steps = 0
    cur_time = start_time
    token_idx = 0
    for token_idx, token in enumerate(tokens):
        try:
            event = codec.decode_event_index(token)
        except ValueError:
            invalid_events += 1
            continue
        if event.type == "shift":
            cur_steps += event.value
            cur_time = start_time + cur_steps / codec.steps_per_second
            if max_time is not None and cur_time > max_time:
                dropped_events = len(tokens) - token_idx
                break
        else:
            cur_steps = 0
            try:
                decode_event_fn(state, cur_time, event, codec)
            except ValueError:
                invalid_events += 1
                continue
    return invalid_events, dropped_events
