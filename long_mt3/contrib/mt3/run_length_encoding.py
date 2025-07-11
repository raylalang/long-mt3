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
from typing import Sequence, Callable, Optional, Tuple, List, Any, TypeVar
import numpy as np
from . import event_codec

# These should be type variables, but unfortunately those are incompatible with
# dataclasses.
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

    # initialize encoding state
    init_encoding_state_fn: Callable[[], EncodingState]
    # convert EventData into zero or more events, updating encoding state
    encode_event_fn: Callable[
        [EncodingState, EventData, event_codec.Codec], Sequence[event_codec.Event]
    ]
    # convert encoding state (at beginning of segment) into events
    encoding_state_to_events_fn: Optional[
        Callable[[EncodingState], Sequence[event_codec.Event]]
    ]
    # create empty decoding state
    init_decoding_state_fn: Callable[[], DecodingState]
    # update decoding state when entering new segment
    begin_decoding_segment_fn: Callable[[DecodingState], None]
    # consume time and Event and update decoding state
    decode_event_fn: Callable[
        [DecodingState, float, event_codec.Event, event_codec.Codec], None
    ]
    # flush decoding state into result
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
    """Encode a sequence of timed events and index to audio frame times.

    Encodes time shifts as repeated single step shifts for later run length
    encoding.

    Optionally, also encodes a sequence of "state events", keeping track of the
    current encoding state at each audio frame. This can be used e.g. to prepend
    events representing the current state to a targets segment.

    Args:
      state: Initial event encoding state.
      event_times: Sequence of event times.
      event_values: Sequence of event values.
      encode_event_fn: Function that transforms event value into a sequence of one or more event_codec.Event objects.
      codec: An event_codec.Codec object that maps Event objects to indices.
      frame_times: Time for every audio frame.
      encoding_state_to_events_fn: Function that transforms encoding state into a sequence of one or more event_codec.Event objects.

    Returns:
      events: Encoded events and shifts.
      event_start_indices: Corresponding start event index for every audio frame.
      event_end_indices: Corresponding end event index for every audio frame.
      event_frame_indices: Indices mapping events to frame_times.
      state_events: Encoded state events if encoding_state_to_events_fn is provided, else empty list.
    """
    # Faithful implementation, adapted from original MT3
    num_frames = len(frame_times)
    events = []
    event_start_indices = np.zeros(num_frames, dtype=np.int32)
    event_end_indices = np.zeros(num_frames, dtype=np.int32)
    event_frame_indices = []
    state_events = []

    event_idx = 0
    num_events = len(event_times)
    last_time = 0.0

    for i, frame_time in enumerate(frame_times):
        # Indexing events that start in this frame
        while event_idx < num_events and event_times[event_idx] < frame_time:
            encoded_events = encode_event_fn(state, event_values[event_idx], codec)
            for ev in encoded_events:
                events.append(codec.encode_event(ev))
                event_frame_indices.append(i)
            event_idx += 1
        event_start_indices[i] = len(events)

        # Optionally encode state
        if encoding_state_to_events_fn is not None:
            for ev in encoding_state_to_events_fn(state):
                state_events.append(codec.encode_event(ev))

        event_end_indices[i] = len(events)

    # Convert all lists to numpy arrays
    events = np.array(events, dtype=np.int32)
    event_start_indices = np.array(event_start_indices, dtype=np.int32)
    event_end_indices = np.array(event_end_indices, dtype=np.int32)
    event_frame_indices = np.array(event_frame_indices, dtype=np.int32)
    state_events = (
        np.array(state_events, dtype=np.int32)
        if state_events
        else np.array([], dtype=np.int32)
    )

    return (
        events,
        event_start_indices,
        event_end_indices,
        event_frame_indices,
        state_events,
    )


def remove_redundant_state_changes(
    events: Sequence[int], state_change_tokens: Sequence[int]
) -> List[int]:
    output = []
    last_state_event = None
    for event in events:
        if event in state_change_tokens:
            if event != last_state_event:
                output.append(event)
                last_state_event = event
        else:
            output.append(event)
            last_state_event = None
    return output


def run_length_encode_shifts(
    events: Sequence[int], shift_token: int
) -> List[Tuple[int, int]]:
    result = []
    i = 0
    while i < len(events):
        event = events[i]
        if event == shift_token:
            run_length = 1
            while i + 1 < len(events) and events[i + 1] == shift_token:
                run_length += 1
                i += 1
            result.append((shift_token, run_length))
        else:
            result.append((event, 1))
        i += 1
    return result


def decode_events(
    encoded_events: Sequence[int], decode_event_fn: Callable[[int], Any]
) -> List[Any]:
    return [decode_event_fn(ev) for ev in encoded_events]


