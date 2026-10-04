"""Offline temporal credit partitions, never future inputs to a live actor.

Every row is retained. Measured terminal event times may label training rows;
the resulting labels are for an explicitly declared loss objective only.
They neither change rewards nor prove which actions caused an outcome.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def event_credit_partitions(
    trajectory_index: Any,
    frame_index: Any,
    measured_event_frame: Any,
    *,
    lead_frames: int,
    after_frames: int,
) -> np.ndarray[Any, Any]:
    """Ordered labels: approach, lead+event, early-after, late-after, no-event.

    Events are one integer per complete trajectory; -1 explicitly represents
    an absent event. The caller must independently bind and audit event data.
    No rows are dropped when an event is absent or outside recorded frames.
    This function is not a policy observation, event predictor or approval.
    """
    groups, frames, events = [
        np.asarray(v) for v in (trajectory_index, frame_index, measured_event_frame)
    ]
    if (
        groups.ndim != 1
        or not 4 <= len(groups) <= 200000
        or frames.shape != groups.shape
        or events.ndim != 1
        or not 1 <= len(events) <= 740
        or any(v.dtype.kind not in "iu" for v in (groups, frames, events))
        or np.any((groups < 0) | (groups >= len(events)))
        or np.any((frames < 0) | (frames > 1_000_000))
        or np.any((events < -1) | (events > 1_000_000))
        or not np.array_equal(np.unique(groups), np.arange(len(events)))
        or np.any(np.diff(groups.astype(np.int64)) < 0)
        or type(lead_frames) is not int
        or not 0 <= lead_frames <= 1000
        or type(after_frames) is not int
        or not 0 <= after_frames <= 1000
    ):
        raise ValueError("complete ordered bounded offline event labels required")
    for group in range(len(events)):
        if np.any(np.diff(frames[groups == group].astype(np.int64)) != 1):
            raise ValueError("all consecutive recorded frames required; no row omission")
    selected = events.astype(np.int64)[groups]
    time = frames.astype(np.int64)
    result = np.full(len(groups), 3, dtype=np.int64)
    result[time < selected - lead_frames] = 0
    result[(time >= selected - lead_frames) & (time <= selected)] = 1
    result[(time > selected) & (time <= selected + after_frames)] = 2
    result[selected == -1] = 4
    result.flags.writeable = False
    return result
