import numpy as np
import pytest

from rosclaw.growth.event_credit_partitions import event_credit_partitions
from rosclaw.growth.sample_weighting import balanced_partition_weights


def labels(groups=None, frames=None, events=None, lead=2, after=2):
    return event_credit_partitions(
        np.repeat([0, 1], 8) if groups is None else groups,
        np.tile(np.arange(8), 2) if frames is None else frames,
        np.asarray([4, -1]) if events is None else events,
        lead_frames=lead,
        after_frames=after,
    )


def test_first_event_action_is_in_lead_not_future_observation_phase():
    result = labels()
    assert result.tolist() == [0, 0, 1, 1, 1, 2, 2, 3] + [4] * 8
    assert result.flags.writeable is False
    weights = balanced_partition_weights(result)
    assert np.all(weights > 0)
    assert len(weights) == 16
    for label in range(5):
        assert np.isclose(weights[result == label].sum(), 16 / 5)


@pytest.mark.parametrize(
    "fault", ["missing", "reorder", "group", "event", "float", "boolean", "lead", "after"]
)
def test_invalid_or_partial_labels_rejected(fault):
    groups = np.repeat([0, 1], 8)
    frames = np.tile(np.arange(8), 2)
    events = np.asarray([4, -1])
    lead, after = 2, 2
    if fault == "missing":
        frames[3] = 4
    elif fault == "reorder":
        groups[[1, 9]] = groups[[9, 1]]
    elif fault == "group":
        groups[7] = 2
    elif fault == "event":
        events[0] = -2
    elif fault == "float":
        events = events.astype(float)
    elif fault == "boolean":
        groups = groups.astype(bool)
    elif fault == "lead":
        lead = True
    else:
        after = 1001
    with pytest.raises(ValueError):
        labels(groups, frames, events, lead, after)


def test_unseen_or_absent_events_do_not_drop_or_relabel_frames():
    assert labels(events=[-1, -1]).tolist() == [4] * 16
    assert labels(events=[20, 20]).tolist() == [0] * 16
    assert labels(events=[0, 0], lead=0, after=0).tolist() == [1] + [3] * 7 + [1] + [3] * 7
