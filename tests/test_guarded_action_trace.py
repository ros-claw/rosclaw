import numpy as np
import pytest

from rosclaw.control.guarded_action_trace import GuardedActionTrace


def trace():
    return GuardedActionTrace(dimension=2, maximum_steps=3, context_hash="sha256:" + "a" * 64)


def record(t, **changes):
    values = {
        "step": t.count,
        "proposal": np.ones(2),
        "guarded": np.full(2, 0.2),
        "applied": np.full(2, 0.2),
        "parent": np.zeros(2),
        "used_parent": False,
        "reason": None,
        "intervention_fraction": 1.0,
    }
    values.update(changes)
    t.record(**values)


def test_raw_proposal_is_not_mislabelled_as_guarded_or_actual():
    t = trace()
    proposal = np.ones(2)
    record(t, proposal=proposal)
    proposal[:] = 123
    a = t.arrays()
    assert np.array_equal(a["actor_proposals"], [[1, 1]])
    assert np.array_equal(a["applied_actions"], [[0.2, 0.2]])
    assert a["actor_proposal_valid"].tolist() == [True]
    a["actor_proposals"][:] = 99
    assert t.arrays()["actor_proposals"][0, 0] == 1


def test_bypassed_inference_is_masked_not_a_zero_action_label():
    t = trace()
    record(
        t,
        proposal=None,
        guarded=np.zeros(2),
        applied=np.zeros(2),
        used_parent=True,
        reason="warmup_parent",
    )
    a = t.arrays()
    assert a["actor_proposal_valid"].tolist() == [False]
    assert a["fallback_reasons"].tolist() == ["warmup_parent"]


def test_rejected_raw_proposal_is_still_a_real_proposal():
    t = trace()
    record(
        t,
        guarded=np.zeros(2),
        applied=np.zeros(2),
        used_parent=True,
        reason="projection_ratio_exceeded",
    )
    a = t.arrays()
    assert a["actor_proposal_valid"][0]
    assert a["used_parent"][0]


def test_zero_intervention_logs_actual_parent_not_discarded_proposal():
    t = trace()
    record(t, intervention_fraction=0.0, applied=np.zeros(2))
    assert np.array_equal(t.arrays()["applied_actions"], [[0, 0]])
    assert np.array_equal(t.arrays()["guarded_actions"], [[0.2, 0.2]])


@pytest.mark.parametrize(
    "changes",
    [
        {"step": 1},
        {"proposal": None},
        {"proposal": np.array([np.nan, 0.0])},
        {"applied": np.ones(2)},
        {"reason": "unmarked"},
        {"used_parent": True},
        {"intervention_fraction": np.nan},
        {"parent": np.zeros(3)},
    ],
)
def test_invalid_or_fabricated_trace_is_rejected_atomically(changes):
    t = trace()
    with pytest.raises(ValueError):
        record(t, **changes)
    assert t.count == 0


def test_declared_horizon_is_terminal():
    t = trace()
    for _ in range(3):
        record(t)
    with pytest.raises(ValueError):
        record(t)
    assert t.count == 3
