import copy

import numpy as np
import pytest

from rosclaw.growth.anchor_kernel import _hash
from rosclaw.growth.anchor_output_memory import AnchorOutputMemory

IDENTITY = "sha256:" + "a" * 64
PARENT = "sha256:" + "b" * 64
EVIDENCE = "sha256:" + "c" * 64


def make(x=((0.0, 0.0), (2.0, 0.0)), y=((7.0, 8.0), (9.0, 10.0)), *, accelerated=True):
    return AnchorOutputMemory(
        x,
        y,
        bandwidth=0.1,
        encoder_hash=IDENTITY,
        parent_policy_hash=PARENT,
        evidence_hash=EVIDENCE,
        accelerated=accelerated,
    )


def test_recorded_later_parent_predictions_not_first_warm_start_are_preserved():
    memory = make()
    for x, y in zip(memory.to_dict()["observations"], memory.to_dict()["predictions"], strict=True):
        assert np.array_equal(memory.blend(x, [-100.0, 100.0], encoder_hash=IDENTITY), y)
    assert np.allclose(memory.blend([100.0, 100.0], [1.0, 2.0], encoder_hash=IDENTITY), [1.0, 2.0])


def test_extension_keeps_existing_predictions_and_leaves_parent_memory_unchanged():
    old = make()
    before = old.to_dict()
    new = old.extend(
        [[3.0, 3.0]], [[11.0, 12.0]], parent_policy_hash=EVIDENCE, evidence_hash=PARENT
    )
    assert old.to_dict() == before
    assert new.to_dict()["predecessor_memory_hash"] == before["memory_hash"]
    assert AnchorOutputMemory.from_dict(new.to_dict()).to_dict() == new.to_dict()
    assert np.array_equal(new.blend([0.0, 0.0], [0.0, 0.0], encoder_hash=IDENTITY), [7.0, 8.0])
    assert np.array_equal(new.blend([3.0, 3.0], [0.0, 0.0], encoder_hash=IDENTITY), [11.0, 12.0])


def test_duplicate_signed_zero_state_cannot_hide_conflicting_predictions():
    with pytest.raises(ValueError, match="causal context"):
        make([[0.0, 0.0], [-0.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="causal context"):
        make().extend([[0.0, 0.0]], [[3.0, 4.0]], parent_policy_hash=PARENT, evidence_hash=EVIDENCE)


def test_numpy_and_acceleration_agree_even_at_reference_ties():
    accelerated = make()
    reference = make(accelerated=False)
    for x in ([1.0, 0.0], [0.0, 0.0], [2.0, 0.0], [1.001, 0.2]):
        assert np.array_equal(
            accelerated.blend(x, [0.1, 0.2], encoder_hash=IDENTITY),
            reference.blend(x, [0.1, 0.2], encoder_hash=IDENTITY),
        )


def test_copy_ownership_and_round_trip_are_immutable():
    x = np.zeros((1, 2))
    y = np.ones((1, 2))
    memory = make(x, y)
    x[:] = 12
    y[:] = 13
    assert np.array_equal(memory.blend([0.0, 0.0], [0.0, 0.0], encoder_hash=IDENTITY), [1.0, 1.0])
    blob = memory.to_dict()
    restored = AnchorOutputMemory.from_dict(blob)
    blob["predictions"][0][0] = 99
    assert np.array_equal(restored.blend([0.0, 0.0], [0.0, 0.0], encoder_hash=IDENTITY), [1.0, 1.0])
    with pytest.raises(AttributeError):
        memory.encoder_hash = PARENT


@pytest.mark.parametrize("mutation", ["encoder", "nan_input", "nan_proposal", "shape", "overflow"])
def test_unbound_or_invalid_predictions_are_rejected(mutation):
    memory = make()
    x = [0.0, 0.0]
    y = [0.0, 0.0]
    encoder = IDENTITY
    if mutation == "encoder":
        encoder = PARENT
    elif mutation == "nan_input":
        x = [float("nan"), 0.0]
    elif mutation == "nan_proposal":
        y = [float("nan"), 0.0]
    elif mutation == "shape":
        y = [0.0]
    else:
        y = [1e7, 0.0]
    with pytest.raises(ValueError):
        memory.blend(x, y, encoder_hash=encoder)


@pytest.mark.parametrize(
    "field", ["promotion_authorized", "hardware_authorized", "distributional_retention_guaranteed"]
)
def test_resealed_authority_changes_are_still_rejected(field):
    blob = copy.deepcopy(make().to_dict())
    blob[field] = True
    blob["memory_hash"] = _hash({k: v for k, v in blob.items() if k != "memory_hash"})
    with pytest.raises(ValueError):
        AnchorOutputMemory.from_dict(blob)


def test_extension_refuses_capacity_overflow_instead_of_dropping_old_states():
    memory = make(np.zeros((32768, 1)), np.zeros((32768, 1)), accelerated=False)
    with pytest.raises(ValueError):
        memory.extend([[1.0]], [[1.0]], parent_policy_hash=PARENT, evidence_hash=EVIDENCE)
