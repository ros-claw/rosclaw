from __future__ import annotations

import pytest

from rosclaw.continual.teacher_prior import ConditionalTeacherPriorContract
from tests.continual.helpers import digest


def _contract() -> ConditionalTeacherPriorContract:
    return ConditionalTeacherPriorContract(
        prior_id="humanoid.motion.teacher.v2",
        artifact_hash=digest("teacher"),
        body_hash=digest("body"),
        observation_names=("gravity", "joint_position"),
        output_names=("target_position", "target_velocity"),
        condition_vocabulary={
            "task": ("ready", "save", "recovery"),
            "region": ("upper_left", "upper_right"),
        },
    )


def test_conditional_teacher_query_is_complete_and_content_addressed() -> None:
    contract = _contract()

    query = contract.query({"task": "save", "region": "upper_left"})

    assert query.prior_contract_hash == contract.contract_hash
    assert query.condition_values == {"region": "upper_left", "task": "save"}
    assert query.query_hash.startswith("sha256:")


def test_teacher_query_fails_closed_on_average_or_missing_condition() -> None:
    contract = _contract()

    with pytest.raises(ValueError, match="every condition"):
        contract.query({"task": "save"})
    with pytest.raises(ValueError, match="frozen vocabulary"):
        contract.query({"task": "save", "region": "average"})


def test_teacher_cannot_become_a_deployed_actor_dependency() -> None:
    with pytest.raises(ValueError, match="train-only"):
        ConditionalTeacherPriorContract(
            prior_id="unsafe.teacher",
            artifact_hash=digest("teacher"),
            body_hash=digest("body"),
            observation_names=("state",),
            output_names=("action",),
            condition_vocabulary={"task": ("save",)},
            deployed_actor_depends_on_teacher=True,
        )
