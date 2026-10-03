"""Resource rejection happens before entering noninterruptible native batch execution."""

import pytest

from rosclaw.sim.backends.mujoco import batch
from rosclaw.sim.backends.mujoco.backend import MujocoBackend

XML = '<mujoco><option timestep=".002"/><worldbody><body><joint/><geom size=".1" mass="1"/></body></worldbody></mujoco>'


@pytest.mark.parametrize(
    "budgets,branches,steps",
    [
        ({"max_duration_s": 0.003}, 1, 2),
        ({"max_branch_count": 2}, 3, 1),
    ],
)
def test_excess_batch_work_is_rejected_before_native_call(
    tmp_path, monkeypatch, budgets, branches, steps
):
    backend = MujocoBackend(tmp_path)
    ref = backend.load_model_xml(XML, source={"kind": "fixture"}).model_ref

    def forbidden(*args, **kwargs):
        pytest.fail("over-budget request entered native batch")

    monkeypatch.setattr(batch, "run_batch", forbidden)
    with pytest.raises(ValueError, match="SIM_BUDGET_EXCEEDED"):
        backend.rollout_batch(
            [ref] * branches, controller={"hold": True}, steps=steps, budgets=budgets
        )
