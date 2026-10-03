"""Generic CLI recording-budget wiring, no physics rollouts."""

import pytest

from rosclaw.sim import cli, runtime


def test_record_points_forwarded_to_native_runtime(monkeypatch, tmp_path, capsys):
    seen = []

    class FakeRuntime:
        def __init__(self, root):
            pass

        def rollout(self, ref, **kwargs):
            seen.append(kwargs)
            return {"ok": True}

    monkeypatch.setattr(runtime, "SimulationRuntime", FakeRuntime)
    assert (
        cli.dispatch_sim_argv(
            [
                "sim",
                "--root",
                str(tmp_path),
                "rollout",
                "simmdl_dummy",
                "--controller",
                '{"hold":true}',
                "--steps",
                "2",
                "--max-record-points",
                "6000",
            ]
        )
        == 0
    )
    assert seen[0]["budgets"] == {"max_record_points": 6000}


@pytest.mark.parametrize("value", ["0", "-1", "100001", "abc", "1.5"])
def test_invalid_record_point_budget_rejected_before_runtime(monkeypatch, value, capsys):
    monkeypatch.setattr(runtime, "SimulationRuntime", lambda root: pytest.fail("runtime initialized"))
    with pytest.raises(SystemExit) as exc:
        cli.dispatch_sim_argv(
            [
                "sim",
                "rollout",
                "simmdl_dummy",
                "--controller",
                '{"hold":true}',
                "--steps",
                "2",
                "--max-record-points",
                value,
            ]
        )
    assert exc.value.code == 2
