"""Generic isolated replay plumbing; no task controller or hardware."""

import pytest

from rosclaw.sim import cli, runtime


def test_recording_density_replayed_with_original_budget(loaded_backend):
    backend, model = loaded_backend
    receipt = backend.run_experiment(
        model.model_ref,
        controller={"hold": True},
        steps=30,
        budgets={"max_record_points": 3},
        audit=False,
    )
    trace = backend.store.get(receipt.trace_ref)
    assert trace["recording"]["max_record_points"] == 3
    report = backend.strict_replay(receipt.receipt_ref)
    assert report["mode"] == "RAW_EXACT"
    saved = backend.store.get(report["replay_ref"])
    assert saved["kind"] == "strict_replay_report"
    assert saved["receipt_ref"] == receipt.receipt_ref
    assert saved["trace_ref"] == receipt.trace_ref
    assert saved["initial_state_ref"] == receipt.initial_state_ref


def test_cli_strict_replay_forwards_actual_receipt(monkeypatch, tmp_path, capsys):
    seen = []

    class FakeRuntime:
        def __init__(self, root):
            pass

        def strict_replay(self, ref):
            seen.append(ref)
            return {"verified": True, "mode": "RAW_EXACT", "receipt_ref": ref}

    monkeypatch.setattr(runtime, "SimulationRuntime", FakeRuntime)
    assert (
        cli.dispatch_sim_argv(["sim", "--root", str(tmp_path), "strict-replay", "simexp_test"]) == 0
    )
    assert seen == ["simexp_test"]


@pytest.mark.parametrize("invalid", [True, 0, -1, 100001, "3"])
def test_replay_rejects_invalid_recording_budget_before_integration(
    loaded_backend, monkeypatch, invalid
):
    backend, model = loaded_backend
    receipt = backend.run_experiment(
        model.model_ref,
        controller={"hold": True},
        steps=3,
        audit=False,
    )
    trace = backend.store.get(receipt.trace_ref)
    trace["recording"] = {"max_record_points": invalid}
    changed_trace = backend.store.put("traces", trace)
    payload = backend.store.get(receipt.receipt_ref)
    payload["trace_ref"] = changed_trace
    changed_receipt = backend.store.put("experiments", payload)
    from rosclaw.sim.backends.mujoco import rollout

    monkeypatch.setattr(rollout, "run_rollout", lambda *a, **k: pytest.fail("integration invoked"))
    with pytest.raises(ValueError, match="REPLAY_RECORDING_INVALID"):
        backend.strict_replay(changed_receipt)


def test_native_runtime_client_and_registered_mcp_replay(loaded_backend, monkeypatch):
    import asyncio

    from rosclaw.mcp import tools
    from rosclaw.mcp.adapters.runtime_client import RuntimeClient

    backend, model = loaded_backend
    sim = runtime.SimulationRuntime.__new__(runtime.SimulationRuntime)
    sim._backend = backend
    client = RuntimeClient(
        project_root=backend.store.root.parent.parent,
        robot_id=None,
        runtime_profile={},
        daemon_client=object(),
    )
    client._sim_runtime = sim
    receipt = asyncio.run(
        client.sim_rollout(
            model.model_ref,
            {"hold": True},
            steps=30,
            max_record_points=3,
        )
    )
    monkeypatch.setattr(tools, "_client", lambda: client)
    report = asyncio.run(tools._sim_strict_replay(receipt["receipt_ref"]))
    assert report["verified"] is True and report["mode"] == "RAW_EXACT"
    assert backend.store.get(report["replay_ref"])["receipt_ref"] == receipt["receipt_ref"]


def test_fixture_mcp_does_not_fabricate_replay_success(tmp_path):
    import asyncio

    from rosclaw.mcp.adapters.runtime_client import RuntimeClient

    client = RuntimeClient(
        project_root=tmp_path,
        robot_id=None,
        runtime_profile={},
        fixture_mode=True,
        daemon_client=object(),
    )
    result = asyncio.run(client.sim_strict_replay("simexp_missing"))
    assert result["verified"] is False
    assert result["mode"] == "fixture"
    assert result["replay_status"] == "NOT_RUN"
