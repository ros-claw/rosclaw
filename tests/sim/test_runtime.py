"""SimulationRuntime Agent 面编排测试（PR-MH6，ADR-0014，规格 §25/§28，红→绿）。

runtime 返回全部 JSON 友好的 dict（契约 canonical form）。
"""

from __future__ import annotations

import pytest

from rosclaw.sim.runtime import SimulationRuntime


@pytest.fixture
def runtime(tiny_task_root):
    return SimulationRuntime(tiny_task_root)


def test_get_capabilities(runtime) -> None:
    caps = runtime.get_capabilities()
    assert caps["backend"] == "mujoco"
    assert caps["capabilities"]["mjspec"] is True
    assert caps["usable_for_real_execution"] is False


def test_full_agent_workflow(runtime) -> None:
    """inspect → patch → snapshot → rollout → observe → audit → compare：
    Agent 标准实验链，几次结构化调用完成（规格 §21）。"""
    loaded = runtime.load_model("arm.xml")
    model_ref = loaded["model_ref"]
    assert loaded["body_description"]["nq"] == 2

    inspected = runtime.inspect_model(model_ref)
    assert inspected["nq"] == 2
    assert "关节" in inspected["summary"]

    patched = runtime.patch_model(
        model_ref,
        [
            {
                "op": "set",
                "target": {"type": "actuator", "name": "shoulder_servo"},
                "field": "kp",
                "value": 100.0,
            }
        ],
    )
    assert patched["ok"] is True
    new_ref = patched["new_model_ref"]

    snap = runtime.snapshot(new_ref)
    assert snap["state_ref"].startswith("simsta_")

    receipt = runtime.rollout(new_ref, controller={"position_targets": [0.3, 0.1]}, duration_s=0.2)
    assert receipt["trust_level"] == "SIMULATED"
    assert receipt["usable_for_real_execution"] is False
    assert receipt["trace_ref"].startswith("simtrc_")

    observed = runtime.observe(new_ref, receipt["final_state_ref"], ["joint_positions"])
    assert observed["values"]["joint_positions"][0] > 0.02

    audited = runtime.audit(new_ref, trace_ref=receipt["trace_ref"])
    assert audited["status"] in ("PASS", "WARN")
    assert audited["audit_ref"].startswith("simadt_")

    receipt2 = runtime.rollout(new_ref, controller={"hold": True}, duration_s=0.1)
    compared = runtime.compare([receipt["receipt_ref"], receipt2["receipt_ref"]])
    assert compared["best_ref"]
    assert len(compared["metric_table"]) == 2


def test_runtime_errors_are_structured(runtime) -> None:
    with pytest.raises(ValueError, match="MODEL_NOT_FOUND"):
        runtime.load_model("ghost.xml")
    loaded = runtime.load_model("arm.xml")
    with pytest.raises(ValueError, match="MODEL_FIELD_UNSUPPORTED"):
        runtime.patch_model(
            loaded["model_ref"],
            [{"op": "add", "target": {"type": "geom"}, "field": "x", "value": 1}],
        )


def test_render_produces_artifact(runtime) -> None:
    loaded = runtime.load_model("arm.xml")
    receipt = runtime.rollout(
        loaded["model_ref"], controller={"position_targets": [0.3, 0.0]}, duration_s=0.1
    )
    try:
        result = runtime.render(receipt["trace_ref"], width=160, height=120, max_frames=4)
    except ValueError as exc:
        if "SIM_RENDER_UNAVAILABLE" in str(exc):
            pytest.skip(f"GL backend unavailable on this machine: {exc}")
        raise
    assert result["artifact_ref"].startswith("simrnd_")
    assert result["frames"] >= 2
    assert result["renderer_backend"]  # 诚实记录实际后端


def test_runtime_client_delegates_to_simulation_runtime(tiny_task_root, monkeypatch) -> None:
    """规格 §28：RuntimeClient.sim_* 调用 SimulationRuntime，不是自己实现。"""
    import asyncio
    import shutil

    work = tiny_task_root / "simwork"
    work.mkdir()
    shutil.copy(tiny_task_root / "arm.xml", work / "arm.xml")
    monkeypatch.setenv("ROSCLAW_SIM_TASK_ROOT", str(work))

    from rosclaw.mcp.adapters.runtime_client import RuntimeClient

    client = RuntimeClient(project_root=tiny_task_root, robot_id=None, runtime_profile={})
    caps = asyncio.run(client.sim_get_capabilities())
    assert caps["backend"] == "mujoco"
    assert caps["usable_for_real_execution"] is False

    loaded = asyncio.run(client.sim_load_model("arm.xml"))
    receipt = asyncio.run(client.sim_rollout(loaded["model_ref"], {"hold": True}, duration_s=0.05))
    assert receipt["trust_level"] == "SIMULATED"
    assert receipt["usable_for_real_execution"] is False


def test_runtime_client_fixture_mode_has_no_physics(tmp_path) -> None:
    import asyncio

    from rosclaw.mcp.adapters.runtime_client import RuntimeClient

    client = RuntimeClient(
        project_root=tmp_path, robot_id=None, runtime_profile={}, fixture_mode=True
    )
    payload = asyncio.run(client.sim_rollout("simmdl_x", {"hold": True}, steps=10))
    assert payload["mode"] == "fixture"
    assert payload["usable_for_real_execution"] is False
