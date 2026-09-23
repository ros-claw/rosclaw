"""MH26 Release/Platform Qualification 测试（讨论总纲 §47-§52，红→绿）。

1. **Representative Robot Matrix（§51/G48）**：四类真实复杂度
   （fixed manipulator=xarm7 / mobile manipulator=stretch_3 /
   quadruped=go2 / humanoid=g1）× load/inspect/state v2/audit/
   rollout/patch lineage/mjz/strict replay——Harness 不只是
   tiny fixture / UR5e 形态。
2. **Performance Baseline（§50）**：按 robot scale 记录关键操作
   耗时（load/compile/snapshot/restore/rollout/audit）——记录不
   硬 Gate（不拿 tiny fixture 代表全部）。
3. **Clean Wheel Install（§49/G46）**：build wheel → 干净 venv →
   pip install → `rosclaw sim` 冒烟——不永远在 editable checkout
   里测。
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("mujoco_menagerie", reason="mujoco-menagerie not installed")

#: 四类代表（§51）。
REPRESENTATIVE = {
    "fixed_manipulator": "ufactory_xarm7",
    "mobile_manipulator": "hello_robot_stretch_3",
    "quadruped": "unitree_go2",
    "humanoid": "unitree_g1",
}


@pytest.fixture
def backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    return MujocoBackend(tmp_path)


@pytest.mark.parametrize("category,model_name", sorted(REPRESENTATIVE.items()))
def test_representative_robot_full_chain(backend, category, model_name) -> None:
    """§51 全链：load → inspect → state v2 快照/恢复 → rollout →
    audit → patch 血缘 → mjz 导出/重建 → strict replay。"""
    ref = backend.load_menagerie(model_name)
    manifest = backend.store.get(ref.model_ref)
    assert manifest["source"]["kind"] == "menagerie"

    # inspect：编译真相。
    inspected = backend.inspect_model(ref.model_ref)
    assert inspected.nq > 0 and inspected.nv > 0

    # state v2：快照→恢复精确往返（FULL_INTEGRATION）。
    from rosclaw.sim.backends.mujoco import state_v2

    manifest = backend._manifest(ref.model_ref)
    spec = backend._spec_from_manifest(manifest)
    model, _ = backend._compile_smoke(spec)
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(10):
        mujoco.mj_step(model, data)
    vector, spec_value = (
        state_v2.capture_state_v2(model, data, fidelity=state_v2.FIDITY_FULL_INTEGRATION)
        if hasattr(state_v2, "FIDITY_FULL_INTEGRATION")
        else state_v2.capture_state_v2(model, data, fidelity=state_v2.FIDELITY_FULL_INTEGRATION)
    )
    data2 = mujoco.MjData(model)
    state_v2.apply_state_v2(model, data2, vector, spec_value)
    import numpy as np

    assert np.array_equal(np.asarray(data.qpos), np.asarray(data2.qpos))

    # rollout + audit（run_experiment 的 receipt 供 strict replay）。
    receipt = backend.run_experiment(ref.model_ref, controller={"hold": True}, steps=50)
    assert receipt.trace_ref.startswith("simtrc_")
    assert receipt.physical_audit_pass in (True, False)  # 诚实记录（大模型可 FAIL——如实）

    # patch 血缘：阻尼改一点 → 新 ref，母模型不动（freejoint 不在
    # 白名单——选第一个 hinge/slide 关节）。
    joints = inspected.detail["joints_detail"]
    assert joints
    patchable = next(j for j in joints if j.get("type") in ("hinge", "slide", 2, 3))
    patched = backend.patch_model(
        ref.model_ref,
        [
            {
                "op": "set",
                "target": {"type": "joint", "name": patchable["name"]},
                "field": "damping",
                "value": 1.0,
            }
        ],
    )
    assert patched.new_model_ref != ref.model_ref
    parent = backend.store.get(patched.new_model_ref)["parent_model_ref"]
    assert parent == ref.model_ref

    # mjz 自包含导出（MH10b 通道：artifact_ref + size 记录）。
    exported = backend.export_model_mjz(ref.model_ref)
    assert exported["size_bytes"] > 1000 and exported["artifact_ref"].startswith("simmdl_")

    # strict replay：rollout receipt 重放验证。
    report = backend.strict_replay(receipt.receipt_ref)
    assert report["verified"] is True


def test_performance_baseline_recorded(backend, tmp_path) -> None:
    """§50：按 scale 记录基线（记录不硬 Gate——只允许慢于
    离谱阈值的报警，阈值取宽松物理量级）。"""
    rows = []
    for scale, model_name in (("tiny", None), ("quadruped", "unitree_go2")):
        started = time.monotonic()
        if model_name is None:
            from tests.sim.conftest import TINY_MJCF

            (tmp_path / "arm.xml").write_text(TINY_MJCF, encoding="utf-8")
            ref = backend.load_model("arm.xml")
        else:
            ref = backend.load_menagerie(model_name)
        load_ms = (time.monotonic() - started) * 1000

        started = time.monotonic()
        backend.inspect_model(ref.model_ref)
        inspect_ms = (time.monotonic() - started) * 1000

        manifest = backend._manifest(ref.model_ref)
        spec = backend._spec_from_manifest(manifest)
        import mujoco

        from rosclaw.sim.backends.mujoco import state_v2

        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        started = time.monotonic()
        vector, spec_value = state_v2.capture_state_v2(
            model, data, fidelity=state_v2.FIDELITY_FULL_INTEGRATION
        )
        snapshot_ms = (time.monotonic() - started) * 1000

        data2 = mujoco.MjData(model)
        started = time.monotonic()
        state_v2.apply_state_v2(model, data2, vector, spec_value)
        restore_ms = (time.monotonic() - started) * 1000

        started = time.monotonic()
        backend.rollout(ref.model_ref, controller={"hold": True}, steps=200)
        rollout_ms = (time.monotonic() - started) * 1000

        rows.append(
            {
                "scale": scale,
                "model": model_name or "tiny_arm",
                "nq": int(model.nq),
                "load_ms": round(load_ms, 1),
                "inspect_ms": round(inspect_ms, 1),
                "snapshot_ms": round(snapshot_ms, 2),
                "restore_ms": round(restore_ms, 2),
                "rollout_200steps_ms": round(rollout_ms, 1),
            }
        )
    assert len(rows) == 2
    for row in rows:
        # 宽松物理量级报警（防病态退化，不是硬性能 Gate）。
        assert row["rollout_200steps_ms"] < 30_000, row
        assert row["snapshot_ms"] < 1000, row
    (tmp_path / "performance_baseline.json").write_text(
        __import__("json").dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    baseline = (tmp_path / "performance_baseline.json").read_text(encoding="utf-8")
    assert "load_ms" in baseline


def test_clean_wheel_install_smoke(tmp_path) -> None:
    """§49/G46：build wheel → 干净 venv → pip install →
    rosclaw sim capabilities + load/rollout 冒烟。

    不标 slow：G46 是发布门，必须进 CI gate 全回归（gate 跑
    -m 'not slow'，标 slow = 永远只有本地证据——审查实证）。"""
    import json
    import subprocess
    import sys
    import venv

    repo = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dist = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(dist), repo],
        check=True,
        capture_output=True,
        timeout=600,
    )
    wheels = list(dist.glob("rosclaw*.whl"))
    assert wheels, "wheel 未产出"

    clean = tmp_path / "clean_venv"
    venv.create(clean, with_pip=True)
    python = clean / "bin" / "python"
    # 干净 venv 装 wheel + 最小运行依赖（mujoco/numpy——
    # HarnessBench A 条件同款最小栈）。
    subprocess.run(
        [str(python), "-m", "pip", "install", "-q", str(wheels[0]), "mujoco>=3.13,<3.14", "numpy"],
        check=True,
        capture_output=True,
        timeout=900,
    )
    work = tmp_path / "smoke"
    work.mkdir()
    (work / "arm.xml").write_text(
        '<mujoco><worldbody><body name="b" pos="0 0 0.3">'
        '<joint name="j" type="hinge" damping="0.5"/>'
        '<geom type="capsule" size="0.05 0.2" mass="1.0"/></body></worldbody>'
        '<actuator><position name="s" joint="j" kp="10"/></actuator></mujoco>',
        encoding="utf-8",
    )
    caps = subprocess.run(
        [str(clean / "bin" / "rosclaw"), "sim", "capabilities"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert caps.returncode == 0, caps.stderr[-500:]
    payload = json.loads(caps.stdout)
    assert payload["backend"] == "mujoco"
    loaded = subprocess.run(
        [str(clean / "bin" / "rosclaw"), "sim", "load", "arm.xml"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert loaded.returncode == 0, loaded.stderr[-500:]
    assert json.loads(loaded.stdout)["model_ref"].startswith("simmdl_")
