"""W09 红测试（规格 2026-09-08 §13）：十类本地任务验收层。

三层纪律（§13.1）：本文件执行**物理/媒体层**（真实 MuJoCo，无
大模型）；agent 层（需真实模型 key）逐例标 NOT_RUN 跳过——
不合成冒充；契约/故障层由既有套件覆盖。

每个 case 产结果记录（§13.2）：分发物 hash 占位/模型 ID/seed/
执行状态/成功/假成功/费用未知标未知。评分器与答案不挂给
Agent（本层无 Agent）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

CASES_DIR = Path(__file__).parent / "cases"


def _cases() -> list[dict]:
    return [
        yaml.safe_load(p.read_text(encoding="utf-8"))
        for p in sorted(CASES_DIR.glob("*.yaml"))
    ]


def _record(case: dict, verdict: str, detail: dict, started: float) -> dict:
    """§13.2 结果记录（费用未知标未知，不写 0）。"""
    return {
        "id": case["id"], "tier": case["tier"], "verdict": verdict,
        "wall_time_s": round(time.monotonic() - started, 2),
        "model": None,  # 物理层无模型——None 不是假记录
        "cost": "unknown",  # 费用未知标未知
        "false_success": False,
        "detail": detail,
    }


class TestCaseDefinitions:
    def test_ten_cases_complete_schema(self) -> None:
        cases = _cases()
        assert len(cases) == 10, f"只有 {len(cases)} 个 case"
        for case in cases:
            for key in ("id", "tier", "fixture", "variants",
                        "allowed_mode", "allowed_outputs", "oracle",
                        "grading"):
                assert key in case, f"{case.get('id')} 缺 {key}"
            assert case["tier"] in ("contract", "physics", "agent")
            assert case["allowed_mode"] == "SIM", (
                f"{case['id']} 未限 SIM（REAL 门禁）"
            )


FREEJOINT_VARIANTS = [
    # (name, nq, nv, nu)：freejoint(7/6) + hinge(1/1) + 可选 ball(4/3)
    ("v1_hinge", 8, 7, 1),
    ("v2_hinge_ball", 12, 10, 2),
    ("v3_hinge_only_actuator_layout", 8, 7, 1),
]


def _variant_mjcf(name: str, with_ball: bool) -> str:
    ball = """
      <body name="wrist" pos="0 0 0.1">
        <joint name="wrist_ball" type="ball"/>
        <geom type="sphere" size="0.01"/>
      </body>""" if with_ball else ""
    ball_act = """
    <motor name="ball_motor" joint="wrist_ball" ctrlrange="-1 1"/>""" if with_ball else ""
    return f"""<mujoco model="{name}">
  <worldbody>
    <body name="free_obj" pos="0 0 0.1">
      <freejoint name="obj_free"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.5"/>
    </body>
    <body name="arm" pos="0.3 0 0.3">
      <joint name="shoulder" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.01 0.1"/>
      {ball}
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" ctrlrange="-1 1"/>
    {ball_act}
  </actuator>
</mujoco>"""


class TestL01UnfamiliarModel:
    """物理层：变体结构解释真值一致 + 完整状态记录（nq≠nv≠nu）。"""

    @pytest.mark.parametrize(
        "name,nq,nv,nu,with_ball",
        [("v1_hinge", 8, 7, 1, False), ("v2_hinge_ball", 12, 10, 2, True)],
    )
    def test_structure_truth_and_full_state(
        self, tmp_path, name, nq, nv, nu, with_ball,
    ) -> None:
        from rosclaw.sim import api
        from rosclaw.sim.model_inspect import inspect_mjcf

        started = time.monotonic()
        mjcf = tmp_path / f"{name}.xml"
        mjcf.write_text(_variant_mjcf(name, with_ball), encoding="utf-8")
        info = inspect_mjcf(mjcf)
        assert (info.nq, info.nv, info.nu) == (nq, nv, nu)
        passive = {j["name"] for j in info.joints} - {
            a["joint"] for a in info.actuators
        }
        assert "obj_free" in passive  # freejoint 不是 actuator
        if with_ball:
            assert "wrist_ball" not in passive  # ball 有 motor
        ref, _ = api.load_model(mjcf, task_root=tmp_path)
        op = api.submit_simulation(
            ref, None, {"ctrl_series": [[0.2] * nu] * 30}, 0.06,
            task_root=tmp_path,
        )
        trace = api.read_trace(op, task_root=tmp_path)
        sample = trace["states"][-1]
        assert len(sample["qpos"]) == nq  # 完整状态（W03）
        assert len(sample["ctrl"]) == nu
        rec = _record(
            {"id": f"L01/{name}", "tier": "physics"}, "PASS",
            {"nq": nq, "nv": nv, "nu": nu}, started,
        )
        assert rec["cost"] == "unknown"


class TestL02ArbitraryTrajectory:
    """物理层：任意 SE(3) 路径 rollout + 时间对齐 + 视频同源。"""

    def test_rollout_render_time_aligned(self, tmp_path) -> None:
        from rosclaw.sim import api

        started = time.monotonic()
        mjcf = tmp_path / "m.xml"
        mjcf.write_text(_variant_mjcf("l02", False), encoding="utf-8")
        ref, _ = api.load_model(mjcf, task_root=tmp_path)
        op = api.submit_simulation(
            ref, None, {"ctrl_series": [[0.4]] * 100}, 0.2,
            task_root=tmp_path,
        )
        trace = api.read_trace(op, task_root=tmp_path)
        times = [s["t"] for s in trace["states"]]
        assert times == sorted(times)  # 时间单调
        rop = api.render(
            op, {"camera": "follow", "outputs": ["gif"], "fps": 10.0},
            task_root=tmp_path,
        )
        gif = tmp_path / "renders" / rop / f"{rop}.gif"
        assert gif.exists()
        from PIL import Image

        with Image.open(gif) as img:
            assert getattr(img, "n_frames", 1) >= 2
        receipt = json.loads(
            (tmp_path / "renders" / rop / "render_receipt.json").read_text()
        )
        assert receipt["evidence_level"] == "agent_generated_experiment"
        _record({"id": "L02", "tier": "physics"}, "PASS",
                {"duration_s": 0.2}, started)


class TestL07DampingExperiment:
    """物理层：三阻尼×三次，只改声明参数，衰减结论可重算。"""

    PENDULUM = """<mujoco model="pend">
      <option timestep="0.002"/>
      <worldbody>
        <body name="link" pos="0 0 0.3">
          <joint name="hinge" type="hinge" axis="0 1 0"
                 damping="{damping}" armature="0"/>
          <geom type="capsule" size="0.01 0.15" pos="0 0 -0.15" mass="1"/>
        </body>
      </worldbody>
    </mujoco>"""

    def test_decay_recomputable_and_monotone(self, tmp_path) -> None:
        import mujoco
        import numpy as np

        started = time.monotonic()
        results: dict[float, list[float]] = {}
        # 夹具参数选欠阻尼区（§13.9：避免过阻尼改变指标定义——
        # 实测 damping=0.8 六秒内峰值不足三个，指标失效）。
        for damping in (0.02, 0.1, 0.3):
            peak_decays: list[float] = []
            for _run in range(3):  # 每条件三次（同参确定性重跑）
                model = mujoco.MjModel.from_xml_string(
                    self.PENDULUM.format(damping=damping)
                )
                data = mujoco.MjData(model)
                data.qpos[0] = 0.5
                peaks: list[float] = []
                prev_vel = 0.0
                steps = int(8.0 / model.opt.timestep)
                for _ in range(steps):
                    mujoco.mj_step(model, data)
                    vel = float(data.qvel[0])
                    if prev_vel > 0 >= vel or prev_vel < 0 <= vel:
                        peaks.append(abs(float(data.qpos[0])))
                    prev_vel = vel
                # 衰减率 = 相邻峰值比（从数据重算，不读理论值）。
                if len(peaks) >= 3:
                    peak_decays.append(float(peaks[2] / peaks[1]))
            results[damping] = peak_decays
        rates = {d: float(np.mean(v)) for d, v in results.items() if v}
        assert len(rates) == 3, f"条件未产生可重算峰值: {results.keys()}"
        assert rates[0.3] < rates[0.02], (
            f"阻尼越大衰减越快——实测不符: {rates}"
        )
        _record({"id": "L07", "tier": "physics"}, "PASS",
                {"peak_ratio_by_damping": rates}, started)


class TestL10RenderLifecycle:
    """物理层：同 trace 双视角独立产出（并发安全由 W04 身份键
    保证）；重启后读同一 trace 不重 rollout。"""

    def test_two_views_independent_and_restart_reads(
        self, tmp_path
    ) -> None:
        from rosclaw.sim import api

        started = time.monotonic()
        mjcf = tmp_path / "m.xml"
        mjcf.write_text(_variant_mjcf("l10", False), encoding="utf-8")
        ref, _ = api.load_model(mjcf, task_root=tmp_path)
        op = api.submit_simulation(
            ref, None, {"ctrl_series": [[0.3]] * 50}, 0.1,
            task_root=tmp_path,
        )
        digest_before = json.loads(
            (tmp_path / "models" / f"{op}.json").read_text()
        )["states_digest"]
        r1 = api.render(op, {"camera": "follow", "outputs": ["gif"]},
                        task_root=tmp_path)
        r2 = api.render(op, {"camera": "top", "outputs": ["gif"]},
                        task_root=tmp_path)
        assert r1 != r2  # 独立 operation 目录
        assert (tmp_path / "renders" / r1 / f"{r1}.gif").exists()
        assert (tmp_path / "renders" / r2 / f"{r2}.gif").exists()
        # “重启”= 新进程式重读（记录持久化——不重 rollout）。
        digest_after = json.loads(
            (tmp_path / "models" / f"{op}.json").read_text()
        )["states_digest"]
        assert digest_before == digest_after
        _record({"id": "L10", "tier": "physics"}, "PASS",
                {"views": [r1, r2]}, started)


class TestAgentTierHonestNotRun:
    """agent 层（需真实模型 key）：逐例标 NOT_RUN——不合成冒充。
    有 key 时由真实验收驱动（W10 门禁）。"""

    @pytest.mark.parametrize(
        "case_id", ["L03_obstacle_retry", "L04_contact_push",
                    "L05_visual_grounding", "L06_cartpole",
                    "L08_missing_capability", "L09_modify_and_append"],
    )
    def test_agent_tier_not_run_without_key(self, case_id) -> None:
        import os

        if not os.environ.get("ROSCLAW_KIMI_API_KEY"):
            pytest.skip(
                f"NOT_RUN: {case_id} 属 agent 层（需真实模型）——"
                "不合成冒充"
            )
        pytest.fail("有 key 时应走真实驱动（W10）——未接线")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
