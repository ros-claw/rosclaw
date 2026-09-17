"""`rosclaw sim` CLI 合成测试（MH11，0916 优化 §五/§十，红→绿）。

B 侧 Agent 触达 Harness 的产品面：SimulationRuntime 的 JSON CLI 投影。
合成层不碰真实模型——子进程跑 CLI 对临时 task root 全链验证。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

TINY_MJCF = """<mujoco model="tiny_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
      <geom name="base_geom" type="capsule" size="0.05 0.2" mass="1.0"/>
      <body name="forearm" pos="0 0 0.4">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="forearm_geom" type="capsule" size="0.03 0.2" mass="0.5"/>
        <site name="tool0" pos="0 0 0.3"/>
      </body>
    </body>
    <camera name="top" pos="0 0 2"/>
  </worldbody>
  <actuator>
    <position name="shoulder_servo" joint="shoulder" kp="10" ctrlrange="-1.57 1.57" forcerange="-50 50"/>
    <position name="elbow_servo" joint="elbow" kp="5"/>
  </actuator>
  <sensor>
    <jointpos name="shoulder_pos" joint="shoulder"/>
  </sensor>
</mujoco>
"""


def _cli(root: Path, *args: str) -> tuple[int, dict]:
    """跑 CLI，返回 (exit_code, stdout_json)。stdout 必须是纯 JSON。"""
    proc = subprocess.run(
        [sys.executable, "-m", "rosclaw.entrypoint", "sim", "--root", str(root), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(proc.stdout) if proc.stdout.strip() else {}
    return proc.returncode, payload


@pytest.fixture
def root(tmp_path):
    (tmp_path / "arm.xml").write_text(TINY_MJCF, encoding="utf-8")
    return tmp_path


def test_capabilities(root) -> None:
    code, out = _cli(root, "capabilities")
    assert code == 0
    assert out["backend"] == "mujoco"
    assert out["usable_for_real_execution"] is False


def test_full_chain_via_cli(root) -> None:
    """load → inspect → patch → rollout → observe → audit → compare：
    Agent 标准实验链全部走 CLI（每步 stdout 是纯 JSON）。"""
    code, loaded = _cli(root, "load", "arm.xml")
    assert code == 0 and loaded["model_ref"].startswith("simmdl_")
    model_ref = loaded["model_ref"]

    code, inspected = _cli(root, "inspect", model_ref)
    assert code == 0 and inspected["nq"] == 2

    patches = json.dumps(
        [
            {
                "op": "set",
                "target": {"type": "actuator", "name": "shoulder_servo"},
                "field": "kp",
                "value": 100.0,
            },
            {
                "op": "set",
                "target": {"type": "actuator", "name": "elbow_servo"},
                "field": "kp",
                "value": 100.0,
            },
        ]
    )
    code, patched = _cli(root, "patch", model_ref, "--patches", patches)
    assert code == 0 and patched["ok"] is True
    new_ref = patched["new_model_ref"]

    code, receipt = _cli(
        root,
        "rollout",
        new_ref,
        "--controller",
        json.dumps({"position_targets": [0.4, 0.2]}),
        "--duration-s",
        "0.2",
    )
    assert code == 0
    assert receipt["trust_level"] == "SIMULATED"
    assert receipt["trace_ref"].startswith("simtrc_")

    code, observed = _cli(
        root,
        "observe",
        new_ref,
        receipt["final_state_ref"],
        "--channels",
        "joint_positions",
    )
    assert code == 0 and observed["values"]["joint_positions"][0] > 0.02

    code, audited = _cli(root, "audit", new_ref, "--trace-ref", receipt["trace_ref"])
    assert code == 0 and audited["status"] in ("PASS", "WARN")

    code, receipt2 = _cli(
        root, "rollout", new_ref, "--controller", '{"hold": true}', "--steps", "50"
    )
    assert code == 0
    code, compared = _cli(root, "compare", receipt["receipt_ref"], receipt2["receipt_ref"])
    assert code == 0 and compared["best_ref"]


def test_patches_from_file(root) -> None:
    """--patches @file：长 patch 列表从文件读（Agent 不落长命令行）。"""
    code, loaded = _cli(root, "load", "arm.xml")
    patch_file = root / "patches.json"
    patch_file.write_text(
        json.dumps(
            [
                {
                    "op": "set",
                    "target": {"type": "joint", "name": "shoulder"},
                    "field": "damping",
                    "value": 2.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    code, patched = _cli(root, "patch", loaded["model_ref"], "--patches", f"@{patch_file}")
    assert code == 0 and patched["ok"] is True


def test_branch_experiment_via_cli(root) -> None:
    code, loaded = _cli(root, "load", "arm.xml")
    branches = json.dumps(
        [
            {"name": "baseline", "patches": []},
            {
                "name": "kp100",
                "patches": [
                    {
                        "op": "set",
                        "target": {"type": "actuator", "name": "shoulder_servo"},
                        "field": "kp",
                        "value": 100.0,
                    },
                    {
                        "op": "set",
                        "target": {"type": "actuator", "name": "elbow_servo"},
                        "field": "kp",
                        "value": 100.0,
                    },
                ],
            },
        ]
    )
    code, result = _cli(
        root,
        "branch-experiment",
        loaded["model_ref"],
        "--branches",
        branches,
        "--controller",
        json.dumps({"position_targets": [0.4, 0.2]}),
        "--steps",
        "60",
    )
    assert code == 0
    assert result["count"] == 2
    assert all(r["trace_ref"] for r in result["receipts"])


def test_structured_error(root) -> None:
    """失败 = 结构化错误（exit 1 + stderr JSON），绝不 traceback 糊屏。"""
    code, out = _cli(root, "load", "ghost.xml")
    assert code == 1
    assert out == {}  # stdout 保持干净
    code2, _ = _cli(root, "inspect", "simmdl_nonexistent000")
    assert code2 == 1


def test_compile_world_via_cli(root) -> None:
    spec = {
        "schema_version": "rosclaw.sim.worldspec.v1",
        "world": {"gravity": [0, 0, -9.81], "ground": True, "seed": 0},
        "objects": [
            {
                "id": "cube",
                "shape": "box",
                "size": [0.03, 0.03, 0.03],
                "pos": [0.4, 0, 0.03],
                "mass": 0.1,
            },
        ],
        "interaction_points": [],
        "task": {"goal": "", "success": []},
    }
    spec_file = root / "world.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    code, world = _cli(root, "compile-world", "--spec", f"@{spec_file}", "--name", "bench_world")
    assert code == 0 and world["model_ref"].startswith("simmdl_")


def test_root_defaults_to_cwd(tmp_path) -> None:
    """--root 省略时 task root = cwd（HarnessBench 独立 workspace 纪律）。"""
    (tmp_path / "arm.xml").write_text(TINY_MJCF, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "rosclaw.entrypoint", "sim", "load", "arm.xml"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert proc.returncode == 0
    loaded = json.loads(proc.stdout)
    assert loaded["model_ref"].startswith("simmdl_")
    # store 落在 cwd 内（不污染 ~/.rosclaw）。
    assert (tmp_path / "sim").is_dir()
