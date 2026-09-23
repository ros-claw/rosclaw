"""G43 ROS2 exact-step agreement live 门（MH25 后续，2026-09-23）。

纪律锁定：
- 无桥环境（CI）：脚本立即诚实 NOT_RUN，绝不伪造 AGREEMENT；
- 有桥环境（operator）：真桥 pause → reset(keyframe) → step N →
  joint_states vs 本地 MuJoCo rollout 逐位比对。

live 实证（本机 GB10 + ros-jazzy-mujoco-ros2-control 0.1.1）：
桥 MuJoCo 3.12.0 vs rosclaw 3.13.0，500 步单摆摆动，qpos/qvel
终态逐位一致（diff 0.0/0.0）——证据 docs/reports/ros2-bridge/。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "ros2_bridge_live.py"


def _bridge_present() -> bool:
    # ros2 不在默认 PATH（需 source setup.bash）——查安装路径本身。
    return Path("/opt/ros/jazzy/bin/ros2").is_file() and Path(
        "/opt/ros/jazzy/lib/mujoco_ros2_control"
    ).is_dir()


@pytest.mark.skipif(_bridge_present(), reason="有桥环境走 live 用例")
def test_g43_script_honest_not_run_without_bridge(tmp_path) -> None:
    """无桥（CI 实况）：立即 NOT_RUN + 机读原因，绝不产出 AGREEMENT。"""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--steps", "5", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["verdict"] == "NOT_RUN"
    assert evidence["notes"]
    # 绝不伪造一致性证据
    assert "final_abs_diff" not in evidence


@pytest.mark.integration
@pytest.mark.skipif(not _bridge_present(), reason="无 ROS2 桥——诚实 skip")
def test_g43_exact_step_agreement_live(tmp_path) -> None:
    """真桥 exact-step 一致性（operator/本机）：AGREEMENT 才算过。

    桥 3.12.0 vs 本地 3.13.x 跨版本逐位一致是物理事实级证据；
    任何回归（桥包升级/本地 mujoco 升级/协议漂移）都应在此显形。
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--steps", "500", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-500:]
    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    # 诚实中间态可机读（DIVERGED/NOT_RUN 都让门红，不吞）
    assert evidence["verdict"] == "AGREEMENT", json.dumps(evidence, ensure_ascii=False)[:600]
    assert evidence["final_abs_diff"]["qpos"] <= evidence["tolerance_rad"]
