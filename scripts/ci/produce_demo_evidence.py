#!/usr/bin/env python3
"""demo 证据包生产者（PR-H9）：SIM demo（SimTrajectoryService 确定性
闭环）→ EvidencePackWriter 落包。

替代已删除的 bench/limo_acceptance.run_acceptance（旧 loop 驱动）。
证据等级诚实：SIM_DYN_ROLLOUT 动力学 rollout——非真机证据。
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path


async def _demo(home: Path) -> dict:
    from rosclaw.agentd.runtime_manager import RuntimeManager
    from rosclaw.agentd.sim_trajectory import SimTrajectoryService

    manager = RuntimeManager(home)
    manager.ensure("rosclaw-simulation")
    sim = SimTrajectoryService(home, runtime_manager=manager)
    plan = await asyncio.to_thread(
        sim.generate_planar_path,
        shape="star5", center_m=[0.35, 0.25, 0.30], scale_m=0.10,
    )
    result = await asyncio.to_thread(
        sim.simulate_cartesian_trajectory, plan["plan_id"]
    )
    render = await asyncio.to_thread(sim.render_trace, result["trace_id"], format="gif")
    verify = await asyncio.to_thread(
        sim.verify_tracking, result["trace_id"], max_tracking_error_m=0.05
    )
    return {"result": result, "render": render, "verify": verify}


def main() -> int:
    from rosclaw.agentd.bench.evidence_levels import EvidenceLevel
    from rosclaw.agentd.bench.evidence_pack import EvidencePackWriter

    evidence_root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/evidence-ci")
    with tempfile.TemporaryDirectory() as d:
        out = asyncio.run(_demo(Path(d)))
    verify = out["verify"]
    result = out["result"]
    render = out["render"]
    passed = (
        verify["verdict"] == "PASS"
        and render["artifact"]["frames"] >= 30
        and result.get("is_safe")
    )

    pack = EvidencePackWriter(evidence_root, run_id="demo_sim_star")
    pack.write_environment(provider="none (deterministic sim)")
    pack.write_commands(["scripts/ci/produce_demo_evidence.py"])
    pack.write_events([
        {"type": "demo.plan", "payload": {"shape": "star5"}},
        {"type": "demo.rollout", "payload": {"is_safe": result.get("is_safe")}},
        {"type": "demo.verify", "payload": {"verdict": verify["verdict"]}},
    ])
    pack.write_mission_snapshot({"mission_id": "demo_sim_star", "mode": "SIMULATION"})
    pack.write_public_records(approvals=[], permits=[], receipts=[])
    pack.write_metrics({
        "passed": passed,
        "max_error_m": verify["metrics"]["max_error_m"],
        "frames": render["artifact"]["frames"],
    })
    pack.write_observer(
        "SIM 动力学 rollout（star5）——证据等级 SIM_DYN_ROLLOUT，"
        "不能证明真机执行效果。"
    )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    pack.finalize(
        level=EvidenceLevel.E3_SIM_VERIFIED,
        git_commit=git_commit or "unknown",
        dirty=False,
        test_ids=["H9-demo"],
        operator="ci",
    )
    print(f"evidence pack at {pack.dir} passed={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
