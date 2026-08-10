"""PR-12 LIMO 完整闭环验收（SIMULATION 证据域）。

真实路径：limo-sim MCP（stdio）→ OBSERVE 证据封装 → REQUEST_APPROVAL →
operator.sock（peer identity + display hash）→ Broker EXACT_ACTION →
REQUEST_ACTION → SimActionChannel 执行 → SIMULATED receipt →
位姿验证 → practice candidate。SHADOW/REAL 门控诚实拒绝。
"""

from __future__ import annotations

from pathlib import Path

from rosclaw.agentd.bench.limo_acceptance import run_acceptance
from tests.agentd.conftest import LOCAL_PRINCIPAL


async def test_limo_acceptance_simulation(tmp_path: Path) -> None:
    report = await run_acceptance(tmp_path)
    failed = [k for k, v in report.checks.items() if not v]
    assert not failed, f"acceptance failed: {failed}\nnotes: {report.notes}"
    assert report.practice_candidate
    # SIM 证据域诚实：grants 单次消费、receipts 持久化。
    assert report.checks["C9_grants_single_use"]
    assert report.checks["C10_trace_persisted"]
    assert report.checks["C5_honest_no_acoustic_proof"]


async def test_shadow_real_gates_honest(tmp_path: Path) -> None:
    """无 rosclawd/真实硬件时 SHADOW/REAL 不得用 SIM 证据冒充。"""
    import yaml

    from rosclaw.agentd.bench.limo_acceptance import limo_sim_mcp_server_config
    from rosclaw.agentd.config import load_agent_config
    from rosclaw.agentd.models.gateway import MockModelGateway
    from rosclaw.agentd.models.profiles import mock_profile
    from rosclaw.agentd.service import AgentService

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {"agent": {"enabled": True}, "kits": {"disabled": ["rosclaw/ur5e-sim"]}, "mcp_servers": [limo_sim_mcp_server_config()]}
        ),
        encoding="utf-8",
    )
    config = load_agent_config(tmp_path / "config.yaml")
    service = AgentService(config, tmp_path, gateway=MockModelGateway(mock_profile(), []))
    try:
        # SimActionChannel 只接受 SIMULATION。
        import pytest

        from rosclaw.agentd.sim_executor import SimActionChannel

        channel = SimActionChannel(command="true", args=())
        with pytest.raises(Exception, match="SIMULATION"):
            await channel.execute(
                capability_id="limo.speaker.play_tone",
                arguments={},
                grant_id="g",
                mode="SHADOW",
            )
        # REAL 模式 mission 在无 daemon 时 estop 诚实不可用。
        from rosclaw.contracts.common import ValidationError

        with pytest.raises(ValidationError, match="estop unavailable"):
            await service.estop("test", principal=LOCAL_PRINCIPAL)
    finally:
        await service.close()
