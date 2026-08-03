"""External harness WorkerPack tests (PR-WF-054 exits).

- packs: version_ok、card 合法性（注册校验通过）
- probe：缺失二进制 → T0 诚实详情 + 安装指导；版本过旧 → not ready
- adapter：假 claude CLI 全链路（manager → ACCEPTED）；env 白名单透传
- live conformance（slow）：真实 claude CLI 分析任务
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from rosclaw.agentd.mission import MissionStore
from rosclaw.agentd.workers import WorkerManager, WorkerRegistry
from rosclaw.agentd.workers.external import ExternalHarnessAdapter
from rosclaw.agentd.workers.packs import (
    ALL_PACKS,
    CLAUDE_CODE_PACK,
    CODEX_CLI_PACK,
    card_for_pack,
    version_ok,
)
from rosclaw.agentd.workers.registry import validate_card
from rosclaw.agentd.workers.scheduler import CandidateView
from rosclaw.contracts.common import new_id
from rosclaw.contracts.worker.order import (
    BudgetEnvelope,
    SideEffectPolicy,
    WorkOrderV1,
)

ACTOR = "agent:test"


class TestPacks:
    def test_version_ok(self) -> None:
        assert version_ok("2.1.220", "2.0.0")
        assert version_ok("2.0.0", "2.0.0")
        assert not version_ok("1.9.9", "2.0.0")
        assert version_ok("0.20.1", "0.20.0")

    def test_cards_valid(self) -> None:
        for pack in ALL_PACKS:
            validate_card(card_for_pack(pack))  # raises on invalid

    def test_claude_pack_metadata(self) -> None:
        assert CLAUDE_CODE_PACK.executable == "claude"
        assert "claude.com" in CLAUDE_CODE_PACK.install_hint
        assert CODEX_CLI_PACK.executable == "codex"
        assert "codex" in CODEX_CLI_PACK.install_hint


def _fake_cli(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


FAKE_CLAUDE = """#!/usr/bin/env python3
import json, sys
if "--version" in sys.argv:
    print("2.1.0 (Claude Code)")
    sys.exit(0)
prompt = sys.argv[2]
print(json.dumps({"result": "分析：给定日志的根因是超时配置过短 [推断]", "usage": {"input_tokens": 120, "output_tokens": 30}, "total_cost_usd": 0.0004}))
"""

OUTDATED_CLAUDE = """#!/usr/bin/env bash
echo "1.5.0 (Claude Code)"
"""


class TestProbe:
    async def test_missing_binary_t0_honest(self) -> None:
        adapter = ExternalHarnessAdapter()
        result = await adapter.probe(CODEX_CLI_PACK.worker_id)
        assert not result.ready
        assert "T0" in result.detail
        assert "安装" in result.detail or "install" in result.detail.lower()

    async def test_outdated_binary_rejected(self, tmp_path: Path, monkeypatch) -> None:
        _fake_cli(tmp_path, "claude", OUTDATED_CLAUDE)
        monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
        adapter = ExternalHarnessAdapter()
        result = await adapter.probe(CLAUDE_CODE_PACK.worker_id)
        assert not result.ready
        assert "最小兼容版本" in result.detail

    async def test_valid_binary_ready(self, tmp_path: Path, monkeypatch) -> None:
        _fake_cli(tmp_path, "claude", FAKE_CLAUDE)
        monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
        adapter = ExternalHarnessAdapter()
        result = await adapter.probe(CLAUDE_CODE_PACK.worker_id)
        assert result.ready


class TestExternalAdapterFlow:
    async def test_full_manager_path_accepted(self, tmp_path: Path, monkeypatch) -> None:
        _fake_cli(tmp_path, "claude", FAKE_CLAUDE)
        monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
        store = MissionStore(tmp_path / "m.db")
        registry = WorkerRegistry(store.connection)
        card = card_for_pack(CLAUDE_CODE_PACK)
        registry.register(card, actor_id=ACTOR)
        manager = WorkerManager(
            store.connection,
            adapters={"external_cli": ExternalHarnessAdapter(cwd=tmp_path)},
            actor_id=ACTOR,
        )
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id="mis_x",
            issued_by=ACTOR,
            capability="code.repository_analysis",
            goal="分析这段失败日志",
            inputs={"instructions": "只基于给定日志", "artifacts": ["log://1"]},
            budgets=BudgetEnvelope(wall_time_sec=30),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
        )
        scheduled = manager.hire(order, [CandidateView(card=card)])
        result, report = await manager.run_to_completion(scheduled)
        assert result.status == "COMPLETED"
        assert "根因" in result.summary
        assert report.accepted, report.reasons
        assert result.usage.prompt_tokens == 120
        assert result.usage.cost_microunits == 400

    async def test_start_missing_binary_honest(self, tmp_path: Path) -> None:
        store = MissionStore(tmp_path / "m.db")
        registry = WorkerRegistry(store.connection)
        card = card_for_pack(CODEX_CLI_PACK)
        registry.register(card, actor_id=ACTOR)
        manager = WorkerManager(
            store.connection,
            adapters={"external_cli": ExternalHarnessAdapter(cwd=tmp_path)},
            actor_id=ACTOR,
        )
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id="mis_x",
            issued_by=ACTOR,
            capability="code.repository_analysis",
            goal="分析",
            budgets=BudgetEnvelope(wall_time_sec=10),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
        )
        scheduled = manager.hire(order, [CandidateView(card=card)])
        result, report = await manager.run_to_completion(scheduled)
        assert result.status == "FAILED"
        assert not report.accepted
        assert "not found" in result.summary or "install" in result.summary.lower()

    async def test_env_passthrough_whitelist(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-visible")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-hidden")
        monkeypatch.setenv("PATH", os.environ["PATH"])
        adapter = ExternalHarnessAdapter()
        env = adapter._env(CLAUDE_CODE_PACK, {})
        assert env.get("ANTHROPIC_API_KEY") == "sk-test-visible"
        assert "AWS_SECRET_ACCESS_KEY" not in env


@pytest.mark.slow
class TestLiveClaudeConformance:
    async def test_real_claude_analysis(self, tmp_path: Path) -> None:
        import shutil

        if shutil.which("claude") is None:
            pytest.skip("claude CLI 不存在（T0）")
        store = MissionStore(tmp_path / "m.db")
        registry = WorkerRegistry(store.connection)
        card = card_for_pack(CLAUDE_CODE_PACK)
        registry.register(card, actor_id=ACTOR)
        manager = WorkerManager(
            store.connection,
            adapters={"external_cli": ExternalHarnessAdapter(cwd=tmp_path)},
            actor_id=ACTOR,
        )
        order = WorkOrderV1(
            work_order_id=new_id("wo"),
            mission_id="mis_x",
            issued_by=ACTOR,
            capability="code.repository_analysis",
            goal="用一句话分析：连接超时 3 秒失败的根因与建议",
            inputs={"instructions": "不超过两句话"},
            budgets=BudgetEnvelope(wall_time_sec=180),
            side_effect_policy=SideEffectPolicy(**{"class": "none"}),
        )
        scheduled = manager.hire(order, [CandidateView(card=card)])
        result, report = await manager.run_to_completion(scheduled)
        assert result.status == "COMPLETED", result.summary
        assert report.accepted, report.reasons
        assert result.summary.strip()
