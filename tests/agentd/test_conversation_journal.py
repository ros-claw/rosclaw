"""批次 A 回归（补充实施文档 §3.3）：conversation journal 与 compact 持久化。

- append_conversation 稳定赋 entry_id + 单调 seq
- 内部 journal 键永不越过 provider wire（gateway sanitizer）
- CompactionEntryV1 记录 covered_span_hash / covered_entry_ids / supersedes /
  provider / model / protected_groups
- find_cut_point 不拆 atomic_group
- reactive overflow 走持久化压缩（不原地改写）→ 新消息仍落盘、重启一致
"""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.context.compaction import CompactionStore, find_cut_point
from rosclaw.agentd.models.gateway import (
    MockModelGateway,
    ModelGatewayError,
    _sanitize_message,
)
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1


def _answer(request) -> ModelTurnResultV1:
    decision = {
        "schema_version": "rosclaw.decision.v1",
        "decision_id": "d",
        "mission_id": request.mission_id,
        "context_id": request.context_id,
        "context_revision": request.context_revision,
        "next_intent": "ANSWER",
        "summary": "ok",
        "evidence_refs": [],
    }
    return ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="m",
        content=f"```json\n{json.dumps(decision)}\n```",
        assistant_message={"role": "assistant", "content": "x"},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


def _service(tmp_path: Path, script) -> AgentService:
    config = load_agent_config(tmp_path / "config.yaml")
    return AgentService(config, tmp_path, gateway=MockModelGateway(mock_profile(), script))


class TestStableEntryIds:
    async def test_append_stamps_monotonic_entry_ids(self, tmp_path: Path) -> None:
        service = _service(tmp_path, [_answer] * 10)
        try:
            mission = service.create_mission("entry id 测试")
            await service.send_turn(mission.mission_id, "第一轮")
            await service.send_turn(mission.mission_id, "第二轮")
            history = service.store.conversation(mission.mission_id)
            ids = [m.get("entry_id") for m in history]
            seqs = [m.get("seq") for m in history]
            assert all(ids), history
            assert len(set(ids)) == len(ids), "entry_id 必须唯一"
            assert seqs == sorted(seqs), "seq 必须单调"
            # 重启后 id 不变（稳定身份）。
            service2 = _service(tmp_path, [_answer] * 10)
            history2 = service2.store.conversation(mission.mission_id)
            assert [m.get("entry_id") for m in history2] == ids
            await service2.close()
        finally:
            await service.close()


class TestWireSanitizer:
    def test_internal_keys_stripped(self) -> None:
        msg = {
            "role": "user",
            "content": "hi",
            "entry_id": "conv_mis_x_3",
            "seq": 3,
            "atomic_group": "obs_abc",
            "source": "user",
        }
        assert _sanitize_message(msg) == {"role": "user", "content": "hi"}
        standard = {"role": "assistant", "content": None, "tool_calls": []}
        assert _sanitize_message(standard) is standard

    async def test_model_requests_never_carry_journal_keys(self, tmp_path: Path) -> None:
        """真实 OpenAICompatGateway._build_inputs（无网络）必须净化内部键。"""
        from rosclaw.agentd.models.gateway import ModelTurnRequest, OpenAICompatGateway
        from rosclaw.agentd.models.profiles import kimi_k3_profile

        service = _service(tmp_path, [_answer] * 10)
        try:
            mission = service.create_mission("wire 净化")
            await service.send_turn(mission.mission_id, "hello")
            history = service.store.conversation(mission.mission_id)
            assert any("entry_id" in m for m in history), "journal 必须带 entry_id"
            import os

            os.environ["ROSCLAW_TEST_DUMMY_KEY"] = "test-dummy-not-a-secret"
            gateway = OpenAICompatGateway(
                kimi_k3_profile(api_key_ref="env:ROSCLAW_TEST_DUMMY_KEY")
            )
            request = ModelTurnRequest(
                system_prompt="sys",
                messages=history,
                tools=[],
                max_output_tokens=16,
                mission_id=mission.mission_id,
                context_id="ctx",
                context_revision=1,
            )
            wire = gateway._build_inputs(request)
            for msg in wire["messages"]:
                assert "entry_id" not in msg, msg
                assert "seq" not in msg
                assert "atomic_group" not in msg
            await gateway.close()
        finally:
            await service.close()


class TestCompactionAuditFields:
    async def test_entry_records_coverage_and_supersedes(self, tmp_path: Path) -> None:
        service = _service(tmp_path, [_answer] * 50)
        try:
            mission = service.create_mission("审计字段")
            for i in range(4):
                await service.send_turn(mission.mission_id, f"第 {i} 轮 {'长文本' * 100}")
            await service.compact(mission.mission_id)
            for i in range(4, 8):
                await service.send_turn(mission.mission_id, f"第 {i} 轮 {'长文本' * 100}")
            await service.compact(mission.mission_id)
            store = CompactionStore(service.store.connection)
            entries = store.list(mission.mission_id)
            assert len(entries) == 2
            first, second = entries
            assert first.covered_span_hash.startswith("cmp_span_")
            assert first.covered_entry_ids, "covered entry ids 必须记录"
            assert first.provider == "mock" and first.model == "mock-model"
            assert second.supersedes == first.compaction_id
            # covered ids 确实来自 journal。
            history = service.store.conversation(mission.mission_id)
            journaled_ids = {m.get("entry_id") for m in history}
            # marker 消息也带 entry_id；covered 的原始 id 已不在 view 中，
            # 但 canonical journal（含 compact 前事件）里仍可审计。
            assert all(eid.startswith("conv_") for eid in first.covered_entry_ids)
            assert journaled_ids, "view 必须保留 entry_id"
        finally:
            await service.close()


class TestAtomicGroupProtection:
    def test_cut_never_splits_atomic_group(self) -> None:
        messages = [{"role": "user", "content": "q " + "长" * 500}]
        for i in range(8):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"obs_{i}",
                    "content": "证据 " + "长" * 500,
                    "atomic_group": f"obs_{i}",
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "continue",
                    "atomic_group": f"obs_{i}",
                }
            )
        cut = find_cut_point(messages, keep_recent_tokens=400)
        if 0 < cut < len(messages):
            left_group = messages[cut - 1].get("atomic_group")
            right_group = messages[cut].get("atomic_group")
            assert left_group != right_group or left_group is None, (
                f"cut {cut} splits atomic group {left_group}"
            )


class TestReactiveOverflowPersistentCompaction:
    async def test_overflow_compacts_persistently_and_keeps_persisting(
        self, tmp_path: Path
    ) -> None:
        fired = {"done": False}

        def overflow_then_answer(request) -> ModelTurnResultV1:
            trigger = any(
                "触发 overflow" in str(m.get("content") or "") for m in request.messages
            )
            if trigger and not fired["done"]:
                fired["done"] = True
                raise ModelGatewayError("http_error", "HTTP 400: prompt is too long")
            return _answer(request)

        service = _service(tmp_path, [overflow_then_answer] * 20)
        try:
            mission = service.create_mission("overflow 压缩")
            for i in range(8):
                await service.send_turn(mission.mission_id, f"第 {i} 轮 {'长文本' * 2000}")
            # 触发 overflow → 持久化压缩 → 重试成功。
            result = await service.send_turn(mission.mission_id, "触发 overflow 的一轮")
            assert result.degraded == "reactive_compacted"
            store = CompactionStore(service.store.connection)
            entry = store.latest(mission.mission_id)
            assert entry is not None and entry.reason == "overflow"
            # §3.3 核心回归：压缩后新消息必须继续落盘。
            await service.send_turn(mission.mission_id, "压缩后的新问题")
            history = service.store.conversation(mission.mission_id)
            assert any("压缩后的新问题" in str(m.get("content")) for m in history)
            # 重启 view 一致。
            service2 = _service(tmp_path, [_answer] * 5)
            restored = service2.store.conversation(mission.mission_id)
            assert [m.get("entry_id") for m in restored] == [
                m.get("entry_id") for m in history
            ]
            assert any("压缩后的新问题" in str(m.get("content")) for m in restored)
            await service2.close()
        finally:
            await service.close()
