"""K0 live API probe against Kimi K3（总纲 §18 的存活部分）。

Excluded from default CI (``integration`` marker); skips unless
``ROSCLAW_KIMI_API_KEY`` (+ optional ``ROSCLAW_KIMI_BASE_URL``) is set.
The key is read from the environment only. No fixtures may substitute for
the real API — if the API is unreachable the test fails, never fabricates.

退役说明（0911 验证，V-3）：原 test_kimi_live.py 的 K1–K6 已随
AgentService.send_turn / _worker_manager / _broker / _team_coordinator
一并腐烂（该架构已被 Pi 单链取代），且 integration 标记使其休眠
不可见。退役时的真实模型继任覆盖：
- 具身诚实/缺失能力拒绝：tests/eval/agent_tier/test_l08_missing_capability.py
  （真实 K3 已验收——诚实拒绝 + fake REAL 零 action_txn）；
- 授权链/单次 grant：ADR-0007 双层授权套件 + W06 consent 探针（合成层）
  + agent 层 fake REAL（真实层）；
- A/B 与产品闭环：scripts/ab_compare.py + Gate 2（真实 K3 已验收）。
本文件只保留完全不依赖已删架构的 K0（网关层实测探针）。
"""

from __future__ import annotations

import os

import pytest

KEY = os.environ.get("ROSCLAW_KIMI_API_KEY", "")
BASE_URL = os.environ.get("ROSCLAW_KIMI_BASE_URL", "https://api.kimi.com/coding/v1")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not KEY, reason="ROSCLAW_KIMI_API_KEY not set"),
]


@pytest.fixture
async def gateway():
    from rosclaw.agentd.models.gateway import OpenAICompatGateway
    from rosclaw.agentd.models.profiles import kimi_code_k3_profile

    profile = kimi_code_k3_profile(base_url=BASE_URL)
    gw = OpenAICompatGateway(profile)
    yield gw
    await gw.close()


async def test_k0_api_probe(gateway) -> None:
    from rosclaw.agentd.models.gateway import ModelTurnRequest, StrictTool

    probe = await gateway.probe()
    assert probe.reachable, f"endpoint unreachable: {probe.error}"
    assert probe.expected_model_present is not False, f"k3 not visible: {probe.models_visible}"
    assert probe.chat_ok, f"chat probe failed: {probe.error}"
    assert probe.tool_call_ok, f"strict tool call failed: {probe.error}"

    # Parallel tool calls.
    ping = StrictTool(
        name="ping",
        description="ping",
        parameters={
            "type": "object",
            "properties": {"echo": {"type": "boolean"}},
            "required": ["echo"],
            "additionalProperties": False,
        },
    )
    turn = await gateway.complete(
        ModelTurnRequest(
            system_prompt="Call the ping tool twice in parallel. No text answer.",
            messages=[{"role": "user", "content": "ping x2"}],
            tools=[ping],
            tool_choice="required",
            max_output_tokens=512,
        )
    )
    assert turn.tool_calls, "no tool calls emitted"
    assert turn.provider_request_id, "request id missing for diagnosis"
    assert turn.usage.total_tokens > 0
    # Tool result回填: complete assistant message must round-trip.
    messages = [dict(turn.assistant_message)]
    for call in turn.tool_calls:
        messages.append({"role": "tool", "tool_call_id": call.call_id, "content": '{"ok": true}'})
    follow = await gateway.complete(
        ModelTurnRequest(
            system_prompt="Reply with exactly: done",
            messages=messages,
            max_output_tokens=256,
        )
    )
    assert follow.finish_reason in ("stop", "length", "tool_calls")
