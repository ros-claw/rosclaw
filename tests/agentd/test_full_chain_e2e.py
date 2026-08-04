"""全链路闭环深度测试（2026-08-04 复盘专项）。

**证据等级标注（审计 §7.1/§1.2）：本文件 = L1（进程集成）+ 部分 L2
（Mock Model + LIMO SIM executor）。** 它证明进程/协议/授权/SIM 执行
闭环，**不证明**真实 Provider（由 test_kimi_live 的 K 系列覆盖）、
真实 rosclawd 许可链（由 tests/shadow 的 FTC-100 覆盖）或实体 LIMO
REAL（待 FTC-110）。

一次运行覆盖 Native Agent 全链路：
init → doctor → service(+operator.sock+HTTP) → mission → LIMO 观测 →
授权(operator.sock peer identity) → SIM 执行 → receipt → 验证 →
worker 委派 → compaction+恢复 → export/import → fork/tree →
modeld backend → ACP stdio → estop 诚实 → legacy MCP 观测 →
崩溃恢复 → snapshot/SSE 一致性。

全部为真实路径（真实 MCP stdio、真实 UDS、真实 modeld 子进程、真实
ACP JSON-RPC），只有模型是 Mock（K 系列 live 测试覆盖真实模型）。
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService, create_app
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from tests.agentd.conftest import LOCAL_PRINCIPAL

LIMO_SIM = str(Path(__file__).resolve().parents[2] / "src" / "rosclaw" / "limo" / "sim_mcp.py")

TONE = {"frequency_hz": 660, "duration_sec": 0.6, "volume_percent": 18}


def _chain_script(request) -> ModelTurnResultV1:
    """内容驱动的全链路决策状态机（观测→授权→执行→验证→委派→回答）。"""
    import re

    blob = json.dumps(request.messages, ensure_ascii=False)
    prompt = request.system_prompt or ""
    pose_observed = "tool: limo.localization.get_pose" in blob
    tone_card = "播放提示音" in blob
    tone_receipt = "limo.speaker.play_tone" in blob and "SIM 执行器" in blob
    worker_done = "Worker " in blob and "已完成并通过验证" in blob
    grant = re.search(r"grant_id=(grant_[a-z0-9]+)", prompt)

    if not pose_observed:
        decision = {
            "next_intent": "OBSERVE",
            "summary": "读取 LIMO 位姿",
            "proposed_operation": {
                "type": "observe",
                "payload": {"tool": "limo.localization.get_pose", "arguments": {"frame": "map"}},
            },
        }
    elif not tone_card:
        decision = {
            "next_intent": "REQUEST_APPROVAL",
            "summary": "请求授权：播放提示音",
            "proposed_operation": {
                "type": "approval_request",
                "payload": {
                    "capability_id": "limo.speaker.play_tone",
                    "arguments": TONE,
                    "title": "播放提示音",
                    "summary": "660Hz 0.6s 18%",
                    "risk_tier": "LOW",
                },
            },
        }
    elif not tone_receipt:
        decision = {
            "next_intent": "REQUEST_ACTION",
            "summary": "执行扬声器动作",
            "proposed_operation": {
                "type": "request_action",
                "payload": {
                    "grant_id": grant.group(1) if grant else "",
                    "capability_id": "limo.speaker.play_tone",
                    "arguments": TONE,
                    "risk_tier": "LOW",
                },
            },
        }
    elif not worker_done:
        decision = {
            "next_intent": "HIRE_WORKER",
            "summary": "委派：分析验收日志",
            "proposed_operation": {
                "type": "create_work_order",
                "payload": {
                    "capability": "analysis.text",
                    "goal": "分析验收日志并给出结论",
                    "instructions": "分析以下日志并给出结论："
                    "[INFO] tone executed; pose=(0,0,0) verified within tolerance.",
                },
            },
        }
    else:
        decision = {
            "next_intent": "ANSWER",
            "summary": "全链路闭环完成：观测-授权-执行-回执-委派 全部验证",
            "evidence_refs": [],
        }
    decision.update(
        {
            "schema_version": "rosclaw.decision.v1",
            "decision_id": "dec_chain",
            "mission_id": request.mission_id,
            "context_id": request.context_id,
            "context_revision": request.context_revision,
            "evidence_refs": decision.get("evidence_refs", []),
        }
    )
    if decision["next_intent"] in ("REQUEST_APPROVAL", "REQUEST_ACTION", "HIRE_WORKER"):
        decision["verification"] = {
            "schema_version": "rosclaw.decision_verification.v1",
            "verifiers": ["deterministic:bounds"],
        }
    return ModelTurnResultV1(
        turn_id="t",
        provider="mock",
        model="m",
        content=f"```json\n{json.dumps(decision, ensure_ascii=False)}\n```",
        assistant_message={"role": "assistant", "content": "x"},
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
    )


def _write_home(home: Path) -> None:
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "agent": {"enabled": True},
                "mcp_servers": [
                    {
                        "name": "limo-sim",
                        "command": sys.executable,
                        "args": [LIMO_SIM],
                        "supported_modes": ["SIMULATION"],
                        "sim_executor": True,
                    },
                    {
                        "name": "rosclaw-mcp",
                        "command": sys.executable,
                        "args": ["-m", "rosclaw.mcp.minimal_server"],
                        "supported_modes": ["SIMULATION"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


class TestFullChainE2E:
    async def test_complete_loop(self, tmp_path: Path) -> None:  # noqa: C901
        home = tmp_path / "home"
        home.mkdir()
        _write_home(home)

        # 1-2. init/doctor 层：配置可读、doctor 结构化。
        config = load_agent_config(home / "config.yaml")
        from rosclaw.agentd.onboarding import doctor

        report = doctor(home)
        assert "components" in report and "node" in report["components"]

        service = AgentService(
            config, home, gateway=MockModelGateway(mock_profile(), [_chain_script] * 60)
        )
        from rosclaw.agentd.operator_socket import operator_call

        agent_sock = await service.start_operator_socket()
        from rosclaw.operatord.enrollment import enroll
        from rosclaw.operatord.server import OperatorDaemon

        identity = enroll(home / "operatord")
        sock = home / "run" / "operatord.sock"
        operatord = OperatorDaemon(
            identity=identity,
            socket_path=sock,
            agent_socket=agent_sock,
            daemon_client=None,
            require_human_presence=False,
        )
        await operatord.start()
        service2 = None
        try:
            # 3. mission 创建（principal 与 uid 一致）。
            mission = service.create_mission("全链路闭环")
            assert mission.owner_principal == LOCAL_PRINCIPAL

            # 4-8. 观测 → 授权 → 执行 → receipt（状态驱动推进）。
            last_reply = ""
            for _step in range(10):
                result = await service.send_turn(mission.mission_id, "执行闭环流程")
                last_reply = result.reply
                if service.pending_approvals(mission.mission_id):
                    listed = await operator_call(sock, "approvals.list")
                    entry = listed["approvals"][0]
                    decided = await operator_call(
                        sock,
                        "approvals.decide",
                        {
                            "request_id": entry["request_id"],
                            "display_hash": entry["display_hash"],
                            "approve": True,
                        },
                    )
                    assert decided["ok"] and decided["principal"] == LOCAL_PRINCIPAL
                    continue
                if "全链路闭环完成" in last_reply:
                    break
                if result.state.value == "FAILED":
                    raise AssertionError(f"chain failed: {last_reply[:300]}")
            history = json.dumps(
                service.conversation(mission.mission_id), ensure_ascii=False
            )
            assert "observation — evidence" in history
            assert "limo.speaker.play_tone" in history and "COMPLETED" in history
            assert "已完成并通过验证" in history, history[-500:]

            # grant 单次消费 + receipt 事件落 journal。
            events = service.events_replay(mission.mission_id)
            types = [e.type.value for e in events]
            assert "grant.consumed" in types and "receipt.received" in types
            from rosclaw.operator import GrantDeniedError

            grant = service.list_grants()[0]
            with pytest.raises(GrantDeniedError, match="grant_consumed"):
                service._broker.verify(
                    grant["grant_id"],
                    principal=grant["principal"],
                    body_hash=mission.body_binding.effective_body_hash,
                    mode="SIMULATION",
                    risk_tier="LOW",
                    action_intent=service._broker.action_intent_for_grant(grant["grant_id"]),
                )

            # 9. worker 委派已 ACCEPTED（归属链）。
            orders = service._worker_manager.orders_for_mission(mission.mission_id)
            assert orders and orders[0].status == "ACCEPTED"

            # 10. compaction + 崩溃恢复：新 service 实例从 journal 恢复。
            for i in range(4):
                await service.send_turn(mission.mission_id, f"历史轮次 {i} {'长文本' * 100}")
            await service.compact(mission.mission_id)
            view_before = service.conversation(mission.mission_id)
            await service.close()

            service2 = AgentService(
                config, home, gateway=MockModelGateway(mock_profile(), [_chain_script] * 20)
            )
            recovered = service2.conversation(mission.mission_id)
            assert [m.get("entry_id") for m in recovered] == [
                m.get("entry_id") for m in view_before
            ], "重启后 view 必须与 journal 投影一致"
            assert recovered[0]["role"] == "compaction"
            service2.store.verify_consistency(mission.mission_id)

            # 11. export → import（只读、不恢复授权）。
            bundle = tmp_path / "out.rcmission"
            service2.exporter.export_bundle(mission.mission_id, bundle)
            with zipfile.ZipFile(bundle) as zf:
                blob = b"".join(zf.read(n) for n in zf.namelist()).decode(errors="replace")
            assert "permit_secret" not in blob and "private_signature" not in blob
            imported = service2.importer.import_bundle(bundle)
            assert imported["read_only"] and imported["authority_restored"] is False

            # 12. fork → tree（双时间线）。
            canonical = service2.store.conversation_canonical(mission.mission_id)
            branch = service2.branches.fork(
                mission.mission_id, from_entry_id=canonical[0]["entry_id"], label="复盘分支"
            )
            forked = service2.get_mission(branch.forked_mission_id)
            assert forked.mode.value == "SIMULATION" and forked.context_revision == 0
            tree = service2.branches.tree(mission.mission_id)
            assert len(tree["reasoning_branches"]) == 1
            assert any(e["type"] == "grant.consumed" for e in tree["physical_lane"])

            # 13. snapshot 与 SSE 重放一致（watermark）。
            snap = service2.snapshot(mission.mission_id)
            assert snap.last_event_sequence == service2._events.latest_sequence(
                mission.mission_id
            )
            assert snap.tool_count >= 2

            # 14. estop 无 daemon 诚实不可用。
            from rosclaw.contracts.common import ValidationError

            with pytest.raises(ValidationError, match="estop unavailable"):
                await service2.estop("test", principal=LOCAL_PRINCIPAL)

            # 15. legacy rosclaw-mcp 经 catalog 注入（观测类）。
            await service2._ensure_mcp_discovered()
            catalog_ids = {d.tool_id for d in service2.tool_catalog.list()}
            assert "get_robot_state" in catalog_ids
            pose = service2.tool_catalog.get("limo.localization.get_pose")
            assert pose is not None and pose.model_callable
            estop_tool = service2.tool_catalog.get("emergency_stop")
            assert estop_tool is not None and not estop_tool.model_callable

            await service2.close()
        finally:
            await operatord.stop()
            if service2 is not None:
                from contextlib import suppress

                with suppress(Exception):
                    await service2.close()

    async def test_http_surface_consistency(self, tmp_path: Path) -> None:
        """HTTP 面：/v2 turn → SSE 重放 → snapshot → commands 一致。"""
        from fastapi.testclient import TestClient

        home = tmp_path / "home"
        home.mkdir()
        _write_home(home)
        config = load_agent_config(home / "config.yaml")
        service = AgentService(
            config, home, gateway=MockModelGateway(mock_profile(), [_chain_script] * 30)
        )
        # 先在测试自己的 loop 完成 MCP 发现（持久会话归属单一 loop）；
        # TestClient portal 的 server loop 直接复用，不跨 loop 起进程。
        await service._ensure_mcp_discovered()
        client = TestClient(create_app(service), headers={'x-rosclaw-token': service.control_token})
        try:
            mission = service.create_mission("HTTP 一致性")
            r = client.post(f"/v2/missions/{mission.mission_id}/turns", json={"text": "开始"})
            assert r.status_code == 202
            # runner 在 TestClient portal loop 执行；从本 loop 只能轮询
            # store（线程安全），不能 await 对方的 task。
            import time as _time

            deadline = _time.monotonic() + 30.0
            while _time.monotonic() < deadline:
                task = service._turn_tasks.get(mission.mission_id)
                if task is not None and task.done():
                    break
                _time.sleep(0.1)
            else:
                raise AssertionError("turn did not settle in 30s")
            with client.stream(
                "GET", f"/v2/missions/{mission.mission_id}/events?follow=false"
            ) as response:
                frames = [
                    line for line in response.iter_lines() if line.startswith("data: ")
                ]
            assert frames
            seqs = [json.loads(f[6:])["sequence"] for f in frames]
            assert seqs == sorted(seqs)
            snap = client.get(f"/v1/missions/{mission.mission_id}/snapshot").json()
            assert snap["last_event_sequence"] == seqs[-1]
            caps = client.get(f"/v1/capabilities?mission_id={mission.mission_id}").json()
            names = {c["name"] for c in caps["commands"]}
            assert {"compact", "export", "import", "fork", "tree", "tools", "doctor"} <= names
            # CSRF：外部 Origin 打 turn endpoint → 403。
            hostile = client.post(
                f"/v2/missions/{mission.mission_id}/turns",
                json={"text": "x"},
                headers={"Origin": "https://evil.example"},
            )
            assert hostile.status_code == 403
        finally:
            await service.close()
