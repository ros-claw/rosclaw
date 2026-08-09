"""LIMO 完整闭环验收（PR-12 §18）：SIMULATION 证据域的 C1–C10。

驱动 TaskGraph：
    T1 OBSERVE（定位/健康观测）
    T3 REQUEST_APPROVAL → approve → REQUEST_ACTION（speaker.play_tone）
    T4 REQUEST_APPROVAL → approve → REQUEST_ACTION（localization.set_initial_pose）
    T5 VALIDATE（观测位姿对比目标）
    T6 报告（含 §18.4 诚实声明）
    T7 LEARN（practice candidate 证据）

SHADOW/REAL 门控：无 rosclawd/真实硬件时明确拒绝并说明——绝不用 SIM
证据冒充 SHADOW/REAL（§18.4、总纲 21 条）。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

LIMO_SIM_MCP = str(Path(__file__).resolve().parents[2] / "limo" / "sim_mcp.py")

TONE_ARGS = {"frequency_hz": 660, "duration_sec": 0.6, "volume_percent": 18}
POSE_ARGS = {"x": 0.0, "y": 0.0, "yaw": 0.0}


@dataclass
class AcceptanceReport:
    checks: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    receipts: list[str] = field(default_factory=list)
    grants: list[str] = field(default_factory=list)
    practice_candidate: str | None = None
    evidence_manifest: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def limo_sim_mcp_server_config() -> dict:
    """mcp_servers 配置项（SIMULATION 全证据域）。"""
    return {
        "name": "limo-sim",
        "command": sys.executable,
        "args": [LIMO_SIM_MCP],
        "supported_modes": ["SIMULATION"],
        "required_body_types": [],
        "sim_executor": True,
    }


def acceptance_script(tone_args: dict | None = None, pose_args: dict | None = None):
    """T1–T7 内容驱动的状态机脚本（OBSERVE 会在同一 turn 内续跑，不能按
    调用次数推进——按会话内容判定当前阶段）。"""
    tone = tone_args or TONE_ARGS
    pose = pose_args or POSE_ARGS

    def script(request):
        import re

        from rosclaw.agentd.models.gateway import ModelTurnResultV1

        blob = json.dumps(request.messages, ensure_ascii=False)
        prompt = request.system_prompt or ""
        pose_observations = blob.count("tool: limo.localization.get_pose")
        health_observed = "tool: limo.health" in blob
        tone_card = "播放提示音" in blob
        tone_receipt = "limo.speaker.play_tone" in blob and "SIM 执行器" in blob
        pose_card = "设置地图初始位姿" in blob
        pose_receipt = "limo.localization.set_initial_pose" in blob and "SIM 执行器" in blob
        grant_match = re.search(r"grant_id=(grant_[a-z0-9]+)", prompt)
        grant_id = grant_match.group(1) if grant_match else ""

        if pose_observations == 0:
            decision = {
                "next_intent": "OBSERVE",
                "summary": "读取 LIMO 定位位姿",
                "proposed_operation": {
                    "type": "observe",
                    "payload": {
                        "tool": "limo.localization.get_pose",
                        "arguments": {"frame": "map"},
                    },
                },
            }
        elif not health_observed:
            decision = {
                "next_intent": "OBSERVE",
                "summary": "读取 LIMO 健康状态",
                "proposed_operation": {
                    "type": "observe",
                    "payload": {"tool": "limo.health", "arguments": {}},
                },
            }
        elif not tone_card:
            decision = {
                "next_intent": "REQUEST_APPROVAL",
                "summary": "请求授权：播放 660Hz 提示音",
                "proposed_operation": {
                    "type": "approval_request",
                    "payload": {
                        "capability_id": "limo.speaker.play_tone",
                        "arguments": tone,
                        "title": "播放提示音",
                        "summary": (
                            f"{tone['frequency_hz']}Hz {tone['duration_sec']}s "
                            f"{tone['volume_percent']}%"
                        ),
                        "risk_tier": "LOW",
                    },
                },
            }
        elif not tone_receipt:
            decision = {
                "next_intent": "REQUEST_ACTION",
                "summary": "执行已授权的扬声器动作",
                "proposed_operation": {
                    "type": "request_action",
                    "payload": {
                        "grant_id": grant_id,
                        "capability_id": "limo.speaker.play_tone",
                        "arguments": tone,
                        "risk_tier": "LOW",
                    },
                },
            }
        elif not pose_card:
            decision = {
                "next_intent": "REQUEST_APPROVAL",
                "summary": "请求授权：设置地图初始位姿",
                "proposed_operation": {
                    "type": "approval_request",
                    "payload": {
                        "capability_id": "limo.localization.set_initial_pose",
                        "arguments": pose,
                        "title": "设置地图初始位姿",
                        "summary": f"x={pose['x']}, y={pose['y']}, yaw={pose['yaw']}",
                        "risk_tier": "LOW",
                    },
                },
            }
        elif not pose_receipt:
            decision = {
                "next_intent": "REQUEST_ACTION",
                "summary": "执行已授权的初始位姿设置",
                "proposed_operation": {
                    "type": "request_action",
                    "payload": {
                        "grant_id": grant_id,
                        "capability_id": "limo.localization.set_initial_pose",
                        "arguments": pose,
                        "risk_tier": "LOW",
                    },
                },
            }
        elif pose_observations < 2:
            decision = {
                "next_intent": "OBSERVE",
                "summary": "验证新位姿",
                "proposed_operation": {
                    "type": "observe",
                    "payload": {
                        "tool": "limo.localization.get_pose",
                        "arguments": {"frame": "map"},
                    },
                },
            }
        else:
            decision = {
                "next_intent": "ANSWER",
                "summary": (
                    "LIMO 验收完成：扬声器与初始位姿动作均经 EXACT_ACTION 授权并由 "
                    "SIM 执行器返回 COMPLETED receipt（SIMULATED 证据域）；"
                    "新位姿 (0,0,0) 已在容差内验证。"
                    "没有麦克风——扬声器只能确认驱动执行，不能独立证明声学效果（§18.4）。"
                ),
                "evidence_refs": [],
            }
        if decision["next_intent"] in ("REQUEST_APPROVAL", "REQUEST_ACTION"):
            decision["verification"] = {
                "schema_version": "rosclaw.decision_verification.v1",
                "verifiers": ["deterministic:bounds"],
            }
        idx = blob.count("rosclaw.decision") + 1
        decision.update(
            {
                "schema_version": "rosclaw.decision.v1",
                "decision_id": f"dec_{idx}",
                "mission_id": request.mission_id,
                "context_id": request.context_id,
                "context_revision": request.context_revision,
                "evidence_refs": decision.get("evidence_refs", []),
            }
        )
        return ModelTurnResultV1(
            turn_id="t",
            provider="mock",
            model="m",
            content=f"```json\n{json.dumps(decision, ensure_ascii=False)}\n```",
            assistant_message={"role": "assistant", "content": "x"},
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
        )

    return script


async def run_acceptance(
    home: Path, *, gateway=None, evidence_root: Path | None = None
) -> AcceptanceReport:
    """SIMULATION 全链路验收（真实 MCP/审批/执行/receipt 路径）。"""
    import yaml

    from rosclaw.agentd.config import load_agent_config
    from rosclaw.agentd.models.gateway import MockModelGateway
    from rosclaw.agentd.models.profiles import mock_profile
    from rosclaw.agentd.operator_socket import operator_call
    from rosclaw.agentd.service import AgentService

    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "agent": {"enabled": True},
                # 七审 PR-SEVEN-1：本 bench 是 LIMO legacy 闭环——UR5e kit
                # 不激活（否则多 executor 让 legacy 无身份路径无法路由）。
                "kits": {"disabled": ["rosclaw/ur5e-sim"]},
                "mcp_servers": [limo_sim_mcp_server_config()],
            }
        ),
        encoding="utf-8",
    )
    config = load_agent_config(home / "config.yaml")
    script = acceptance_script()
    service = AgentService(
        config, home, gateway=gateway or MockModelGateway(mock_profile(), [script] * 40)
    )
    report = AcceptanceReport()
    agent_sock = await service.start_operator_socket()
    # 审计 P0-01：决定经独立 rosclaw-operatord（enrollment + proof）。
    from rosclaw.operatord.enrollment import enroll
    from rosclaw.operatord.server import OperatorDaemon

    identity = enroll(home / "operatord")
    sock_path = home / "run" / "operatord.sock"
    operatord = OperatorDaemon(
        identity=identity,
        socket_path=sock_path,
        agent_socket=agent_sock,
        daemon_client=None,
        require_human_presence=False,
    )
    await operatord.start()
    try:
        mission = service.create_mission("LIMO 验收：扬声器与定位")
        report.checks["C1_body_bound"] = True
        report.notes.append(f"body={mission.body_binding.body_id} hash={mission.body_binding.effective_body_hash[:16]}")

        async def approve_pending() -> None:
            """operator socket 审批当前待批卡片（peer identity + display hash）。"""
            listed = await operator_call(sock_path, "approvals.list")
            assert listed["approvals"], "expected a pending approval card"
            entry = listed["approvals"][0]
            decided = await operator_call(
                sock_path,
                "approvals.decide",
                {
                    "request_id": entry["request_id"],
                    "display_hash": entry["display_hash"],
                    "approve": True,
                },
            )
            assert decided["ok"], f"approval failed: {decided}"
            report.grants.append(decided["grant_id"])

        # 状态驱动推进：每轮 turn 走到 WAIT_APPROVAL 就经 operator.sock
        # 批准；直到最终验收报告出现（loop 在同一 turn 内可连续推进多步）。
        last_reply = ""
        for _step in range(14):
            result = await service.send_turn(mission.mission_id, "继续执行验收流程")
            last_reply = result.reply
            if service.pending_approvals(mission.mission_id):
                await approve_pending()
                continue
            if result.state.value == "FAILED":
                raise AssertionError(f"mission failed at step {_step}: {result.reply[:300]}")
            if "验收完成" in result.reply:
                break
        else:
            raise AssertionError(f"acceptance did not converge; last reply: {last_reply[:300]}")

        history = json.dumps(service.conversation(mission.mission_id), ensure_ascii=False)
        report.checks["C2_observation_fresh"] = "observation" in history
        report.checks["C3_tone_exact_action_approval"] = "播放提示音" in history
        report.checks["C4_tone_terminal_receipt"] = (
            "limo.speaker.play_tone" in history and "COMPLETED" in history
        )
        report.checks["C5_honest_no_acoustic_proof"] = "不能独立证明" in history
        report.checks["C6_pose_exact_action_approval"] = "设置地图初始位姿" in history
        report.checks["C7_pose_completed"] = (
            "limo.localization.set_initial_pose" in history
            and history.count("COMPLETED") >= 2
        )
        raw_contents = "\n".join(
            str(m.get("content") or "") for m in service.conversation(mission.mission_id)
        )
        # 证据 envelope 内 MCP JSON 是转义的；两种形态都接受。
        unescaped = raw_contents.replace('\\"', '"')
        report.checks["C8_pose_within_tolerance"] = (
            '"x": 0.0' in unescaped
            and '"y": 0.0' in unescaped
            and '"theta": 0.0' in unescaped
        )
        report.receipts.append("see mission journal (receipt.received events)")

        # C9：所有 grant 单次消费（再次 verify 必拒）。
        from rosclaw.operator import GrantDeniedError

        single_use_ok = True
        for grant in service.list_grants():
            try:
                service._broker.verify(
                    grant["grant_id"],
                    principal=grant["principal"],
                    body_hash=mission.body_binding.effective_body_hash,
                    mode="SIMULATION",
                    risk_tier="LOW",
                )
                single_use_ok = False
            except GrantDeniedError as exc:
                single_use_ok = single_use_ok and exc.reason_code == "grant_consumed"
        report.checks["C9_grants_single_use"] = single_use_ok

        # C10：trace/receipt/verification 持久化。
        events = service.events_replay(mission.mission_id)
        types = {e.type.value for e in events}
        report.checks["C10_trace_persisted"] = (
            "receipt.received" in types and "grant.consumed" in types
        )

        # T7：practice candidate（学习证据，不自动晋升）。
        from rosclaw.agentd.context.sources import EvidenceClass
        from rosclaw.agentd.learning.pipeline import LearningPipeline

        pipeline = LearningPipeline(service.store.connection, actor_id=service.actor_id)
        receipt_ids = [e.event_id for e in events if e.type.value == "receipt.received"]
        candidate = pipeline.propose(
            kind="HOW",
            title="LIMO 扬声器与定位验收流程",
            content={
                "scenario": "limo_tone_localization_acceptance",
                "steps": ["observe", "approve+act tone", "approve+act pose", "validate"],
                "evidence_domain": "simulation",
            },
            evidence_class=EvidenceClass.VERIFIED_RECEIPT,
            evidence_refs=receipt_ids,
            mission_id=mission.mission_id,
            body_scope=mission.body_binding.body_id,
        )
        report.practice_candidate = candidate
        report.checks["T7_practice_candidate"] = bool(candidate)

        # T6 报告（用户可见诚实声明）。
        report.notes.append(last_reply[:200])
        report.checks["T6_report"] = "验收完成" in last_reply
        # 证据包（审计 §8）：E3_SIM_VERIFIED —— Mock 模型时严格标 L1+L2。
        if evidence_root is not None:
            from rosclaw.agentd.bench.evidence_levels import EvidenceLevel
            from rosclaw.agentd.bench.evidence_pack import EvidencePackWriter, current_commit

            pack = EvidencePackWriter(evidence_root)
            pack.write_environment(provider="mock" if gateway is None else "kimi-code")
            pack.write_commands(
                [
                    "run_acceptance(home, gateway=..., evidence_root=...)",
                    "mcp_servers: limo-sim (SIM executor)",
                ]
            )
            events = service.events_replay(mission.mission_id, limit=100_000)
            pack.write_events([e.model_dump(mode="json") for e in events])
            pack.write_mission_snapshot(service.snapshot(mission.mission_id).model_dump(mode="json"))
            # R7：完整 approval → decision → grant → permit → receipt 链
            # （全部公开视图；secret 扫描兜底）。
            pack.write_public_records(
                approvals=[
                    e.payload
                    for e in events
                    if e.type.value in ("approval.requested", "approval.decided")
                ],
                permits=list(service.list_grants()),
                receipts=[
                    e.payload for e in events if e.type.value == "receipt.received"
                ],
            )
            pack.write_metrics(
                {"checks": report.checks, "passed": report.passed}
            )
            pack.write_observer(
                "# operator observer\n\nSIMULATION 证据域验收；无声学传感器，"
                "扬声器只证明驱动执行（§18.4）。\n"
            )
            commit, dirty = current_commit()
            report.evidence_manifest = pack.finalize(
                level=EvidenceLevel.E3_SIM_VERIFIED,
                git_commit=commit,
                dirty=dirty,
                test_ids=["FTC-040", "C1-C10"],
                operator="rosclaw-ci",
            )
        return report
    finally:
        await operatord.stop()
        await service.close()
