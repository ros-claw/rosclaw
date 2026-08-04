"""FTC-100（二次复核 R1 版）：LIMO SHADOW 全链路。

真实 rosclawd（in-process）+ 真实 Ed25519 operator identity +
真实 challenge.get → sign → decide（与 operatord 同一条协议路径）→
daemon 签名 DecisionReceiptV1 → Permit/Lease → SHADOW action →
SHADOW receipt（evidence_domain=shadow, actuated=false）→ 公开回执验证。

攻击面（fabricated foreign peer / 篡改字段）：
- 无 enrollment 的 identity 决定 → PERMISSION_DENIED；
- 篡改 challenge 任意字段 → OPERATOR_PROOF_CHALLENGE_MISMATCH；
- 伪造签名 → PERMISSION_DENIED；
- 空 registry 时非管理员注册 → PERMISSION_DENIED（无首调抢注窗口）；
- registry 持久化：重启后 enrollment/revoked 状态不变。
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rosclaw.daemon.client import DaemonClient, DaemonRequestError
from rosclaw.daemon.ledger import DaemonLedger
from rosclaw.daemon.protocol import PeerCredentials
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import ControlPlaneError, DaemonControlPlane
from rosclaw.kernel.contracts import (
    ActionEnvelope,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)
from tests.agentd.conftest import LOCAL_PRINCIPAL
from tests.daemon.test_operator_proposals import _runtime
from tests.operator_proof import (
    build_proof,
    decide_via_proof,
    make_identity,
    register_identity,
)


def _action(action_id: str, capability: str, arguments: dict, mode: ExecutionMode) -> ActionEnvelope:
    return ActionEnvelope(
        action_id=action_id,
        actor_id="rosclaw-native",
        agent_framework="rosclaw-native",
        session_id=f"session-{action_id}",
        body_id="limo-test",
        body_snapshot_hash="sha256:limo-body",
        capability_id=capability,
        arguments=arguments,
        execution_mode=mode,
        deadline_at=datetime.now(UTC) + timedelta(minutes=2),
        authorization=AuthorizationContext(
            principal_id="forged-agent", approved=True, approval_id="forged", scopes=["*"]
        ),
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.TASK_VERIFIED, timeout_sec=2.0
        ),
    )


@pytest.fixture
def shadow_daemon():
    from rosclaw.limo.shadow_executor import limo_shadow_executor

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        runtime = _runtime()
        for capability in ("limo.speaker.play_tone", "limo.localization.set_initial_pose"):
            runtime.action_gateway.register_executor(
                capability, ExecutionMode.SHADOW, limo_shadow_executor
            )
        with DaemonLedger(
            tmp / "state" / "ledger.sqlite3", key_path=tmp / "state" / "ledger.key"
        ) as ledger:
            daemon = RosclawDaemon(
                service=DaemonControlPlane(runtime=runtime, ledger=ledger),
                socket_path=tmp / "run" / "rosclawd.sock",
            )
            daemon.start()
            client = DaemonClient(socket_path=daemon.socket_path)
            try:
                yield client, tmp
            finally:
                daemon.stop()


class TestLimoShadowChain:
    def test_tone_shadow_full_chain(self, shadow_daemon) -> None:
        """FTC-100：enroll→register→proposal→challenge→sign→decide→
        daemon receipt→SUBMITTED→SHADOW receipt（actuation 硬阻断）。"""
        client, tmp = shadow_daemon
        # 1. Ed25519 identity + daemon 持久化 registry 登记。
        identity = make_identity(tmp / "operatord")
        registered = register_identity(client, identity)
        assert registered["registered"]
        assert registered["fingerprint"] == identity.fingerprint

        # 2. agentd 侧创建 proposal（公开视图无 nonce/permit）。
        action = _action(
            "act-shadow-tone",
            "limo.speaker.play_tone",
            {"frequency_hz": 660, "duration_sec": 0.6, "volume_percent": 18},
            ExecutionMode.SHADOW,
        )
        public = client.create_operator_proposal(
            action,
            display={"title": "播放提示音 660Hz 0.6s 18%", "risk_tier": "LOW"},
            ttl_sec=60.0,
            client_reference={"agent_request_id": "req-shadow-1", "mission_id": "mis-shadow-1"},
        )["proposal"]
        assert "challenge_nonce" not in public
        assert "permit" not in str(public).lower()

        # 3. challenge.get → Ed25519 sign → decide（operatord 协议路径）。
        decided = decide_via_proof(
            client, public["request_id"], identity, "ACCEPT", principal_id=LOCAL_PRINCIPAL
        )
        assert decided["proposal"]["state"] == "SUBMITTED"

        # 4. daemon 签名 receipt：字段与 proposal/client_reference 一致且
        # 签名可用 daemon 公钥验证。
        receipt = decided["decision_receipt"]
        assert receipt["protocol_version"] == "decision-receipt/1"
        assert receipt["decision"] == "ACCEPT"
        assert receipt["proposal_id"] == public["request_id"]
        assert receipt["agent_request_id"] == "req-shadow-1"
        assert receipt["mission_id"] == "mis-shadow-1"
        assert receipt["execution_mode"] == "SHADOW"
        assert receipt["capability_id"] == "limo.speaker.play_tone"
        assert receipt["operator_enrollment_id"] == identity.enrollment_id
        assert receipt["signature_b64"]
        from rosclaw.contracts.operator.decision import DecisionReceiptV1

        identity_info = client.daemon_identity()
        parsed = DecisionReceiptV1.from_dict(receipt)
        assert parsed.verify_signature(identity_info["public_key_pem"])
        assert receipt["daemon_key_id"] == identity_info["daemon_key_id"]

        # 5. SHADOW receipt：evidence 严格分域 + 硬阻断。
        terminal = client.wait_for_action(public["action_id"], timeout_sec=10.0)
        action_receipt = terminal["receipt"]
        assert action_receipt["final_state"] == "COMPLETED"
        assert action_receipt["evidence_domain"].upper() == "SHADOW"
        sim = action_receipt.get("simulation_result") or {}
        assert sim.get("actuated") is False, "SHADOW 绝不允许真实驱动"
        assert sim.get("usable_for_real_execution") is False
        assert sim.get("planned_ros_commands"), "SHADOW 必须携带可审计的拟执行命令"
        verification = action_receipt.get("verification_result") or {}
        assert verification.get("actuation_gate") == "hard_blocked"

    def test_decision_replay_never_double_decides(self, shadow_daemon) -> None:
        """同一 proof 的重放绝不能产生第二次决定/permit。"""
        client, tmp = shadow_daemon
        identity = make_identity(tmp / "operatord")
        register_identity(client, identity)
        action = _action(
            "act-shadow-replay",
            "limo.speaker.play_tone",
            {"frequency_hz": 440, "duration_sec": 0.3, "volume_percent": 10},
            ExecutionMode.SHADOW,
        )
        public = client.create_operator_proposal(
            action, display={"title": "replay probe"}, ttl_sec=60.0
        )["proposal"]
        challenge = client.get_operator_challenge(public["request_id"])["challenge"]
        proof = build_proof(identity, challenge, "ACCEPT")
        decided = client.decide_operator_proposal(
            public["request_id"],
            decision="ACCEPT",
            principal_id=LOCAL_PRINCIPAL,
            channel="rosclaw_operatord",
            reason="first",
            proof=proof,
        )
        assert decided["proposal"]["state"] == "SUBMITTED"
        with pytest.raises(DaemonRequestError) as replay:
            client.decide_operator_proposal(
                public["request_id"],
                decision="DECLINE",
                principal_id=LOCAL_PRINCIPAL,
                channel="rosclaw_operatord",
                reason="replay",
                proof=proof,
            )
        assert replay.value.code in {
            "PROPOSAL_ALREADY_DECIDED",
            "NONCE_REPLAY",
            "PROPOSAL_NOT_PENDING",
            "OPERATOR_PROOF_DECISION_MISMATCH",
        }

    def test_pose_shadow_validates_and_blocks_bad_args(self, shadow_daemon) -> None:
        client, tmp = shadow_daemon
        identity = make_identity(tmp / "operatord")
        register_identity(client, identity)
        bad = _action(
            "act-shadow-bad",
            "limo.speaker.play_tone",
            {"frequency_hz": 999_999, "duration_sec": 0.6, "volume_percent": 18},
            ExecutionMode.SHADOW,
        )
        public = client.create_operator_proposal(bad, display={"title": "bad"}, ttl_sec=60.0)[
            "proposal"
        ]
        decide_via_proof(
            client, public["request_id"], identity, "ACCEPT", principal_id=LOCAL_PRINCIPAL
        )
        terminal = client.wait_for_action(public["action_id"], timeout_sec=10.0)
        assert terminal["receipt"]["final_state"] == "FAILED"
        assert terminal["receipt"]["errors"], "非法参数必须失败并给出原因"

    def test_attack_matrix(self, shadow_daemon) -> None:
        """FTC-050（二次复核版）攻击矩阵。"""
        client, tmp = shadow_daemon
        foreign_uid = os.geteuid() + 1
        foreign = PeerCredentials(pid=99999, uid=foreign_uid, gid=0)

        # 空 registry：非管理员（foreign peer）注册 → PERMISSION_DENIED
        # （R2：无首调抢注窗口）。
        attacker = make_identity(tmp / "attacker", uid=foreign_uid)
        service = DaemonControlPlane(runtime=_runtime())
        service.start()
        with pytest.raises(ControlPlaneError) as denied_register:
            service.register_operator_enrollment(
                attacker.enrollment_id,
                public_key_pem=attacker.public_key_pem,
                operator_uid=foreign_uid,
                peer=foreign,
            )
        assert denied_register.value.code == "PERMISSION_DENIED"
        # foreign peer list pending → PERMISSION_DENIED（未登记）。
        with pytest.raises(ControlPlaneError) as denied_list:
            service.list_pending_operator_proposals(foreign)
        assert denied_list.value.code == "PERMISSION_DENIED"
        service.close()

        # 正经登记一个 operator，再测 proof 攻击。
        identity = make_identity(tmp / "operatord")
        register_identity(client, identity)
        action = _action(
            "act-shadow-attack",
            "limo.speaker.play_tone",
            {"frequency_hz": 660, "duration_sec": 0.6, "volume_percent": 18},
            ExecutionMode.SHADOW,
        )
        public = client.create_operator_proposal(
            action, display={"title": "attack probe"}, ttl_sec=60.0
        )["proposal"]
        challenge = client.get_operator_challenge(public["request_id"])["challenge"]

        # 1) 未登记 identity 的合法签名 → PERMISSION_DENIED。
        with pytest.raises(DaemonRequestError) as unregistered:
            client.decide_operator_proposal(
                public["request_id"],
                decision="ACCEPT",
                principal_id=LOCAL_PRINCIPAL,
                channel="x",
                reason="x",
                proof=build_proof(attacker, challenge, "ACCEPT"),
            )
        assert unregistered.value.code == "PERMISSION_DENIED"
        # 2) 篡改 challenge 字段 → OPERATOR_PROOF_CHALLENGE_MISMATCH。
        tampered = dict(challenge, display_hash="TAMPERED")
        with pytest.raises(DaemonRequestError) as tampered_err:
            client.decide_operator_proposal(
                public["request_id"],
                decision="ACCEPT",
                principal_id=LOCAL_PRINCIPAL,
                channel="x",
                reason="x",
                proof=build_proof(identity, tampered, "ACCEPT"),
            )
        assert tampered_err.value.code == "OPERATOR_PROOF_CHALLENGE_MISMATCH"
        # 3) 伪造签名字段 → PERMISSION_DENIED。
        forged = build_proof(identity, challenge, "ACCEPT")
        forged["signature_b64"] = "AAAA"
        with pytest.raises(DaemonRequestError) as forged_err:
            client.decide_operator_proposal(
                public["request_id"],
                decision="ACCEPT",
                principal_id=LOCAL_PRINCIPAL,
                channel="x",
                reason="x",
                proof=forged,
            )
        assert forged_err.value.code == "PERMISSION_DENIED"
        # 4) proof decision 与请求 decision 不一致 → DECISION_MISMATCH。
        mismatched = build_proof(identity, challenge, "DECLINE")
        with pytest.raises(DaemonRequestError) as mismatch_err:
            client.decide_operator_proposal(
                public["request_id"],
                decision="ACCEPT",
                principal_id=LOCAL_PRINCIPAL,
                channel="x",
                reason="x",
                proof=mismatched,
            )
        assert mismatch_err.value.code == "OPERATOR_PROOF_DECISION_MISMATCH"

    def test_registry_persists_across_restart(self, shadow_daemon) -> None:
        """R2：registry 持久化——重启后 enrollment/revoked 状态不变。"""
        client, tmp = shadow_daemon
        identity = make_identity(tmp / "operatord")
        register_identity(client, identity)
        from rosclaw.daemon.operator_registry import OperatorRegistry

        registry = OperatorRegistry(tmp / "state" / "operator-enrollments.json")
        record = registry.active(identity.enrollment_id)
        assert record is not None
        assert record.public_key_pem == identity.public_key_pem
        registry.revoke(identity.enrollment_id)
        reloaded = OperatorRegistry(tmp / "state" / "operator-enrollments.json")
        assert reloaded.active(identity.enrollment_id) is None
        assert reloaded.get(identity.enrollment_id).status == "revoked"
