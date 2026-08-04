"""FTC-100（审计 §7.2/E1）：LIMO SHADOW 全链路。

真实 rosclawd（in-process）+ 真实 operatord（enrollment + proof）+
真实 daemon client RPC + LIMO SHADOW executor（actuation 硬阻断）：

proposal → operatord human decision（proof 经 rosclawd ACL）→
Permit/Lease → SHADOW action → SHADOW receipt（evidence_domain=shadow,
actuated=false）→ 公开回执验证。

攻击面：agentd 直接 decide → PERMISSION_DENIED；伪造 proof → 拒；
display hash 篡改 → 拒。
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rosclaw.daemon.client import DaemonClient
from rosclaw.daemon.ledger import DaemonLedger
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import DaemonControlPlane
from rosclaw.kernel.contracts import (
    ActionEnvelope,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)
from rosclaw.operatord.enrollment import enroll
from tests.agentd.conftest import LOCAL_PRINCIPAL
from tests.daemon.test_operator_proposals import _runtime


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
        """FTC-100：proposal→decide→permit→SHADOW receipt，驱动确认无真实动作。"""
        client, tmp = shadow_daemon
        # 1. enrollment + daemon ACL 登记（operatord 职责）。
        enrollment = enroll(tmp / "operatord")
        registered = client.register_operator_enrollment(
            enrollment.enrollment_id, enrollment.key.hex()
        )
        assert registered["registered"]
        assert registered["fingerprint"] == enrollment.fingerprint

        # 2. agentd 侧创建 proposal（公开视图）。
        action = _action(
            "act-shadow-tone",
            "limo.speaker.play_tone",
            {"frequency_hz": 660, "duration_sec": 0.6, "volume_percent": 18},
            ExecutionMode.SHADOW,
        )
        public = client.create_operator_proposal(
            action, display={"title": "播放提示音 660Hz 0.6s 18%"}, ttl_sec=60.0
        )["proposal"]
        assert "challenge_nonce" not in public
        assert "permit" not in str(public).lower()

        # 3. operatord 决定（human decision + proof → rosclawd ACL）。
        trusted = client.list_pending_operator_proposals()["proposals"][0]
        from rosclaw.operatord.enrollment import sign_decision_proof

        decided_at = datetime.now(UTC).isoformat()
        display_hash = "limo-card-hash"
        proof = sign_decision_proof(
            enrollment,
            request_id=public["request_id"],
            approve=True,
            nonce="nonce-shadow-1",
            decided_at=decided_at,
            display_hash=display_hash,
        )
        decided = client.decide_operator_proposal(
            public["request_id"],
            decision="ACCEPT",
            principal_id=LOCAL_PRINCIPAL,
            challenge_nonce=trusted["challenge_nonce"],
            action_intent_hash=trusted["action_intent_hash"],
            channel="rosclaw_operatord",
            reason="shadow chain acceptance",
            operator_proof=proof,
            enrollment_id=enrollment.enrollment_id,
            display_hash=display_hash,
            decided_at=decided_at,
        )
        assert decided["proposal"]["state"] == "SUBMITTED"

        # 4. SHADOW receipt：evidence 严格分域 + 硬阻断。
        terminal = client.wait_for_action(public["action_id"], timeout_sec=10.0)
        receipt = terminal["receipt"]
        assert receipt["final_state"] == "COMPLETED"
        assert receipt["evidence_domain"].upper() == "SHADOW"
        sim = receipt.get("simulation_result") or {}
        assert sim.get("actuated") is False, "SHADOW 绝不允许真实驱动"
        assert sim.get("usable_for_real_execution") is False
        assert sim.get("planned_ros_commands"), "SHADOW 必须携带可审计的拟执行命令"
        verification = receipt.get("verification_result") or {}
        assert verification.get("actuation_gate") == "hard_blocked"

    def test_pose_shadow_validates_and_blocks_bad_args(self, shadow_daemon) -> None:
        client, tmp = shadow_daemon
        enrollment = enroll(tmp / "operatord")
        client.register_operator_enrollment(enrollment.enrollment_id, enrollment.key.hex())
        bad = _action(
            "act-shadow-bad",
            "limo.speaker.play_tone",
            {"frequency_hz": 999_999, "duration_sec": 0.6, "volume_percent": 18},
            ExecutionMode.SHADOW,
        )
        public = client.create_operator_proposal(bad, display={"title": "bad"}, ttl_sec=60.0)[
            "proposal"
        ]
        trusted = client.list_pending_operator_proposals()["proposals"][0]
        client.decide_operator_proposal(
            public["request_id"],
            decision="ACCEPT",
            principal_id=LOCAL_PRINCIPAL,
            challenge_nonce=trusted["challenge_nonce"],
            action_intent_hash=trusted["action_intent_hash"],
            channel="rosclaw_operatord",
            reason="x",
        )
        terminal = client.wait_for_action(public["action_id"], timeout_sec=10.0)
        assert terminal["receipt"]["final_state"] == "FAILED"
        assert terminal["receipt"]["errors"], "非法参数必须失败并给出原因"

    def test_agentd_cannot_decide_without_proof(self, shadow_daemon) -> None:
        """FTC-050（部分）：非 daemon UID 的调用方必须持有效 enrollment
        proof——伪造 id / 无 proof / 篡改 display_hash 全被拒。

        进程内测试无法伪造 SO_PEERCRED，直接在 service ACL 层用
        fabricated peer（uid != euid）验证拒绝语义。
        """
        import os

        from rosclaw.daemon.protocol import PeerCredentials
        from rosclaw.operatord.enrollment import enroll, sign_decision_proof

        client, tmp = shadow_daemon
        enrollment = enroll(tmp / "operatord")
        client.register_operator_enrollment(enrollment.enrollment_id, enrollment.key.hex())
        # fixture 的 daemon service 无法直接拿到——通过 ACL 语义另行构造：
        from rosclaw.daemon.service import DaemonControlPlane

        # fabricate a non-daemon peer（uid+1 != euid）。
        foreign = PeerCredentials(pid=99999, uid=os.geteuid() + 1, gid=0)

        plane = DaemonControlPlane(runtime=None)
        plane.register_operator_enrollment(
            enrollment.enrollment_id, key_hex=enrollment.key.hex(),
            peer=PeerCredentials(pid=os.getpid(), uid=os.geteuid(), gid=0),
        )
        # 1) 无 proof → 拒。
        assert not plane._decide_acl_allows(
            foreign, "", request_id="r1", approve=True, nonce="n",
            decided_at="t", enrollment_id=enrollment.enrollment_id, display_hash="h",
        )
        # 2) 未登记的 enrollment_id → 拒。
        proof = sign_decision_proof(
            enrollment, request_id="r1", approve=True, nonce="n",
            decided_at="t", display_hash="h",
        )
        assert not plane._decide_acl_allows(
            foreign, proof, request_id="r1", approve=True, nonce="n",
            decided_at="t", enrollment_id="oen_mallory", display_hash="h",
        )
        # 3) 篡改 display_hash → 拒。
        assert not plane._decide_acl_allows(
            foreign, proof, request_id="r1", approve=True, nonce="n",
            decided_at="t", enrollment_id=enrollment.enrollment_id, display_hash="TAMPERED",
        )
        # 4) 合法 proof → 允许（operatord 正常路径）。
        assert plane._decide_acl_allows(
            foreign, proof, request_id="r1", approve=True, nonce="n",
            decided_at="t", enrollment_id=enrollment.enrollment_id, display_hash="h",
        )
        # 5) daemon 服务 UID 直通（生产侧 rosclawd 服务账户）。
        service_peer = PeerCredentials(pid=1, uid=os.geteuid(), gid=0)
        assert plane._decide_acl_allows(
            service_peer, "", request_id="r1", approve=True, nonce="n",
            decided_at="t", enrollment_id="", display_hash="",
        )
