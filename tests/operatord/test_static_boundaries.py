"""T0（二次复核）：静态边界测试——特权方法与决定面不泄漏到 socket。

- daemon 内部 arm/permit/disarm 方法（P0-4）不在 socket dispatch；
- socket method allowlist 不含任何内部方法名；
- agentd 投影 socket 不服务 decide/revoke/estop；
- enrollment 私钥文件名只出现在 operatord 模块（agentd 不读私钥）。
"""

from __future__ import annotations

import inspect
from pathlib import Path

from rosclaw.daemon import server as daemon_server

SRC = Path(__file__).resolve().parents[2] / "src"


def test_internal_privileged_methods_not_socket_exposed() -> None:
    internal = {
        "_arm_after_operator_decision",
        "_issue_permit_after_operator_decision",
        "_disarm_after_operator_rollback",
        "_arm_core",
        "_issue_permit_core",
        "_disarm_core",
    }
    for name in internal:
        assert name not in daemon_server._ALLOWED_METHODS, f"{name} 绝不允许经 IPC"
    dispatch_source = inspect.getsource(daemon_server.RosclawDaemon._dispatch)
    for name in internal:
        assert name not in dispatch_source, f"dispatch 不得直调 {name}"


def test_decide_requires_proof_param() -> None:
    from rosclaw.daemon.service import DaemonControlPlane

    signature = inspect.signature(DaemonControlPlane.decide_operator_proposal)
    assert "proof" in signature.parameters
    # 旧的弱参数不得复活。
    for legacy in ("operator_proof", "challenge_nonce", "action_intent_hash_value"):
        assert legacy not in signature.parameters


def test_agent_projection_refuses_decision_methods() -> None:
    from rosclaw.agentd import operator_socket

    source = inspect.getsource(operator_socket.OperatorSocketServer._dispatch)
    assert "approvals.decide" in source  # 只作为拒绝分支出现
    assert "is not served by agentd" in source
    # 不得存在真正的 decide 调用路径。
    assert "service.decide_approval(\n" not in source.split("approvals.apply_decision")[0]


def test_agentd_never_reads_operator_private_key() -> None:
    """agentd 源码不得引用 operator 私钥文件或加载函数。"""
    agentd = SRC / "rosclaw" / "agentd"
    offenders = []
    for path in agentd.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in ("operator-identity.json", "load_identity", "private_key_pem"):
            if marker in text:
                offenders.append(f"{path.name}:{marker}")
    assert not offenders, f"agentd 不得接触 operator 私钥材料: {offenders}"
