"""确定性类 audit（PR-MH4，规格 §18/§56）：A18/A19/A20。"""

from __future__ import annotations

import hashlib
from typing import Any

from rosclaw.contracts.common import canonical_json
from rosclaw.sim.audit.context import AuditContext


def a18_reset_determinism(ctx: AuditContext) -> dict[str, Any]:
    """A18：两次 fresh reset（qpos0 + mj_forward）状态必须逐字节一致。"""
    first = ctx.fresh_data()
    second = ctx.fresh_data()
    identical = (
        list(first.qpos) == list(second.qpos)
        and list(first.qvel) == list(second.qvel)
        and list(first.ctrl) == list(second.ctrl)
    )
    status = "PASS" if identical else "FAIL"
    return {
        "status": status,
        "violations": [] if identical else [{"reason": "reset_nondeterministic"}],
    }


def a19_replay_determinism(ctx: AuditContext) -> dict[str, Any]:
    """A19：相同模型/初始状态/controller 两次 hold rollout 的
    states digest 必须一致。"""

    def _digest_once() -> str:
        states: list[list[float]] = []

        def visit(data, step) -> None:  # noqa: ANN001
            states.append([float(v) for v in data.qpos])

        ctx.sweep(0.2, visit, ctrl={"kind": "hold"})
        return hashlib.sha256(canonical_json(states).encode("utf-8")).hexdigest()

    digests = [_digest_once(), _digest_once()]
    identical = digests[0] == digests[1]
    status = "PASS" if identical else "FAIL"
    return {
        "status": status,
        "violations": [] if identical else [{"reason": "replay_nondeterministic"}],
        "detail": {"digests": digests},
    }


def a20_state_model_mismatch(ctx: AuditContext) -> dict[str, Any]:
    """A20：外部 state_ref 必须绑定当前 model_digest（fail closed）。"""
    if ctx.state_ref is None or ctx.restore_fn is None:
        return {"status": "PASS", "violations": [], "detail": {"note": "no_external_state"}}
    try:
        ctx.restore_fn(ctx.state_ref)
    except ValueError as exc:
        return {
            "status": "FAIL",
            "violations": [{"reason": "state_model_mismatch", "error": str(exc)}],
        }
    return {"status": "PASS", "violations": []}
