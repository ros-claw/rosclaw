"""Strict replay fidelity 规则测试（MH10，0916 优化 §二.5，红→绿）。

只有 FULL_INTEGRATION 状态允许 RAW_EXACT；旧 v1（LEGACY_PARTIAL）
最多 SEMANTIC——不能让旧证据升级成不存在的强证据。
"""

from __future__ import annotations

import hashlib

from rosclaw.contracts.common import canonical_json
from rosclaw.sim.refs import make_ref


def _forge(backend, payload: dict) -> str:
    ref = make_ref("simexp", hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest())
    backend.store.put("experiments", payload, ref=ref)
    return ref


def test_full_integration_receipt_replays_raw_exact(loaded_backend) -> None:
    backend, ref = loaded_backend
    receipt = backend.run_experiment(ref.model_ref, controller={"hold": True}, duration_s=0.05)
    assert receipt.initial_state_ref
    assert backend.state_fidelity(receipt.initial_state_ref) == "FULL_INTEGRATION"
    report = backend.strict_replay(receipt.receipt_ref)
    assert report["verified"] is True
    assert report["mode"] == "RAW_EXACT"


def test_legacy_partial_caps_at_semantic(loaded_backend) -> None:
    """v1 状态作为 initial_state 的 receipt：即使逐状态一致，
    mode 也只能是 SEMANTIC，并标注 state_fidelity=LEGACY_PARTIAL。"""
    backend, ref = loaded_backend
    receipt = backend.run_experiment(ref.model_ref, controller={"hold": True}, duration_s=0.05)
    payload = backend.store.get(receipt.receipt_ref)
    # 把 initial_state_ref 换成 v1 快照（同物理内容，低保真）。
    v1_ref = backend.initial_state(ref.model_ref)
    forged = _forge(backend, {**payload, "initial_state_ref": v1_ref})
    report = backend.strict_replay(forged)
    assert report["verified"] is True
    assert report["mode"] == "SEMANTIC"
    assert report["state_fidelity"] == "LEGACY_PARTIAL"
