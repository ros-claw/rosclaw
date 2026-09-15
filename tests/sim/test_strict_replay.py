"""Strict replay 测试（PR-MH5，规格 §56/§57，红→绿）。

replay 不一致 → REPLAY_DIVERGED，不得 promotion。
"""

from __future__ import annotations

import hashlib

import pytest

from rosclaw.contracts.common import canonical_json
from rosclaw.sim.refs import make_ref


def _forge(backend, payload: dict) -> str:
    ref = make_ref("simexp", hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest())
    backend.store.put("experiments", payload, ref=ref)
    return ref


def test_strict_replay_verified(loaded_backend) -> None:
    backend, ref = loaded_backend
    receipt = backend.run_experiment(ref.model_ref, controller={"hold": True}, duration_s=0.05)
    report = backend.strict_replay(receipt.receipt_ref)
    assert report["verified"] is True
    assert report["mode"] == "raw"
    assert report["receipt_ref"] == receipt.receipt_ref


def test_replay_diverged_on_wrong_backend_version(loaded_backend) -> None:
    backend, ref = loaded_backend
    receipt = backend.run_experiment(ref.model_ref, controller={"hold": True}, duration_s=0.05)
    payload = backend.store.get(receipt.receipt_ref)
    forged = _forge(backend, {**payload, "backend_version": "0.0.0"})
    with pytest.raises(ValueError, match="REPLAY_DIVERGED"):
        backend.strict_replay(forged)


def test_replay_diverged_on_wrong_raw_digest(loaded_backend) -> None:
    backend, ref = loaded_backend
    receipt = backend.run_experiment(ref.model_ref, controller={"hold": True}, duration_s=0.05)
    payload = backend.store.get(receipt.receipt_ref)
    # raw digest 错 + metrics 也错 → 不可信。
    forged = _forge(
        backend,
        {
            **payload,
            "states_digest": "sha256:" + "0" * 64,
            "metrics": {**payload["metrics"], "tracking_rmse": 999.0},
        },
    )
    with pytest.raises(ValueError, match="REPLAY_DIVERGED"):
        backend.strict_replay(forged)


def test_replay_semantic_match_within_tolerance(loaded_backend) -> None:
    """raw digest 不同（浮点抖动）但语义指标一致 → 仍可验证（semantic 层）。"""
    backend, ref = loaded_backend
    receipt = backend.run_experiment(ref.model_ref, controller={"hold": True}, duration_s=0.05)
    payload = backend.store.get(receipt.receipt_ref)
    forged = _forge(backend, {**payload, "states_digest": "sha256:" + "f" * 64})
    report = backend.strict_replay(forged)
    assert report["verified"] is True
    assert report["mode"] == "semantic"


def test_replay_missing_receipt(loaded_backend) -> None:
    backend, _ = loaded_backend
    ghost = make_ref("simexp", hashlib.sha256(b"ghost").hexdigest())
    with pytest.raises(ValueError, match="REF_NOT_FOUND"):
        backend.strict_replay(ghost)
