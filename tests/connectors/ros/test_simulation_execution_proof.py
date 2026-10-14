"""Synthetic SDK proof contracts; no motion or physical acceptance evidence."""

import dataclasses
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("simulation_execution_proof")


@pytest.fixture
def contract(module):
    pins = module.ExecutionProofPins(
        body_id="synthetic-j100",
        body_snapshot_hash="a" * 64,
        manifest_hash="navbodyworkspace_" + "b" * 64,
        run_id="synthetic-world-1",
        world_sha256="c" * 64,
        observer_epoch="d" * 32,
        daemon_instance_id="daemon_synthetic_1",
        action_id="action_synthetic_1",
    )
    now = 10_000_000_000
    payload = {
        **dataclasses.asdict(pins),
        "schema_version": "rosclaw.sim_execution_supervision.v1",
        "sequence": 1,
        "source_sample": 25,
        "health_sha256": "e" * 64,
        "source_review_ns": now - 100_000_000,
        "health_review_ns": now - 50_000_000,
        "issued_ns": now,
        "expires_ns": now + 250_000_000,
    }
    key = bytes(range(32))
    assert len(key) == 32
    return pins, payload, key, now


def signed(module, contract, **changes):
    pins, payload, key, now = contract
    payload = {**payload, **changes}
    return module.sign_execution_proof(payload, pins=pins, key=key, now_ns=now)


def test_valid_same_proof_can_be_reread_only_until_deadline(module, contract):
    pins, payload, key, now = contract
    verifier = module.ExecutionProofVerifier(pins=pins, key=key)
    raw = signed(module, contract)
    assert verifier.verify(raw, now_ns=now) == payload
    assert verifier.verify(raw, now_ns=now + 249_999_999) == payload
    with pytest.raises(ValueError, match="expired"):
        verifier.verify(raw, now_ns=now + 250_000_000)
    with pytest.raises(ValueError, match="latched"):
        verifier.verify(raw, now_ns=now)


@pytest.mark.parametrize(
    "field",
    [
        "body_id",
        "body_snapshot_hash",
        "manifest_hash",
        "run_id",
        "world_sha256",
        "observer_epoch",
        "daemon_instance_id",
        "action_id",
    ],
)
def test_other_identity_is_rejected_even_with_valid_signature(module, contract, field):
    pins, payload, key, now = contract
    # A different admitted producer with the same fixture key cannot reuse this sink.
    value = (
        ("navbodyworkspace_" + "f" * 64) if field == "manifest_hash" else "f" * len(payload[field])
    )
    other = dataclasses.replace(pins, **{field: value})
    raw = module.sign_execution_proof({**payload, field: value}, pins=other, key=key, now_ns=now)
    verifier = module.ExecutionProofVerifier(pins=pins, key=key)
    with pytest.raises(ValueError, match="identity changed"):
        verifier.verify(raw, now_ns=now)


@pytest.mark.parametrize(
    "changes",
    [
        {"sequence": True},
        {"source_sample": -1},
        {"issued_ns": float("nan")},
        {"expires_ns": 10_250_000_001},
        {"health_review_ns": 9_499_999_999},
        {"source_review_ns": 10_000_000_001},
        {"health_sha256": "READY"},
        {"approved": True},
        {"schema_version": "rosclaw.sim_execution_supervision.v2"},
    ],
)
def test_signer_refuses_unbounded_stale_or_caller_approval_fields(module, contract, changes):
    with pytest.raises(ValueError):
        signed(module, contract, **changes)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"x" * 8193,
        b"[]",
        b"{",
        b"\xff",
        b'{"payload":{},"payload":{},"hmac_sha256":"x"}',
    ],
    ids=["empty", "oversized", "array", "partial", "encoding", "duplicate"],
)
def test_malformed_input_latches_and_cannot_recover(module, contract, raw):
    pins, _, key, now = contract
    verifier = module.ExecutionProofVerifier(pins=pins, key=key)
    with pytest.raises(ValueError):
        verifier.verify(raw, now_ns=now)
    with pytest.raises(ValueError, match="latched"):
        verifier.verify(signed(module, contract), now_ns=now)


def test_tampering_and_wrong_key_are_terminal(module, contract):
    pins, _, key, now = contract
    envelope = json.loads(signed(module, contract))
    envelope["payload"]["source_sample"] += 1
    for raw, verifier_key in (
        (json.dumps(envelope).encode(), key),
        (signed(module, contract), b"z" * 32),
    ):
        with pytest.raises(ValueError, match="authentication"):
            module.ExecutionProofVerifier(pins=pins, key=verifier_key).verify(raw, now_ns=now)


@pytest.mark.parametrize(
    "change",
    [
        {"sequence": 1},
        {"source_sample": 24},
        {"sequence": 2, "health_sha256": "f" * 64},
    ],
)
def test_sequence_rollback_cursor_rollback_or_equivocation_latches(module, contract, change):
    pins, _, key, now = contract
    verifier = module.ExecutionProofVerifier(pins=pins, key=key)
    verifier.verify(signed(module, contract, sequence=2), now_ns=now)
    with pytest.raises(ValueError, match="rolled back"):
        verifier.verify(signed(module, contract, **change), now_ns=now + 1)


def test_advancing_proof_cannot_hide_old_observations(module, contract):
    pins, payload, key, now = contract
    raw = module.sign_execution_proof(
        {**payload, "source_review_ns": now - 500_000_000},
        pins=pins,
        key=key,
        now_ns=now,
    )
    with pytest.raises(ValueError, match="observation expired"):
        module.ExecutionProofVerifier(pins=pins, key=key).verify(raw, now_ns=now + 1)


def test_clock_rollback_latches_even_with_still_valid_signed_proof(module, contract):
    pins, _, key, now = contract
    verifier = module.ExecutionProofVerifier(pins=pins, key=key)
    raw = signed(module, contract)
    verifier.verify(raw, now_ns=now + 10)
    with pytest.raises(ValueError, match="rolled back"):
        verifier.verify(raw, now_ns=now + 9)
