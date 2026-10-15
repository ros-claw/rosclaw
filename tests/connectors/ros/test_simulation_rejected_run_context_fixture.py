"""Synthetic SDK wrong-run signer contracts; no real keys or motion."""

import hashlib
import hmac
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_rejected_run_context_fixture")
    guard = object.__new__(module.RejectedRunContextSupervision)
    guard.binding = SimpleNamespace(run_id="original-run")
    guard._key = bytes(range(32))
    guard._wrong_run_injected = False
    guard._operator_fault_sha256 = "a" * 64
    records, writes = [], []
    guard._record = records.append
    monkeypatch.setattr(
        module.SdkNavigationExecutionSupervision,
        "_write_private",
        lambda self, name, raw: writes.append((name, raw)),
    )
    return module, guard, records, writes


def test_wrong_run_remains_authenticated_but_changes_no_other_identity(fixture):
    module, guard, records, writes = fixture
    original = {
        "run_id": "original-run",
        "body_snapshot_hash": "actual-body",
        "world_sha256": "actual-world",
        "daemon_instance_id": "actual-daemon",
    }
    raw = module._canonical({"payload": original, "hmac_sha256": "synthetic-original"})
    guard._write_private("admission.json", raw)
    envelope = json.loads(writes[0][1])
    assert envelope["payload"] == {**original, "run_id": "original-run-wrong-run"}
    assert hmac.compare_digest(
        envelope["hmac_sha256"],
        hmac.new(guard._key, module._canonical(envelope["payload"]), hashlib.sha256).hexdigest(),
    )
    assert records[0]["actual_rejected_context_sha256"] == hashlib.sha256(writes[0][1]).hexdigest()
    with pytest.raises(ValueError, match="never retry"):
        guard._write_private("admission.json", raw)
    assert len(writes) == 1


def test_original_wrong_run_is_not_silently_replaced(fixture):
    module, guard, _, writes = fixture
    with pytest.raises(ValueError, match="actual run before injection"):
        guard._write_private("admission.json", module._canonical({"payload": {"run_id": "other"}}))
    assert not writes


def test_proof_bytes_are_never_mutated_by_operator_context_fault(fixture):
    _, guard, records, writes = fixture
    guard._write_private("proof.json", b"synthetic-original-proof")
    assert writes == [("proof.json", b"synthetic-original-proof")] and not records


@pytest.mark.parametrize("fault", ["case", "phase", "sha", "body"])
def test_wrong_operator_scope_fails_before_base_supervision_creation(fixture, monkeypatch, fault):
    module, _, _, _ = fixture
    pins = {
        "run_id": "original-run",
        "body_snapshot_hash": "a" * 64,
        "world_sha256": "b" * 64,
        "action_id": "first",
    }
    policy = {
        "schema_version": "rosclaw.governed_sim_fault.v1",
        "case": "wrong_run_id",
        "phase": "before_dispatch",
        **pins,
    }
    if fault == "case":
        policy["case"] = "out_of_bounds_goal"
    if fault == "phase":
        policy["phase"] = "during_motion"
    if fault == "body":
        policy["body_snapshot_hash"] = "c" * 64
    raw = json.dumps(policy).encode()
    digest = hashlib.sha256(raw).hexdigest()
    if fault == "sha":
        digest = "0" * 64
    monkeypatch.setattr(
        module.SdkNavigationExecutionSupervision,
        "__init__",
        lambda self, **kwargs: pytest.fail("base guard must not be created"),
    )
    with pytest.raises(ValueError):
        module.RejectedRunContextSupervision(
            operator_fault_raw=raw,
            operator_fault_sha256=digest,
            binding=SimpleNamespace(
                run_id=pins["run_id"], body_snapshot_hash=pins["body_snapshot_hash"]
            ),
            facts=SimpleNamespace(enrollment={"world_sha256": pins["world_sha256"]}),
            allowed_action_ids=("first",),
        )


@pytest.mark.parametrize("kind", ["coverage", "navigation_only", "passive_observation"])
def test_wrong_run_cli_fault_cannot_start_unrelated_runtime(fixture, monkeypatch, tmp_path, kind):
    module = importlib.import_module("daemon")
    (tmp_path / "execution_config.json").write_text(json.dumps({"executor_kind": kind}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "daemon.py",
            "--directory",
            str(tmp_path),
            "--operator-wrong-run-context-sha256",
            "a" * 64,
        ],
    )
    monkeypatch.setattr(
        module,
        "Runtime",
        lambda *a, **k: pytest.fail("Runtime must not start for invalid fault role"),
    )
    with pytest.raises(ValueError, match="explicit isolated governed SIM"):
        module.main()
