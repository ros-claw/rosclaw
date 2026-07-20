"""Local product-run persistence and path-safety tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.kernel import ActionState, EvidenceLevel, ExecutionMode, ExecutionReceipt
from rosclaw.product.runs import ProductRunStore, RunNotFoundError, RunStoreError


def _receipt(run_id: str = "run_test") -> ExecutionReceipt:
    return ExecutionReceipt(
        action_id=run_id,
        trace_id="trace_test",
        mode=ExecutionMode.SIMULATION,
        body_id="sim_ur5e",
        body_snapshot_hash="sha256:test",
        capability_id="sandbox.reach",
        final_state=ActionState.COMPLETED,
        evidence_level=EvidenceLevel.TASK_VERIFIED,
        verification_result={"success": True},
    )


def test_save_load_list_and_latest_are_consistent(tmp_path) -> None:
    store = ProductRunStore(tmp_path)

    receipt_path = store.save(_receipt())
    loaded, loaded_path = store.load()

    assert loaded_path == receipt_path
    assert loaded["action_id"] == "run_test"
    assert loaded["verified"] is True
    assert store.list()[0]["run_id"] == "run_test"
    pointer = json.loads((tmp_path / "runs" / "latest.json").read_text(encoding="utf-8"))
    assert pointer["receipt"] == "run_test/receipt.json"
    assert len(pointer["receipt_sha256"]) == 64


def test_latest_without_runs_has_actionable_error(tmp_path) -> None:
    with pytest.raises(RunNotFoundError, match="demo run ur5e-reach"):
        ProductRunStore(tmp_path).load()


@pytest.mark.parametrize(
    "reference",
    ["../receipt", "/tmp/receipt", "run/id", "", "a" * 161],
)
def test_run_reference_cannot_escape_store(tmp_path, reference: str) -> None:
    with pytest.raises(RunStoreError, match="Invalid run reference"):
        ProductRunStore(tmp_path).load(reference)


def test_tampered_receipt_identity_is_rejected(tmp_path) -> None:
    store = ProductRunStore(tmp_path)
    path = store.save(_receipt())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["action_id"] = "different_run"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RunStoreError, match="does not match"):
        store.load("run_test")


def test_tampered_summary_is_rejected_and_not_listed(tmp_path) -> None:
    store = ProductRunStore(tmp_path)
    store.save(_receipt())
    metadata_path = tmp_path / "runs" / "run_test" / "run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["verified"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RunStoreError, match="'verified' does not match"):
        store.load("run_test")
    assert store.list() == []


def test_symlinked_receipt_is_rejected(tmp_path) -> None:
    store = ProductRunStore(tmp_path)
    path = store.save(_receipt())
    external = tmp_path / "external.json"
    path.replace(external)
    path.symlink_to(external)

    with pytest.raises(RunStoreError, match="symbolic link"):
        store.load("run_test")


def test_symlinked_run_store_root_is_rejected(tmp_path: Path) -> None:
    external = tmp_path / "external-runs"
    external.mkdir()
    (tmp_path / "runs").symlink_to(external, target_is_directory=True)
    store = ProductRunStore(tmp_path)

    with pytest.raises(RunStoreError, match="store cannot be a symbolic link"):
        store.save(_receipt())
    with pytest.raises(RunStoreError, match="store cannot be a symbolic link"):
        store.load()
    with pytest.raises(RunStoreError, match="store cannot be a symbolic link"):
        store.list()


def test_existing_run_cannot_be_overwritten(tmp_path: Path) -> None:
    store = ProductRunStore(tmp_path)
    store.save(_receipt())

    with pytest.raises(RunStoreError, match="cannot be overwritten"):
        store.save(_receipt())


def test_tampered_latest_pointer_is_rejected(tmp_path) -> None:
    store = ProductRunStore(tmp_path)
    store.save(_receipt())
    pointer_path = tmp_path / "runs" / "latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["receipt"] = "../outside.json"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(RunStoreError, match="invalid receipt path"):
        store.load()
