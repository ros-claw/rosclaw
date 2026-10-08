"""Native portability and genuine mutation tests; only temporary copies change."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "replay_saved.py"
WORKSPACE = Path(__file__).resolve().parents[3]
BUNDLE = Path(
    os.environ.get("NATIVE_CAPABILITY_EVIDENCE", str(WORKSPACE / "protected_capability_evidence"))
)
spec = importlib.util.spec_from_file_location("native_saved_replay_tests", RUNNER_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
replay = cast(Callable[[Path], dict[str, Any]], runner.replay)


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_bytes()))


def _copy(tmp_path: Path) -> Path:
    target = tmp_path / "bundle"
    shutil.copytree(BUNDLE, target)
    for path in target.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    return target


def _rewrite(root: Path, relative: str, data: dict[str, Any]) -> None:
    """Re-index mutation, so negative tests reach semantic verification."""
    path = root / relative
    path.write_text(json.dumps(data), encoding="utf-8")
    manifest = _load(root / "manifest.json")
    for row in manifest["files"]:
        if row["path"] == relative:
            row["size"] = path.stat().st_size
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_valid_bundle_has_measured_18_cases_and_seven_sources() -> None:
    result = replay(BUNDLE)
    assert result["status"] == "PASS_REPLAYED_SAVED_CAPABILITY_EVIDENCE"
    requests = _load(BUNDLE / "moveit/requests.json")["cases"]
    pure = _load(BUNDLE / "vla/pure_cases.json")["cases"]
    evidence = _load(BUNDLE / "vla/genuine_native/reports/evidence.json")["cases"]
    ids = {row["id"] for row in requests + pure}
    ids.update(row["case_id"] for row in evidence)
    assert len(result["cases"]) == len(ids) == 18
    assert {row["case_id"] for row in result["cases"]} == ids
    assert all(row["passed"] is True for row in result["cases"])
    manifest = _load(BUNDLE / "manifest.json")
    assert result["source_hashes"] == {
        row["path"]: hashlib.sha256((BUNDLE / row["path"]).read_bytes()).hexdigest()
        for row in manifest["native_sources"]
    }
    expected_refs = {}
    for label in ("moveit", "vla"):
        for row in _load(BUNDLE / label / "registered_sources.json")["artifacts"]:
            expected_refs[row["artifact_id"]] = {
                "sha256": row["sha256"],
                "role": json.loads(row["metadata_json"])["role"],
            }
    assert result["artifact_refs"] == expected_refs


def test_identical_relocated_bundle(tmp_path: Path) -> None:
    assert replay(_copy(tmp_path)) == replay(BUNDLE)


def test_bad_file_hash_is_rejected(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    path = root / "moveit/actual_cases.json"
    data = path.read_bytes()
    path.write_bytes(data.replace(b"ACTUAL", b"BROKEN", 1))
    with pytest.raises(AssertionError, match="SAVED_FILE_HASH"):
        replay(root)


def test_reindexed_bad_numeric_shape_is_rejected(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    data = _load(root / "vla/model_results.json")
    data["model_results"][0]["actions"][0].pop()
    _rewrite(root, "vla/model_results.json", data)
    with pytest.raises((ValueError, AssertionError)):
        replay(root)


def test_reindexed_wrong_role_is_rejected(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    data = _load(root / "moveit/registered_sources.json")
    row = data["artifacts"][0]
    metadata = json.loads(row["metadata_json"])
    metadata["role"] = "unsupported_actual_role"
    row["metadata_json"] = json.dumps(metadata)
    _rewrite(root, "moveit/registered_sources.json", data)
    with pytest.raises(AssertionError, match="SAVED_REGISTERED_ROLE"):
        replay(root)


def test_reindexed_changed_case_input_is_rejected(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    data = _load(root / "moveit/actual_cases.json")
    data["cases"][0]["input"]["group"] = "unmatched_case_input"
    _rewrite(root, "moveit/actual_cases.json", data)
    with pytest.raises(AssertionError, match="EXACT_CASE_INPUT_REQUIRED"):
        replay(root)


def test_reindexed_changed_native_source_is_rejected(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    relative = "moveit/genuine_native/native/test_moveit_finite.cpp"
    path = root / relative
    path.write_bytes(path.read_bytes() + b"\n// modified native source\n")
    manifest = _load(root / "manifest.json")
    for row in manifest["files"]:
        if row["path"] == relative:
            row["size"] = path.stat().st_size
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AssertionError, match="GENUINE_NATIVE_SOURCE_HASH"):
        replay(root)


def test_reindexed_profile_numeric_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    data = _load(root / "vla/model_results.json")
    data["model_results"][1]["actions"][0][0][0] += 100
    _rewrite(root, "vla/model_results.json", data)
    with pytest.raises((ValueError, AssertionError)):
        replay(root)


def test_reindexed_feedback_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    data = _load(root / "vla/model_observations.json")
    data["preprocessor"][1]["observation.state"]["values"][0][0] += 1
    _rewrite(root, "vla/model_observations.json", data)
    with pytest.raises(AssertionError, match="ACTUAL_FIRST_ACTION_TO_SECOND_INPUT"):
        replay(root)


def test_same_path_changes_are_not_cached(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    assert replay(root)["status"] == "PASS_REPLAYED_SAVED_CAPABILITY_EVIDENCE"
    path = root / "moveit/actual_cases.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(AssertionError, match="SAVED_FILE_SIZE"):
        replay(root)


def test_matrix_scopes_are_exact_and_not_physical() -> None:
    matrix = _load(WORKSPACE / "validation/native_capabilities/evidence_matrix_57.json")
    original = _load(WORKSPACE / "inputs/original_matrix_57.json")
    authority = _load(WORKSPACE / "inputs/CURRENT_ROOT_51_SCOPE_AUTHORITY.json")
    assert {(r["id"], r["direction"]) for r in matrix["rows"]} == {
        (r["id"], r["direction"]) for r in original["rows"]
    }
    assert len(matrix["rows"]) == 57
    assert {r["id"] for r in matrix["rows"] if r["status"] == "VERIFIED_BOUNDED_SCOPE"} == set(
        authority["direction_ids"]
    )
    assert matrix["physical_qualified"] == authority["physical_qualified"] == 0
    assert all(r["limitations"] and r["dependencies"] for r in matrix["rows"])
