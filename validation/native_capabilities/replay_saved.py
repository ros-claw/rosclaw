"""Native saved-evidence replay; fixturelib validators are operator oracles.

No model, ROS, MoveIt runtime, dynamics or hardware is invoked. The pinned
manifest identifies one immutable evidence bundle, independently of location.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

MANIFEST_SHA256 = "034b1f8c6542af84925e9fb9b9c198869be1cf940755db167ce9030401904101"
SOURCE_HASHES = {
    "moveit/genuine_native/native/test_moveit_finite.cpp": (
        "0d247686f672fbb9f3cc6ae42ff25a4c59de529e10e8ae83444392d53653f93f"
    ),
    "moveit/genuine_native/native/moveit_fixture.hpp": (
        "636f188c85e818877131a722b07a8f784725b52f2e8962a995fb1dbd1cdaf40e"
    ),
    "moveit/genuine_native/docs/MOVEIT_FINITE_SCOPE.md": (
        "23df1d48289d3ce42ad43f5e92b99e1453d7060d0b4132f4aaa3d732bc3fa796"
    ),
    "moveit/genuine_native/reports/evidence.json": (
        "1ecb0725bb73caffaed97367b969073f331a0d81953825341048e8496cf8c7f1"
    ),
    "vla/genuine_native/native/test_vla_grounding_contention.py": (
        "d5bee6dc0f66ee55f574e76cd750f07fb4c68f4899c2a5d631f37e860133a145"
    ),
    "vla/genuine_native/docs/VLA_GROUNDING_CONTENTION_SCOPE.md": (
        "000a6ee84df89d908630a2afec02506f58a3c4a95b301a4372905c1e62dc111f"
    ),
    "vla/genuine_native/reports/evidence.json": (
        "c84e3606465e456cb7059b2c855e498c0a6944a5fe327869f4f4e060ff9362d2"
    ),
}
FIXTURE_HASHES = {
    "fixturelib/moveit_case_oracle.py": (
        "dd2b2a8e830f55f280a4b5cce63ef8558de8fb4f3267ef261ede611449248457"
    ),
    "fixturelib/vla_four_case_oracle.py": (
        "6a31b868f67cb186f00f1e22e8f9e0d9bac7424b1a5a3c7f241e52d2481049ef"
    ),
}
Record = dict[str, Any]
Oracle = Callable[..., list[Record]]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _object(data: bytes | str) -> Record:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return cast(Record, value)


def _read(root: Path, relative: str) -> bytes:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("BUNDLE_RELATIVE_PATH_REQUIRED")
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError("BUNDLE_PATH_ESCAPE")
    return target.read_bytes()


def _fixture(data: bytes, name: str) -> Record:
    # Only hash-verified, published operator fixture code is loaded. A fresh
    # namespace on every call avoids sys.path/module/bytecode-cache authority.
    namespace: Record = {"__name__": name}
    exec(compile(data, name, "exec"), namespace)
    return namespace


def _registrations(
    payload: dict[str, bytes], sources: list[Record], hashes: dict[str, str]
) -> dict[str, Record]:
    artifacts: dict[str, Record] = {}
    for label in ("moveit", "vla"):
        saved = _object(payload[f"{label}/registered_sources.json"])
        assert len(saved["task"]) == 1, "ONE_SAVED_TASK_REQUIRED"
        task_id = saved["task"][0]["task_id"]
        rows = saved["artifacts"]
        for row in rows:
            name = Path(row["path"]).name
            if name == "delivery.json":
                declared = "progress_report"
            elif name.endswith((".cpp", ".hpp", ".py")):
                declared = "diagnostic_source"
            else:
                declared = "diagnostic_report"
            role = _object(row["metadata_json"])["role"]
            assert role == declared, "SAVED_REGISTERED_ROLE"
            assert row["task_id"] == task_id, "SAVED_REGISTERED_TASK_BINDING"
            assert row["producer"] == "model:rosclaw_artifact_register"
            ref = row["artifact_id"]
            assert isinstance(ref, str) and ref.startswith("art_")
            assert ref not in artifacts, "DUPLICATE_SAVED_REF"
            digest = row["sha256"]
            assert isinstance(digest, str) and len(digest) == 64
            artifacts[ref] = {"sha256": digest, "role": role}
        for source in sources:
            path = source["path"]
            if not path.startswith(label + "/"):
                continue
            matches = [row for row in rows if row["path"] == source["registered_original_path"]]
            assert len(matches) == 1, "SAVED_REGISTERED_SOURCE_BINDING"
            assert matches[0]["sha256"] == hashes[path], "REGISTERED_SOURCE_HASH"
            assert matches[0]["size_bytes"] == len(payload[path])
    return artifacts


def _replay(root: Path) -> Record:
    manifest_bytes = _read(root, "manifest.json")
    manifest = _object(manifest_bytes)
    assert manifest["schema"] == "Native.CapabilityEvidence.Bundle.v1"
    payload: dict[str, bytes] = {}
    for entry in manifest["files"]:
        path = entry["path"]
        assert path not in payload, "DUPLICATE_BUNDLE_PATH"
        data = _read(root, path)
        assert type(entry["size"]) is int and len(data) == entry["size"], "SAVED_FILE_SIZE"
        assert _sha(data) == entry["sha256"], "SAVED_FILE_HASH"
        payload[path] = data
    sources: list[Record] = manifest["native_sources"]
    assert len(sources) == 7
    assert {row["path"] for row in sources} == set(SOURCE_HASHES)
    hashes = {row["path"]: _sha(payload[row["path"]]) for row in sources}
    assert hashes == SOURCE_HASHES, "GENUINE_NATIVE_SOURCE_HASH"
    assert all(row["sha256"] == hashes[row["path"]] for row in sources)
    assert all(row["source_class"] == "GENUINE_COMPLETED_NATIVE_SOURCE" for row in sources)
    artifacts = _registrations(payload, sources, hashes)
    for path, digest in FIXTURE_HASHES.items():
        assert _sha(payload[path]) == digest, "OPERATOR_FIXTURE_HASH"
    moveit = _fixture(payload["fixturelib/moveit_case_oracle.py"], "operator_moveit")
    vla = _fixture(payload["fixturelib/vla_four_case_oracle.py"], "operator_vla")

    def load(path: str) -> Record:
        return _object(payload[path])

    requests = load("moveit/requests.json")["cases"]
    moveit_verdicts = cast(Oracle, moveit["validate"])(requests, load("moveit/actual_cases.json"))
    pure = load("vla/pure_actual.json")
    pure_cases = load("vla/pure_cases.json")["cases"]
    pure_verdicts = cast(Oracle, vla["validate_pure"])(
        pure_cases, pure["native_result"]["cases"], pure["actual_solve_calls"]
    )
    model_verdicts = cast(Oracle, vla["validate_models"])(
        load("vla/model_results.json")["model_results"],
        load("vla/model_observations.json"),
        load("vla/requests.json")["requests"],
        load("vla/profile.json")["stats"],
    )
    cases: list[Record] = []
    for verdicts, key, status, count in (
        (
            moveit_verdicts,
            "case",
            "PASS_ACTUAL_GENERATED_CPP_LIBRARY_AND_INDEPENDENT_ORACLE",
            10,
        ),
        (
            pure_verdicts,
            "id",
            "PASS_FINITE_EXISTING_NATIVE_STRUCTURED_SOFTWARE",
            4,
        ),
        (
            model_verdicts,
            "id",
            "PASS_ACTUAL_PRETRAINED_MODEL_COORDINATE_INTERFACE_NOT_PHYSICAL",
            4,
        ),
    ):
        assert len(verdicts) == count
        for verdict in verdicts:
            assert verdict["status"] == status, "RECOMPUTED_ORACLE_FAILURE"
            cases.append({"case_id": verdict[key], "passed": True})
    assert len({row["case_id"] for row in cases}) == 18
    # Semantic checks above still run on re-indexed corrupt controls. This final
    # trust anchor also rejects coordinated replacement of inputs and answers.
    assert _sha(manifest_bytes) == MANIFEST_SHA256, "SAVED_MANIFEST_TRUST_ANCHOR"
    return {
        "status": "PASS_REPLAYED_SAVED_CAPABILITY_EVIDENCE",
        "cases": cases,
        "source_hashes": hashes,
        "artifact_refs": artifacts,
    }


def replay(evidence_root: str | Path) -> Record:
    """Verify and recompute saved MoveIt10/VLA8 evidence; reject invalid bundles.

    artifact_refs are checked *historical bundle registrations*, never new
    registrations in the replaying task. No path-based memoization is used.
    """
    try:
        return _replay(Path(evidence_root).resolve())
    except (OSError, KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError("INVALID_SAVED_BUNDLE") from exc
