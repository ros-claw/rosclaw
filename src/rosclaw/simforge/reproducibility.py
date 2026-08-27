"""Reusable process and content closure for simulation evidence.

The closure deliberately contains no simulator- or task-specific scoring.  A
downstream task binds its source trees, runtime dependencies and artifacts,
then asks fresh workers to return task-specific exact replay fields.  The
generic evaluator recomputes the process, identity, authority and equality
gates instead of trusting a pre-filled ``passed`` summary.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_THREAD_ENVIRONMENT_KEYS = (
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ORT_NUM_THREADS",
)
_CHUNK_BYTES = 1024 * 1024


def canonical_json_hash(value: Any) -> str:
    """Return a finite, type-preserving canonical JSON SHA-256 commitment."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("reproducibility value must be finite canonical JSON") from error
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a regular file without loading a large artifact into memory."""

    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError("reproducibility artifact must be a regular non-symlink file")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ValueError("reproducibility artifact must be a regular non-symlink file")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class SourceTreeBinding:
    """Path-independent commitment to a selected source tree."""

    label: str
    digest: str
    file_count: int
    suffixes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_label(self.label, "source tree")
        _require_sha256(self.digest, "source tree digest")
        if not isinstance(self.file_count, int) or isinstance(self.file_count, bool):
            raise ValueError("source tree file_count must be an integer")
        if self.file_count < 1:
            raise ValueError("source tree must bind at least one file")
        if (
            not self.suffixes
            or len(set(self.suffixes)) != len(self.suffixes)
            or any(not _valid_suffix(value) for value in self.suffixes)
        ):
            raise ValueError("source tree suffixes are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "digest": self.digest,
            "file_count": self.file_count,
            "suffixes": list(self.suffixes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceTreeBinding:
        suffixes = value.get("suffixes")
        if not isinstance(suffixes, list):
            raise ValueError("source tree suffixes must be a list")
        return cls(
            label=_strict_string(value.get("label"), "source tree label"),
            digest=_strict_string(value.get("digest"), "source tree digest"),
            file_count=_strict_int(value.get("file_count"), "source tree file_count"),
            suffixes=tuple(_strict_string(item, "source tree suffix") for item in suffixes),
        )


def bind_source_tree(
    label: str,
    root: Path,
    *,
    suffixes: tuple[str, ...] = (".py",),
) -> SourceTreeBinding:
    """Bind all matching regular files under ``root`` by relative path."""

    _require_label(label, "source tree")
    normalized_suffixes = tuple(sorted(suffixes))
    if (
        not normalized_suffixes
        or len(set(normalized_suffixes)) != len(normalized_suffixes)
        or any(not _valid_suffix(value) for value in normalized_suffixes)
    ):
        raise ValueError("source tree suffixes are invalid")
    expanded = root.expanduser()
    if expanded.is_symlink():
        raise ValueError("source tree root cannot be a symlink")
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise ValueError("source tree root must be a directory")
    files = sorted(
        path
        for path in resolved.rglob("*")
        if path.is_file() and path.suffix in normalized_suffixes
    )
    if not files:
        raise ValueError("source tree has no selected files")
    entries: dict[str, str] = {}
    for path in files:
        if path.is_symlink():
            raise ValueError("source tree cannot contain selected symlink files")
        relative = path.relative_to(resolved).as_posix()
        entries[relative] = file_sha256(path)
    return SourceTreeBinding(
        label=label,
        digest=canonical_json_hash(entries),
        file_count=len(entries),
        suffixes=normalized_suffixes,
    )


@dataclass(frozen=True)
class ArtifactBinding:
    """Path-independent file commitment for a model, policy or evidence input."""

    label: str
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        _require_label(self.label, "artifact")
        _require_sha256(self.digest, "artifact digest")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError("artifact size_bytes must be a non-negative integer")

    @classmethod
    def from_path(cls, label: str, path: Path) -> ArtifactBinding:
        _require_label(label, "artifact")
        resolved = path.expanduser().resolve()
        digest = file_sha256(resolved)
        return cls(label=label, digest=digest, size_bytes=resolved.stat().st_size)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "digest": self.digest, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactBinding:
        return cls(
            label=_strict_string(value.get("label"), "artifact label"),
            digest=_strict_string(value.get("digest"), "artifact digest"),
            size_bytes=_strict_int(value.get("size_bytes"), "artifact size_bytes"),
        )


@dataclass(frozen=True)
class RuntimeProcessContract:
    """Stable process attributes that can influence numerical replay."""

    python_version: str
    python_implementation: str
    platform: str
    machine: str
    libc: tuple[str, str]
    cpu_count: int
    hash_randomization: int
    thread_environment: tuple[tuple[str, str | None], ...]

    def __post_init__(self) -> None:
        text = (
            self.python_version,
            self.python_implementation,
            self.platform,
            self.machine,
        )
        if any(not isinstance(value, str) or not value for value in text):
            raise ValueError("runtime process text fields must be non-empty")
        if len(self.libc) != 2 or any(not isinstance(value, str) for value in self.libc):
            raise ValueError("runtime libc contract is invalid")
        if (
            not isinstance(self.cpu_count, int)
            or isinstance(self.cpu_count, bool)
            or self.cpu_count < 1
        ):
            raise ValueError("runtime cpu_count must be positive")
        if self.hash_randomization not in (0, 1):
            raise ValueError("runtime hash_randomization must be zero or one")
        if tuple(name for name, _ in self.thread_environment) != _THREAD_ENVIRONMENT_KEYS:
            raise ValueError("runtime thread environment contract is incomplete")
        if any(
            value is not None and not isinstance(value, str) for _, value in self.thread_environment
        ):
            raise ValueError("runtime thread environment values must be strings or null")

    @classmethod
    def capture(cls) -> RuntimeProcessContract:
        libc_name, libc_version = platform.libc_ver()
        return cls(
            python_version=platform.python_version(),
            python_implementation=platform.python_implementation(),
            platform=platform.platform(),
            machine=platform.machine(),
            libc=(libc_name, libc_version),
            cpu_count=os.cpu_count() or 1,
            hash_randomization=int(sys.flags.hash_randomization),
            thread_environment=tuple(
                (name, os.environ.get(name)) for name in _THREAD_ENVIRONMENT_KEYS
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimeProcessContract:
        try:
            libc = value["libc"]
            environment = value["thread_environment"]
            if (
                not isinstance(libc, list)
                or not isinstance(environment, dict)
                or set(environment) != set(_THREAD_ENVIRONMENT_KEYS)
            ):
                raise ValueError
            return cls(
                python_version=_strict_string(value["python_version"], "python_version"),
                python_implementation=_strict_string(
                    value["python_implementation"], "python_implementation"
                ),
                platform=_strict_string(value["platform"], "platform"),
                machine=_strict_string(value["machine"], "machine"),
                libc=(
                    _strict_string(libc[0], "libc name", allow_empty=True),
                    _strict_string(libc[1], "libc version", allow_empty=True),
                ),
                cpu_count=_strict_int(value["cpu_count"], "runtime cpu_count"),
                hash_randomization=_strict_int(
                    value["hash_randomization"], "runtime hash_randomization"
                ),
                thread_environment=tuple(
                    (name, environment.get(name)) for name in _THREAD_ENVIRONMENT_KEYS
                ),
            )
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ValueError("runtime process contract mapping is invalid") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform": self.platform,
            "machine": self.machine,
            "libc": list(self.libc),
            "cpu_count": self.cpu_count,
            "hash_randomization": self.hash_randomization,
            "thread_environment": dict(self.thread_environment),
        }


@dataclass(frozen=True)
class ReproducibilityClosure:
    """Portable commitment that every fresh simulation worker must bind."""

    source_trees: tuple[SourceTreeBinding, ...]
    dependencies: tuple[tuple[str, str], ...]
    artifacts: tuple[ArtifactBinding, ...]
    process_contract: RuntimeProcessContract
    expected_replays: int = 3
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.simforge.reproducibility_closure.v1"

    def __post_init__(self) -> None:
        _require_unique_labels((item.label for item in self.source_trees), "source tree")
        _require_unique_labels(
            (item.label for item in self.artifacts), "artifact", allow_empty=True
        )
        dependency_names = tuple(name for name, _ in self.dependencies)
        _require_unique_labels(dependency_names, "dependency", allow_empty=True)
        if tuple(sorted(self.source_trees, key=lambda item: item.label)) != self.source_trees:
            raise ValueError("source tree bindings must be label-sorted")
        if tuple(sorted(self.artifacts, key=lambda item: item.label)) != self.artifacts:
            raise ValueError("artifact bindings must be label-sorted")
        if tuple(sorted(self.dependencies)) != self.dependencies:
            raise ValueError("dependency bindings must be name-sorted")
        if any(not isinstance(version, str) or not version for _, version in self.dependencies):
            raise ValueError("dependency versions must be non-empty strings")
        if (
            not isinstance(self.expected_replays, int)
            or isinstance(self.expected_replays, bool)
            or not 2 <= self.expected_replays <= 32
        ):
            raise ValueError("expected_replays must be in [2, 32]")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("reproducibility closure cannot authorize hardware")
        if self.schema_version != "rosclaw.simforge.reproducibility_closure.v1":
            raise ValueError("unsupported reproducibility closure schema")

    @property
    def closure_hash(self) -> str:
        return canonical_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_trees": [item.to_dict() for item in self.source_trees],
            "dependencies": dict(self.dependencies),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "process_contract": self.process_contract.to_dict(),
            "expected_replays": self.expected_replays,
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": self.hardware_authorized,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReproducibilityClosure:
        trees = value.get("source_trees")
        dependencies = value.get("dependencies")
        artifacts = value.get("artifacts")
        process_contract = value.get("process_contract")
        if (
            not isinstance(trees, list)
            or not isinstance(dependencies, dict)
            or not isinstance(artifacts, list)
            or not isinstance(process_contract, dict)
            or any(not isinstance(item, dict) for item in trees)
            or any(not isinstance(item, dict) for item in artifacts)
        ):
            raise ValueError("reproducibility closure mapping is invalid")
        return cls(
            source_trees=tuple(SourceTreeBinding.from_dict(item) for item in trees),
            dependencies=tuple(
                (
                    _strict_string(name, "dependency name"),
                    _strict_string(version, "dependency version"),
                )
                for name, version in dependencies.items()
            ),
            artifacts=tuple(ArtifactBinding.from_dict(item) for item in artifacts),
            process_contract=RuntimeProcessContract.from_dict(process_contract),
            expected_replays=_strict_int(value.get("expected_replays"), "expected_replays"),
            activation_ceiling=_strict_string(
                value.get("activation_ceiling"), "activation_ceiling"
            ),
            hardware_authorized=_strict_bool(
                value.get("hardware_authorized"), "hardware_authorized"
            ),
            schema_version=_strict_string(value.get("schema_version"), "schema_version"),
        )


def build_reproducibility_closure(
    *,
    source_trees: Mapping[str, Path],
    dependency_packages: Iterable[str] = (),
    artifacts: Mapping[str, Path] | None = None,
    expected_replays: int = 3,
) -> ReproducibilityClosure:
    """Capture a deterministic, path-independent closure for a parent run."""

    trees = tuple(
        bind_source_tree(label, root)
        for label, root in sorted(source_trees.items(), key=lambda item: item[0])
    )
    dependencies = tuple(
        (name, importlib.metadata.version(name)) for name in sorted(set(dependency_packages))
    )
    artifact_values = artifacts or {}
    artifact_bindings = tuple(
        ArtifactBinding.from_path(label, path)
        for label, path in sorted(artifact_values.items(), key=lambda item: item[0])
    )
    return ReproducibilityClosure(
        source_trees=trees,
        dependencies=dependencies,
        artifacts=artifact_bindings,
        process_contract=RuntimeProcessContract.capture(),
        expected_replays=expected_replays,
    )


@dataclass(frozen=True)
class CrossProcessReplayVerdict:
    """Derived replay gates; this object never grants hardware authority."""

    passed: bool
    gates: tuple[tuple[str, bool], ...]
    replay_count: int
    process_ids: tuple[int, ...]
    exact_fields: tuple[str, ...]
    reference_commitment: str
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.simforge.cross_process_replay_verdict.v1"

    def __post_init__(self) -> None:
        derived = bool(self.gates and all(value for _, value in self.gates))
        if len({name for name, _ in self.gates}) != len(self.gates):
            raise ValueError("cross-process replay gate names must be unique")
        if self.passed is not derived:
            raise ValueError("cross-process replay verdict does not match its gates")
        if len(set(self.process_ids)) != len(self.process_ids):
            raise ValueError("cross-process replay verdict process IDs must be unique")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("cross-process replay verdict cannot authorize hardware")
        if self.schema_version != "rosclaw.simforge.cross_process_replay_verdict.v1":
            raise ValueError("unsupported cross-process replay verdict schema")
        _require_sha256(self.reference_commitment, "reference commitment")

    @property
    def verdict_hash(self) -> str:
        return canonical_json_hash(self.to_dict())

    def require_passed(self) -> None:
        if not self.passed:
            failed = ", ".join(name for name, value in self.gates if not value)
            raise ValueError(f"cross-process reproducibility failed closed: {failed}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "gates": dict(self.gates),
            "replay_count": self.replay_count,
            "process_ids": list(self.process_ids),
            "exact_fields": list(self.exact_fields),
            "reference_commitment": self.reference_commitment,
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": self.hardware_authorized,
        }


def evaluate_cross_process_replays(
    closure: ReproducibilityClosure,
    workers: Sequence[Mapping[str, Any]],
    *,
    exact_fields: tuple[str, ...],
    launcher_process_id: int | None = None,
) -> CrossProcessReplayVerdict:
    """Recompute fresh-process, exact-replay and SIM-only worker gates."""

    if not exact_fields or len(set(exact_fields)) != len(exact_fields):
        raise ValueError("exact replay fields must be non-empty and unique")
    for field in exact_fields:
        _require_label(field, "exact replay field")
    if launcher_process_id is not None and (
        not isinstance(launcher_process_id, int)
        or isinstance(launcher_process_id, bool)
        or launcher_process_id < 1
    ):
        raise ValueError("launcher_process_id must be a positive integer")
    if not workers:
        raise ValueError("cross-process replay set cannot be empty")

    process_ids: list[int] = []
    exact_values: list[dict[str, Any]] = []
    contract = closure.process_contract.to_dict()
    worker_contracts_match = True
    closure_hashes_match = True
    outcomes_passed = True
    sim_only_safe = True
    for worker in workers:
        if not isinstance(worker, Mapping):
            raise ValueError("worker report must be a mapping")
        process_id = worker.get("process_id")
        if not isinstance(process_id, int) or isinstance(process_id, bool) or process_id < 1:
            raise ValueError("worker process_id must be a positive integer")
        process_ids.append(process_id)
        missing = [field for field in exact_fields if field not in worker]
        if missing:
            raise ValueError(f"worker is missing exact replay fields: {', '.join(missing)}")
        exact_values.append({field: worker[field] for field in exact_fields})
        worker_contracts_match &= worker.get("process_contract") == contract
        closure_hashes_match &= worker.get("closure_hash") == closure.closure_hash
        outcomes_passed &= worker.get("passed") is True
        sim_only_safe &= (
            worker.get("activation_ceiling") == "SIM_ONLY"
            and worker.get("hardware_authorized") is False
            and worker.get("hardware_command_sent") is False
        )

    reference = canonical_json_hash(exact_values[0])
    exact_replay = all(canonical_json_hash(value) == reference for value in exact_values)
    unique_process_ids = len(set(process_ids)) == len(process_ids)
    distinct_from_launcher = launcher_process_id is None or launcher_process_id not in process_ids
    gates = (
        ("expected_worker_count", len(workers) == closure.expected_replays),
        ("fresh_process_identity", unique_process_ids and distinct_from_launcher),
        ("process_contract_identical", worker_contracts_match),
        ("closure_bound", closure_hashes_match),
        ("cross_process_exact_replay", exact_replay),
        ("worker_outcomes_passed", outcomes_passed),
        ("all_workers_sim_only_safe", sim_only_safe),
    )
    return CrossProcessReplayVerdict(
        passed=all(value for _, value in gates),
        gates=gates,
        replay_count=len(workers),
        process_ids=tuple(sorted(set(process_ids))),
        exact_fields=exact_fields,
        reference_commitment=reference,
    )


def _require_unique_labels(values: Iterable[str], kind: str, *, allow_empty: bool = False) -> None:
    labels = tuple(values)
    if (not labels and not allow_empty) or len(labels) != len(set(labels)):
        raise ValueError(f"{kind} labels must be non-empty and unique")
    for label in labels:
        _require_label(label, kind)


def _require_label(value: str, kind: str) -> None:
    if not isinstance(value, str) or _LABEL.fullmatch(value) is None:
        raise ValueError(f"{kind} label is invalid")


def _require_sha256(value: str, kind: str) -> None:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{kind} must be a lowercase sha256 digest")


def _valid_suffix(value: str) -> bool:
    return (
        isinstance(value, str)
        and 2 <= len(value) <= 16
        and value.startswith(".")
        and value[1:].isalnum()
    )


def _strict_int(value: Any, kind: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{kind} must be an integer")
    return value


def _strict_string(value: Any, kind: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not allow_empty):
        raise ValueError(f"{kind} must be a string")
    return value


def _strict_bool(value: Any, kind: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{kind} must be a boolean")
    return value


__all__ = [
    "ArtifactBinding",
    "CrossProcessReplayVerdict",
    "ReproducibilityClosure",
    "RuntimeProcessContract",
    "SourceTreeBinding",
    "bind_source_tree",
    "build_reproducibility_closure",
    "canonical_json_hash",
    "evaluate_cross_process_replays",
    "file_sha256",
]
