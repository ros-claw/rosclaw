from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw.simforge.reproducibility import (
    ArtifactBinding,
    ReproducibilityClosure,
    RuntimeProcessContract,
    bind_source_tree,
    build_reproducibility_closure,
    canonical_json_hash,
    evaluate_cross_process_replays,
    file_sha256,
)


def _closure(tmp_path: Path, *, expected_replays: int = 3) -> ReproducibilityClosure:
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    artifact = tmp_path / "actor.bin"
    artifact.write_bytes(b"actor")
    return build_reproducibility_closure(
        source_trees={"task": source},
        artifacts={"actor": artifact},
        expected_replays=expected_replays,
    )


def _worker(
    closure: ReproducibilityClosure,
    process_id: int,
    *,
    score: float = 1.0,
) -> dict[str, object]:
    return {
        "process_id": process_id,
        "process_contract": closure.process_contract.to_dict(),
        "closure_hash": closure.closure_hash,
        "passed": True,
        "evaluation": {"passed": True, "score": score},
        "trajectory_digest": "sha256:" + "a" * 64,
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }


def test_source_tree_binding_is_path_independent_and_content_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root in (first, second):
        (root / "a.py").write_text("A = 1\n", encoding="utf-8")
        (root / "ignored.txt").write_text("not selected", encoding="utf-8")

    left = bind_source_tree("task", first)
    right = bind_source_tree("task", second)
    assert left == right

    (second / "a.py").write_text("A = 2\n", encoding="utf-8")
    assert bind_source_tree("task", second).digest != left.digest


def test_source_tree_rejects_selected_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "target.py"
    target.write_text("A = 1\n", encoding="utf-8")
    (source / "linked.py").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        bind_source_tree("task", source)

    root_link = tmp_path / "source-link"
    root_link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="root cannot be a symlink"):
        bind_source_tree("task", root_link)


def test_artifact_binding_streams_exact_file_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.bin"
    artifact.write_bytes(b"policy-v1")
    binding = ArtifactBinding.from_path("policy", artifact)

    assert binding.digest == file_sha256(artifact)
    assert binding.size_bytes == 9
    artifact.write_bytes(b"policy-v2")
    assert file_sha256(artifact) != binding.digest


def test_closure_is_path_independent_sorted_and_sim_only(tmp_path: Path) -> None:
    closure = _closure(tmp_path)
    payload = closure.to_dict()

    assert payload["activation_ceiling"] == "SIM_ONLY"
    assert payload["hardware_authorized"] is False
    assert closure.closure_hash == canonical_json_hash(payload)
    assert all("/" not in item["label"] for item in payload["source_trees"])
    assert ReproducibilityClosure.from_dict(payload) == closure
    with pytest.raises(ValueError, match="cannot authorize hardware"):
        replace(closure, hardware_authorized=True)


def test_runtime_process_contract_round_trips() -> None:
    contract = RuntimeProcessContract.capture()

    assert RuntimeProcessContract.from_dict(contract.to_dict()) == contract
    assert contract.cpu_count >= 1

    incomplete = contract.to_dict()
    del incomplete["thread_environment"]["OMP_NUM_THREADS"]
    with pytest.raises(ValueError, match="mapping is invalid"):
        RuntimeProcessContract.from_dict(incomplete)


def test_three_fresh_exact_workers_pass(tmp_path: Path) -> None:
    closure = _closure(tmp_path)
    workers = [_worker(closure, process_id) for process_id in (1001, 1002, 1003)]

    verdict = evaluate_cross_process_replays(
        closure,
        workers,
        exact_fields=("evaluation", "trajectory_digest"),
        launcher_process_id=os.getpid(),
    )

    assert verdict.passed is True
    assert all(dict(verdict.gates).values())
    assert verdict.process_ids == (1001, 1002, 1003)
    assert verdict.verdict_hash.startswith("sha256:")
    verdict.require_passed()


def test_actual_fresh_python_workers_share_the_closure(tmp_path: Path) -> None:
    closure = _closure(tmp_path)
    code = """
import json
import os
from rosclaw.simforge import RuntimeProcessContract

print(json.dumps({
    "process_id": os.getpid(),
    "process_contract": RuntimeProcessContract.capture().to_dict(),
    "closure_hash": os.environ["ROSCLAW_TEST_CLOSURE_HASH"],
    "passed": True,
    "evaluation": {"passed": True, "score": 1.0},
    "trajectory_digest": "sha256:" + "a" * 64,
    "activation_ceiling": "SIM_ONLY",
    "hardware_authorized": False,
    "hardware_command_sent": False,
}))
"""
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH", "")) if value
    )
    environment["ROSCLAW_TEST_CLOSURE_HASH"] = closure.closure_hash
    workers = [
        json.loads(
            subprocess.run(
                (sys.executable, "-c", code),
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            ).stdout
        )
        for _ in range(closure.expected_replays)
    ]

    verdict = evaluate_cross_process_replays(
        closure,
        workers,
        exact_fields=("evaluation", "trajectory_digest"),
        launcher_process_id=os.getpid(),
    )

    assert verdict.passed is True
    assert len(verdict.process_ids) == 3


@pytest.mark.parametrize(
    ("mutation", "failed_gate"),
    (
        ("duplicate_pid", "fresh_process_identity"),
        ("evaluation", "cross_process_exact_replay"),
        ("process_contract", "process_contract_identical"),
        ("closure_hash", "closure_bound"),
        ("hardware", "all_workers_sim_only_safe"),
        ("outcome", "worker_outcomes_passed"),
    ),
)
def test_replay_mutations_fail_closed(
    tmp_path: Path,
    mutation: str,
    failed_gate: str,
) -> None:
    closure = _closure(tmp_path)
    workers = [_worker(closure, process_id) for process_id in (1001, 1002, 1003)]
    if mutation == "duplicate_pid":
        workers[2]["process_id"] = 1002
    elif mutation == "evaluation":
        workers[2]["evaluation"] = {"passed": True, "score": 0.9}
    elif mutation == "process_contract":
        workers[2]["process_contract"] = {
            **closure.process_contract.to_dict(),
            "machine": "changed",
        }
    elif mutation == "closure_hash":
        workers[2]["closure_hash"] = "sha256:" + "b" * 64
    elif mutation == "hardware":
        workers[2]["hardware_authorized"] = True
    else:
        workers[2]["passed"] = False

    verdict = evaluate_cross_process_replays(
        closure,
        workers,
        exact_fields=("evaluation", "trajectory_digest"),
    )

    assert verdict.passed is False
    assert dict(verdict.gates)[failed_gate] is False
    with pytest.raises(ValueError, match=failed_gate):
        verdict.require_passed()


def test_wrong_worker_count_and_launcher_reuse_fail(tmp_path: Path) -> None:
    closure = _closure(tmp_path, expected_replays=2)
    launcher = os.getpid()
    workers = [_worker(closure, launcher)]

    verdict = evaluate_cross_process_replays(
        closure,
        workers,
        exact_fields=("evaluation",),
        launcher_process_id=launcher,
    )

    assert verdict.passed is False
    assert dict(verdict.gates)["expected_worker_count"] is False
    assert dict(verdict.gates)["fresh_process_identity"] is False


def test_nonfinite_or_incomplete_exact_replay_is_rejected(tmp_path: Path) -> None:
    closure = _closure(tmp_path, expected_replays=2)
    workers = [_worker(closure, process_id) for process_id in (1001, 1002)]
    workers[1]["evaluation"] = {"score": float("nan")}
    with pytest.raises(ValueError, match="finite canonical JSON"):
        evaluate_cross_process_replays(
            closure,
            workers,
            exact_fields=("evaluation",),
        )

    with pytest.raises(ValueError, match="worker report must be a mapping"):
        evaluate_cross_process_replays(
            closure,
            [workers[0], "not-a-report"],  # type: ignore[list-item]
            exact_fields=("evaluation",),
        )

    del workers[1]["evaluation"]
    with pytest.raises(ValueError, match="missing exact replay"):
        evaluate_cross_process_replays(
            closure,
            workers,
            exact_fields=("evaluation",),
        )


def test_closure_json_contains_no_local_paths(tmp_path: Path) -> None:
    closure = _closure(tmp_path)

    serialized = json.dumps(closure.to_dict(), sort_keys=True)
    assert str(tmp_path) not in serialized
