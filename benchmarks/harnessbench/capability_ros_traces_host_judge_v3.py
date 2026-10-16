"""Private HOST adapter. Never stage this module or its dependencies to a solver.

Only the accepted Native judge decides mathematical/schema correctness. No
implementation exception from that judge is converted into an accepted refusal.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from benchmarks.harnessbench.capability_challenges_ros_traces_v3 import (
    FAMILIES,
    MAX_BYTES,
    ROLES,
    SCOPE,
    make_instance,
)
from benchmarks.harnessbench.capability_ros_traces_integration_v3 import (
    task_from_instance,
)
from benchmarks.harnessbench.task_common import BenchTask


def _verdict(passed: bool, code: str, detail: str) -> dict[str, Any]:
    return {
        "task_success": passed,
        "verified_success": passed,
        "false_success": False,
        "scope": SCOPE,
        "evidence_class": SCOPE,
        "physical_verified": False,
        "passed": passed,
        "code": code,
        "detail": detail,
        "reason": f"{SCOPE}:{code}:{detail}",
    }


def judge_ros_trace_workspace(task: BenchTask, workspace: Path) -> dict[str, Any]:
    """Bind identity and immutable on-disk bytes before evaluating exact answer.

    Recreate only public input, never reference output. A corrupt task cannot
    bless modified bytes by changing its digest. Metadata, prompt and staged
    text must all equal the frozen public generator for the declared identity.
    """
    metadata = task.oracle
    if not isinstance(metadata, dict):
        return _verdict(False, "TASK_BINDING", "metadata must be an object")
    family, role, seed = (metadata.get(k) for k in ("family", "role", "seed"))
    if family not in FAMILIES or role not in ROLES or type(seed) is not int:
        return _verdict(False, "TASK_BINDING", "invalid family/role/exact integer seed")
    instance = make_instance(family, role, seed)
    canonical_task = task_from_instance(instance)
    if task != canonical_task:
        return _verdict(False, "TASK_BINDING", "public identity or contract changed")
    root = Path(workspace)
    input_path, answer_path = root / "input.json", root / "answer.json"
    try:
        if input_path.is_symlink() or not input_path.is_file():
            return _verdict(False, "INPUT_INTEGRITY", "input is not a regular public file")
        with input_path.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if (raw != instance.staged()["input.json"]
                or hashlib.sha256(raw).hexdigest() != metadata["input_sha256"]):
            return _verdict(False, "INPUT_INTEGRITY", "immutable input bytes changed")
        if answer_path.is_symlink() or not answer_path.is_file():
            return _verdict(False, "ANSWER_MISSING", "answer.json must be a regular file")
        with answer_path.open("rb") as stream:
            answer_bytes = stream.read(MAX_BYTES + 1)
    except OSError as error:
        return _verdict(False, "WORKSPACE_IO", type(error).__name__)

    # Deliberately lazy and host-only. Do not catch judge bugs or replace its
    # schema/mathematics with a second evaluator in the integration layer.
    from benchmarks.harnessbench.capability_challenge_judge_ros_traces_v3 import (
        judge_bytes,
    )

    result = judge_bytes(instance, answer_bytes)
    return _verdict(result["passed"], result["code"], result["detail"])
