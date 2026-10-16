"""Public H11--H20 discovery adapter; no host evaluation or execution.

The ordinary registry is explicitly seed 7 / v3.1. CLI IDs are uppercase
aliases, not replacements for the canonical, case-sensitive Instance identity.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

from benchmarks.harnessbench.capability_challenges_ros_traces_v3 import (
    FAMILIES,
    ROLES,
    SCOPE,
    VERSION,
    Instance,
    make_plan,
)
from benchmarks.harnessbench.task_common import BenchTask

DEFAULT_SEED = 7
REGISTRY_VERSION = VERSION
KIND = "offline_ros_traces_v3"


def registry_id(canonical: str) -> str:
    """Match the existing CLI's uppercase lookup, without adding CLI flags."""
    return canonical.upper()


def staged_text(instance: Instance) -> dict[str, str]:
    """Adapt only public input bytes to the existing UTF8 text writer.

    Never parse/re-serialize input: declaration order, Unicode and final newline
    belong to the immutable fixture. A non-UTF8 leaf is not substitutable.
    """
    leaves = instance.files
    if len(leaves) != 1 or leaves[0][0] != "input.json":
        raise ValueError("H11_V31_PUBLIC_STAGE_ONLY_INPUT")
    raw = leaves[0][1]
    if type(raw) is not bytes:
        raise ValueError("H11_V31_INPUT_BYTES_REQUIRED")
    text = raw.decode("utf-8", errors="strict")
    digest = hashlib.sha256(raw).hexdigest()
    if (text.encode("utf-8", errors="strict") != raw
            or instance.manifest()["input_hashes"] != {"input.json": digest}):
        raise ValueError("H11_V31_UTF8_DIGEST_MISMATCH")
    return {"input.json": text}


def task_from_instance(instance: Instance) -> BenchTask:
    """Bind public metadata; this function has no private evaluation dependency."""
    if (instance.family not in FAMILIES or instance.role not in ROLES
            or type(instance.seed) is not int
            or instance.task_id != f"{instance.family}-{instance.role}-{instance.seed}"):
        raise ValueError("H11_V31_INSTANCE_IDENTITY")
    files = staged_text(instance)
    metadata = {
        "kind": KIND,
        "version": VERSION,
        "scope": SCOPE,
        "canonical_task_id": instance.task_id,
        "registry_task_id": registry_id(instance.task_id),
        "family": instance.family,
        "role": instance.role,
        "seed": instance.seed,
        "input_sha256": hashlib.sha256(instance.files[0][1]).hexdigest(),
    }
    return BenchTask(
        task_id=metadata["registry_task_id"],
        category="understanding",
        prompt=instance.prompt(),
        staged_files=files,
        oracle=metadata,
    )


def make_ros_trace_tasks(
    seed: int = DEFAULT_SEED, *, occupied: Iterable[str] = (),
) -> dict[str, BenchTask]:
    """Pure public factory. Reject both exact and uppercase alias collisions.

    `occupied` permits standalone callers to preflight a namespace. The frozen
    tasks.py registry independently checks its own old/new key intersection.
    """
    if type(seed) is not int:
        raise ValueError("H11_V31_EXACT_INTEGER_SEED")
    used = set()
    for key in occupied:
        if not isinstance(key, str):
            raise ValueError("H11_V31_TASK_ID_TYPE")
        alias = registry_id(key)
        if alias in used:
            raise ValueError("H11_V31_TASK_ID_COLLISION")
        used.add(alias)
    tasks: dict[str, BenchTask] = {}
    canonical_ids: set[str] = set()
    for instance in make_plan(seed):
        task = task_from_instance(instance)
        if instance.task_id in canonical_ids or task.task_id in used:
            raise ValueError("H11_V31_TASK_ID_COLLISION")
        canonical_ids.add(instance.task_id)
        used.add(task.task_id)
        tasks[task.task_id] = task
    return tasks


ROS_TRACE_TASKS = make_ros_trace_tasks(DEFAULT_SEED)
