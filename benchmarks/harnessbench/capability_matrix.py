"""Offline capability contracts, with independent immutable-input grading.

The native agent gets data and task contract only. No reference implementation
or expected answers are staged. PASS means fixture semantics, never live robotics.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from benchmarks.harnessbench.capability_cases import CASES
from benchmarks.harnessbench.capability_hard_cases import HARD_CASES
from benchmarks.harnessbench.task_common import BenchTask


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _task(index: int, direction: str, instruction: str, inputs: dict, expected: dict) -> BenchTask:
    fields = dict.fromkeys(expected, "<required value>")
    return BenchTask(
        task_id=f"C{index:02d}",
        category="offline_contract",
        staged_files={"input.json": _json(inputs)},
        oracle={"kind": "offline_capability", "direction": direction},
        prompt=(
            f"Offline robotics contract probe: {direction}.\n"
            "Read input.json; it is synthetic fixture data, not a live device. "
            "Do not access hardware, DDS, ROS transports or install dependencies.\n"
            f"{instruction}\n"
            "Independently solve the task using ROSClaw tools or local offline computation. "
            "Write answer.json with exactly this contract: "
            + _json(
                {
                    "scope": "FIXTURE_ONLY",
                    "result": fields,
                    "input_sha256": "<SHA256 of exact input.json bytes>",
                    "reason": "<explain your calculation and evidence limits>",
                }
            )
            + "Preserve input.json unchanged. Do not claim live integration or physical success."
        ),
    )


CAPABILITY_TASKS = {f"C{i:02d}": _task(i, *case) for i, case in enumerate(CASES, start=1)}


CASE_BY_ID = {f"C{i:02d}": case for i, case in enumerate(CASES, 1)}
for _index, _case in enumerate(HARD_CASES, 1):
    _id = f"CX{_index:02d}"
    _base = _task(_index, *_case)
    CAPABILITY_TASKS[_id] = BenchTask(
        task_id=_id,
        category=_base.category,
        prompt=_base.prompt,
        staged_files=_base.staged_files,
        oracle=_base.oracle,
    )
    CASE_BY_ID[_id] = _case


def _equal(actual: Any, expected: Any) -> bool:
    """Reject NaN, coercion, bool-as-number, wrong dimensions and extra keys."""
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)):
        return (
            type(actual) in (int, float)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_equal(a, e) for a, e in zip(actual, expected, strict=True))
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_equal(actual[key], value) for key, value in expected.items())
        )
    raise TypeError(f"unsupported expected type {type(expected)}")


def judge_capability(task_id: str, root: Path) -> dict[str, Any]:
    task = CAPABILITY_TASKS[task_id]
    expected = CASE_BY_ID[task_id][3]
    original = task.staged_files["input.json"].encode()
    expected_hash = hashlib.sha256(original).hexdigest()
    checks: dict[str, bool] = {}
    try:
        answer = json.loads((root / "answer.json").read_text())
        checks["answer_object"] = isinstance(answer, dict)
        if not isinstance(answer, dict):
            raise ValueError("answer must be object")
        checks["input_unchanged"] = (root / "input.json").read_bytes() == original
        checks["input_binding"] = answer.get("input_sha256") == expected_hash
        checks["fixture_scope"] = answer.get("scope") == "FIXTURE_ONLY"
        checks["semantic_result"] = _equal(answer.get("result"), expected)
        checks["reason_present"] = isinstance(answer.get("reason"), str) and bool(
            answer["reason"].strip()
        )
        checks["contract_keys"] = set(answer) == {"scope", "result", "input_sha256", "reason"}
    except (OSError, ValueError, TypeError, OverflowError):
        checks["parse_or_input"] = False
    passed = bool(checks) and all(checks.values())
    return {
        "task_success": passed,
        "verified_success": passed,
        "false_success": False,
        "reason": "fixture_verified" if passed else "fixture_contract_failed",
        "checks": checks,
        "evidence_scope": "FIXTURE_ONLY",
        "live_integration": "NOT_RUN",
        "direction": task.oracle["direction"],
    }


def public_manifest() -> dict:
    """Public coverage includes boundaries and success predicates, not answers."""
    return {
        "schema_version": "rosclaw.capability_matrix.v1",
        "scope": "FIXTURE_ONLY",
        "live_integration": "NOT_RUN",
        "task_count": len(CASE_BY_ID),
        "base_task_count": len(CASES),
        "composite_task_count": len(HARD_CASES),
        "cases": [
            {
                "task_id": task_id,
                "direction": case[0],
                "trigger": case[2],
                "contract": case[1],
                "success": "semantic result, immutable-input SHA256, truthful scope",
                "failure": "wrong value/type/shape, altered input, stale binding, unsupported live claim",
            }
            for task_id, case in CASE_BY_ID.items()
        ],
    }
