"""WorkSpecV2（十二审 PR-12.4，总纲 §5）：先定义"交付什么"，再雇 Worker。

任务类型驱动验收（不再是 profile 硬编码 text/plain + text/x-diff）：

- analyze：文字报告，可无 diff；
- code_change：必须有 diff + 测试；
- artifact_build：必须有指定媒体/文件（diff 可选）；
- simulation_run：必须有仿真 trace + receipt（动画是 trace 派生物）。
"""

from __future__ import annotations

from typing import Any, Literal

from rosclaw.contracts.common import ContractModel

TASK_TYPES = ("analyze", "code_change", "artifact_build", "simulation_run")


class DeliverableV1(ContractModel):
    SCHEMA = "rosclaw.deliverable.v1"
    schema_version: Literal["rosclaw.deliverable.v1"] = "rosclaw.deliverable.v1"
    id: str = ""
    media_types: list[str]
    required: bool = True
    validators: list[str] = ["exists", "non_empty"]


class WorkSpecV2(ContractModel):
    SCHEMA = "rosclaw.work_spec.v2"
    schema_version: Literal["rosclaw.work_spec.v2"] = "rosclaw.work_spec.v2"
    task_type: str = "analyze"
    goal: str = ""
    deliverables: list[DeliverableV1] = []
    acceptance_checks: list[str] = []
    budgets: dict[str, Any] = {}
    handoff: dict[str, Any] = {"partial_artifacts": True, "checkpoint_on_timeout": True}


def expected_media_types(spec: WorkSpecV2) -> list[str]:
    """deliverables → expected_output.artifacts（verifier 按 media_type
    匹配）。"""
    out: list[str] = []
    for d in spec.deliverables:
        if d.required:
            out.extend(d.media_types)
    return out


#: 媒体魔数（真实解码前的硬校验）。
MAGIC_BYTES: dict[str, tuple[bytes, ...]] = {
    "image/gif": (b"GIF87a", b"GIF89a"),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "video/mp4": (b"ftyp",),  # bytes 4-8
}


def validate_media_file(path, media_type: str) -> str | None:
    """存在性/非空/魔数校验。返回错误原因或 None（通过）。"""
    import json as _json
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.exists():
        return f"missing: {p.name}"
    data = p.read_bytes()
    if not data:
        return f"empty: {p.name}"
    if media_type == "application/json":
        try:
            _json.loads(data)
        except ValueError:
            return f"invalid json: {p.name}"
        return None
    magic = MAGIC_BYTES.get(media_type)
    if magic:
        if media_type == "video/mp4":
            if len(data) < 12 or data[4:8] != b"ftyp":
                return f"bad mp4 magic: {p.name}"
        elif not any(data.startswith(m) for m in magic):
            return f"bad {media_type} magic: {p.name}"
    return None
