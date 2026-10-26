"""Lossless delivery for large public ROS Expert observations, never task truth."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from rosclaw.agentd.service import AgentService
    from rosclaw.contracts.pi.tool_request import PiToolRequestV1

MAX_OBSERVATION_BYTES = 64 * 1024**2
_PUBLIC_OBSERVATIONS = frozenset(
    {
        "ros_inspect_system",
        "ros_diagnose_system",
        "ros_resolve_task",
        "ros_verify_mission",
    }
)


def project_observation(
    service: AgentService | None,
    request: PiToolRequestV1,
    projection: dict[str, Any],
) -> dict[str, Any]:
    """Store the complete projection in the existing owning task's evidence zone.

    Unsupported or unowned output retains the caller's explicit size rejection.
    No tail fields, original references or UNKNOWN facts are dropped. The small
    pointer requires reading the complete artifact; it asserts no task success.
    """
    if service is None or projection.get("capability_id") not in _PUBLIC_OBSERVATIONS:
        return projection
    kernel = service._task_kernel
    owner = kernel.active_task_for(request.mission_id, request.pi_session_id)
    if owner is None:
        return projection
    task = kernel.get_task(str(owner["task_id"]))
    if task is None:
        return projection
    content = json.dumps(projection, ensure_ascii=False, allow_nan=False, indent=2).encode()
    if len(content) > MAX_OBSERVATION_BYTES:
        raise ValueError("Public observation exceeds bounded artifact size")
    digest = hashlib.sha256(content).hexdigest()
    folder = Path(str(task["run_dir"])) / "evidence"
    # The task creates this zone. Do not create another output tree in a project.
    directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    filename = f"ros-observation-{digest}.json"
    try:
        if not _existing_matches(directory, filename, content):
            temporary = ".ros-observation-" + uuid4().hex + ".tmp"
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    # Publish only complete bytes; never overwrite an existing artifact.
                    os.link(
                        temporary,
                        filename,
                        src_dir_fd=directory,
                        dst_dir_fd=directory,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    if not _existing_matches(directory, filename, content):
                        raise ValueError("Concurrent observation artifact disappeared") from None
            finally:
                os.unlink(temporary, dir_fd=directory)
            os.fsync(directory)
    finally:
        os.close(directory)
    path = folder / filename
    ref = kernel.register_artifact(
        task_id=str(task["task_id"]),
        path=str(path),
        media_type="application/json",
        producer="kernel:capability:" + projection["capability_id"],
        metadata={"scope": "public_ROS_observation", "task_success_asserted": False},
    )
    if ref["sha256"] != digest or ref["size_bytes"] != len(content):
        raise ValueError("Registered observation differs from captured bytes")
    return {
        "status": projection["status"],
        "capability_id": projection["capability_id"],
        "value": {
            "result_delivery": "registered_lossless_artifact",
            "inline_value_available": False,
            "read_full_result_required": True,
            "task_success_asserted": False,
        },
        "artifact_refs": [
            {
                "artifact_id": ref["artifact_id"],
                "path": str(path),
                "sha256": digest,
                "size_bytes": len(content),
                "media_type": "application/json",
                "open_command": "rosclaw artifact open " + ref["artifact_id"],
            }
        ],
    }


def _existing_matches(directory: int, filename: str, content: bytes) -> bool:
    try:
        descriptor = os.open(
            filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
    except FileNotFoundError:
        return False
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Observation artifact must be a regular file")
        if stream.read(MAX_OBSERVATION_BYTES + 1) != content:
            raise ValueError("Observation artifact content changed")
    return True
