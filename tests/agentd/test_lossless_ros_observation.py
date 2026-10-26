"""Large public observations retain exact tails and refs in owned task evidence."""

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.agentd.pi_bridge.tool_dispatch import ToolBridgeError, _envelope_result
from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.service import TaskKernel


@pytest.fixture
def context(tmp_path):
    home = tmp_path / "home"
    workspace = tmp_path / "project"
    workspace.mkdir()
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    MigrationRunner().apply(connection, "sqlite")
    kernel = TaskKernel(connection, home)
    bound = kernel.bind_message(
        mission_id="mission",
        session_ref="native",
        backend_native_id="native",
        message_id="input",
        text="Inspect public observations",
        cwd=str(workspace),
    )
    task = kernel.get_task(bound["task_id"])
    yield (
        SimpleNamespace(_task_kernel=kernel),
        SimpleNamespace(request_id="request", mission_id="mission", pi_session_id="native"),
        task,
        workspace,
    )
    connection.close()


def envelope(value=None, capability="ros_inspect_system"):
    return SimpleNamespace(
        status=SimpleNamespace(value="SUCCEEDED"),
        capability_id=capability,
        value=value
        or {
            "head": "x" * 9000,
            "tail": {"actual_source": "public ROS", "critical_status": "UNKNOWN"},
        },
        artifact_refs=[{"artifact_id": "previous_ref", "path": "original.json"}],
    )


def test_registered_result_preserves_complete_observation_and_references(context):
    service, request, task, workspace = context
    original = envelope()
    result = _envelope_result(request, original, service=service)
    pointer = json.loads(result.summary)
    assert result.ok and len(result.summary) <= 8000
    assert pointer["value"]["read_full_result_required"] is True
    assert pointer["value"]["task_success_asserted"] is False
    ref = pointer["artifact_refs"][0]
    path = Path(ref["path"])
    assert path.parent == Path(task["run_dir"]) / "evidence"
    assert not (workspace / "outputs").exists()
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == ref["sha256"]
    assert len(content) == ref["size_bytes"]
    document = json.loads(content)
    assert document["value"] == original.value
    assert document["artifact_refs"] == original.artifact_refs
    repeated = _envelope_result(request, original, service=service)
    assert json.loads(repeated.summary)["artifact_refs"][0]["artifact_id"] == ref["artifact_id"]


def test_no_active_owner_or_unsupported_capability_does_not_create_fake_success(context):
    service, request, task, _workspace = context
    request.pi_session_id = "foreign"
    with pytest.raises(ToolBridgeError, match="No partial JSON"):
        _envelope_result(request, envelope(), service=service)
    request.pi_session_id = "native"
    with pytest.raises(ToolBridgeError, match="No partial JSON"):
        _envelope_result(
            request, envelope(capability="semantic.observe_candidates"), service=service
        )
    assert not list((Path(task["run_dir"]) / "evidence").iterdir())


def test_existing_file_alias_is_rejected_without_overwriting_target(context, tmp_path):
    service, request, _task, _workspace = context
    first = json.loads(_envelope_result(request, envelope(), service=service).summary)
    path = Path(first["artifact_refs"][0]["path"])
    path.unlink()
    target = tmp_path / "unrelated"
    target.write_text("untouched")
    path.symlink_to(target)
    with pytest.raises(ToolBridgeError):
        _envelope_result(request, envelope(), service=service)
    assert target.read_text() == "untouched"


def test_evidence_directory_alias_is_rejected_without_writing_outside_task(context, tmp_path):
    service, request, task, _workspace = context
    folder = Path(task["run_dir"]) / "evidence"
    folder.rmdir()
    other = tmp_path / "outside"
    other.mkdir()
    folder.symlink_to(other, target_is_directory=True)
    with pytest.raises(ToolBridgeError):
        _envelope_result(request, envelope(), service=service)
    assert not list(other.iterdir())


def test_large_artifact_budget_fails_explicitly_before_writing(context, monkeypatch):
    from rosclaw.agentd.pi_bridge import observation_projection

    service, request, task, _workspace = context
    monkeypatch.setattr(observation_projection, "MAX_OBSERVATION_BYTES", 1024)
    with pytest.raises(ToolBridgeError):
        _envelope_result(request, envelope(), service=service)
    assert not list((Path(task["run_dir"]) / "evidence").iterdir())


def test_existing_content_is_never_overwritten_to_match_an_observation(context):
    service, request, _task, _workspace = context
    pointer = json.loads(_envelope_result(request, envelope(), service=service).summary)
    path = Path(pointer["artifact_refs"][0]["path"])
    path.write_text("changed outside the observation producer")
    with pytest.raises(ToolBridgeError):
        _envelope_result(request, envelope(), service=service)
    assert path.read_text() == "changed outside the observation producer"


def test_non_regular_existing_artifact_is_refused_without_blocking(context):
    import os

    service, request, _task, _workspace = context
    pointer = json.loads(_envelope_result(request, envelope(), service=service).summary)
    path = Path(pointer["artifact_refs"][0]["path"])
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ToolBridgeError):
        _envelope_result(request, envelope(), service=service)


def test_interrupted_publish_does_not_expose_or_register_partial_json(context, monkeypatch):
    from rosclaw.agentd.pi_bridge import observation_projection

    service, request, task, _workspace = context
    original_link = observation_projection.os.link

    def interrupted(*args, **kwargs):
        raise OSError("injected publish interruption")

    monkeypatch.setattr(observation_projection.os, "link", interrupted)
    with pytest.raises(ToolBridgeError):
        _envelope_result(request, envelope(), service=service)
    folder = Path(task["run_dir"]) / "evidence"
    assert not list(folder.iterdir())
    monkeypatch.setattr(observation_projection.os, "link", original_link)
    pointer = json.loads(_envelope_result(request, envelope(), service=service).summary)
    assert (
        json.loads(Path(pointer["artifact_refs"][0]["path"]).read_text())["value"]["tail"][
            "critical_status"
        ]
        == "UNKNOWN"
    )
