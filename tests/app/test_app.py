"""Capability-only App schema, store, CLI, and daemon runner tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rosclaw.app.cli import dispatch_app_argv
from rosclaw.app.runner import AppRunner
from rosclaw.app.schema import AppManifest
from rosclaw.app.store import AppStore
from rosclaw.kernel import ActionEnvelope, ExecutionMode


def test_bundled_apps_are_valid_and_only_call_declared_capabilities() -> None:
    root = AppStore.builtin_root()
    manifests = [AppManifest.from_path(path) for path in sorted(root.iterdir())]

    assert [manifest.metadata.name for manifest in manifests] == [
        "realsense-inspect",
        "rh56-rps",
    ]
    for manifest in manifests:
        assert {step.call for step in manifest.workflow}.issubset(manifest.requires.capabilities)


@pytest.mark.parametrize(
    "step_input",
    [
        {"device_path": "/dev/ttyUSB0"},
        {"register": 1135},
        {"topic": "/cmd_vel"},
        {"nested": {"value": "/sys/class/tty"}},
    ],
)
def test_app_schema_rejects_southbound_hardware_details(step_input: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="southbound"):
        AppManifest.model_validate(
            {
                "apiVersion": "rosclaw.io/v1",
                "kind": "App",
                "metadata": {"name": "unsafe-app"},
                "requires": {"capabilities": ["hand.move_finger"]},
                "workflow": [{"call": "hand.move_finger", "input": step_input}],
            }
        )


def test_app_store_install_is_digest_locked(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "home")
    installed = store.install("realsense-inspect")
    record, manifest = store.resolve("realsense-inspect")

    assert installed == record
    assert record.manifest_digest.startswith("sha256:")
    assert manifest.metadata.name == "realsense-inspect"

    (Path(record.path) / "app.yaml").write_text("kind: App\n", encoding="utf-8")
    with pytest.raises(Exception, match="integrity"):
        store.resolve("realsense-inspect")


def test_app_store_accepts_public_bundled_identifier(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "home")

    installed = store.install("ros-claw/realsense-inspect")

    assert installed.name == "realsense-inspect"


class _FakeDaemonClient:
    def __init__(self) -> None:
        self.actions: list[ActionEnvelope] = []
        self.closed: list[str] = []
        self.session: dict[str, Any] | None = None

    def create_session(self, **kwargs: Any) -> dict[str, Any]:
        self.session = kwargs
        return {"session": kwargs}

    def request_action(self, action: ActionEnvelope) -> dict[str, Any]:
        self.actions.append(action)
        return {"action_id": action.action_id}

    def wait_for_action(self, action_id: str, *, timeout_sec: float) -> dict[str, Any]:
        action = next(item for item in self.actions if item.action_id == action_id)
        receipt: dict[str, Any] = {
            "action_id": action_id,
            "final_state": "COMPLETED",
            "trust_level": "UNVERIFIED",
            "observations": [],
        }
        if action.capability_id == "camera.capture_rgbd":
            receipt["observations"] = [
                {
                    "artifacts": {
                        "color": "file:///evidence/color.png",
                        "depth": "file:///evidence/depth.png",
                    }
                }
            ]
        return {"state": "FINISHED", "receipt": receipt}

    def close_session(self, session_id: str, *, reason: str) -> dict[str, Any]:
        self.closed.append(session_id)
        return {"session_id": session_id, "reason": reason}


def test_app_runner_routes_every_step_through_daemon_capabilities() -> None:
    manifest = AppManifest.from_path(AppStore.builtin_root() / "realsense-inspect")
    client = _FakeDaemonClient()
    result = AppRunner(client).run(
        manifest,
        body_id="camera-one",
        body_snapshot_hash="sha256:camera",
        execution_mode=ExecutionMode.SHADOW,
    )

    assert result.status == "success"
    assert [action.capability_id for action in client.actions] == [
        "camera.capture_rgbd",
        "vlm.risk_assessment",
    ]
    assert client.actions[1].arguments["image"] == "file:///evidence/color.png"
    assert client.session is not None
    assert client.session["capability_scope"] == manifest.requires.capabilities
    assert client.closed == [result.session_id]


def test_app_cli_init_add_validate_and_install(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    assert dispatch_app_argv(["app", "init", "inspection", "--path", str(project)]) == 0
    capsys.readouterr()
    manifest_path = project / "inspection" / "app.yaml"
    assert (
        dispatch_app_argv(
            [
                "app",
                "add",
                "camera.capture_rgbd",
                "--app",
                str(manifest_path),
                "--save-as",
                "frame",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert dispatch_app_argv(["app", "validate", str(manifest_path), "--json"]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["capabilities"] == ["camera.capture_rgbd"]
    assert validation["steps"] == 1

    home = tmp_path / "home"
    assert (
        dispatch_app_argv(["app", "install", "realsense-inspect", "--home", str(home), "--json"])
        == 0
    )
    installed = json.loads(capsys.readouterr().out)
    assert installed["kind"] == "App"
