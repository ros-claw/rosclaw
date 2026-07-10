"""Tests for the standalone LeRobot worker_main.py.

These tests call ``worker_main.main`` directly with the current interpreter so
no real LeRobot runtime is needed for the ``inspect`` operation.
"""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.integrations.lerobot.worker_main import main


def test_worker_main_inspect_local_config(minimal_policy_dir: Path, tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "rosclaw.lerobot.worker.v1",
                "op": "inspect",
                "policy_path": str(minimal_policy_dir),
                "allow_network": False,
            }
        ),
        encoding="utf-8",
    )

    rc = main(["--request-json", str(request_path), "--output-json", str(response_path)])

    assert rc == 0
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "ok"
    assert response["op"] == "inspect"
    assert response["policy_metadata"]["policy_type"] == "act"
    assert response["policy_metadata"]["config_found"] is True


def test_worker_main_inspect_missing_config(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    empty_dir = tmp_path / "empty_policy"
    empty_dir.mkdir()
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "rosclaw.lerobot.worker.v1",
                "op": "inspect",
                "policy_path": str(empty_dir),
            }
        ),
        encoding="utf-8",
    )

    rc = main(["--request-json", str(request_path), "--output-json", str(response_path)])

    assert rc == 1
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "error"
    assert response["error"]["code"] == "policy_config_not_found"


def test_worker_main_rejects_network_for_hf_id(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "rosclaw.lerobot.worker.v1",
                "op": "inspect",
                "policy_path": "lerobot/example_policy",
                "allow_network": False,
            }
        ),
        encoding="utf-8",
    )

    rc = main(["--request-json", str(request_path), "--output-json", str(response_path)])

    assert rc == 1
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "error"
    assert response["error"]["code"] == "network_disabled"


def test_worker_main_unknown_op(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "rosclaw.lerobot.worker.v1",
                "op": "train",
                "policy_path": "/tmp/policy",
            }
        ),
        encoding="utf-8",
    )

    rc = main(["--request-json", str(request_path), "--output-json", str(response_path)])

    assert rc == 1
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "error"
    assert response["error"]["code"] == "worker_invalid_json"
