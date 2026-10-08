"""Terminal capture reads counters and states without promoting failed tasks."""

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"


def load():
    spec = importlib.util.spec_from_file_location("native_counter_capture", ROOT / "native.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def database(root, state="FAILED", rows=1):
    p = root / "home/agentd/missions.db"
    p.parent.mkdir(parents=True)
    with sqlite3.connect(p) as db:
        db.execute(
            "create table model_usage(provider,model,prompt_tokens,completion_tokens,total_tokens,finish_reason)"
        )
        db.executemany(
            "insert into model_usage values(?,?,?,?,?,?)",
            [("source-provider", "actual-model", 11, 5, 16, "stop")] * rows,
        )
        db.execute("create table tasks(state)")
        db.execute("insert into tasks values(?)", (state,))
        db.execute("create table private_authorization(secret)")
        db.execute("insert into private_authorization values('never-export-this-secret')")
    return p


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED", "SUCCEEDED", "RUNNING"])
def test_capture_retains_actual_terminal_or_unclosed_state_and_public_usage(tmp_path, state):
    p = database(tmp_path, state)
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    load().capture_terminal_counters(tmp_path)
    assert json.loads((tmp_path / "task-kernel-final-states.json").read_text())["states"] == [state]
    usage = json.loads((tmp_path / "usage.json").read_text())
    assert usage == [
        {
            "provider": "source-provider",
            "model": "actual-model",
            "prompt_tokens": 11,
            "completion_tokens": 5,
            "total_tokens": 16,
            "finish_reason": "stop",
        }
    ]
    assert "never-export" not in (tmp_path / "usage.json").read_text()
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before


def test_missing_database_is_not_created_by_capture(tmp_path):
    with pytest.raises(sqlite3.Error):
        load().capture_terminal_counters(tmp_path)
    assert not (tmp_path / "home/agentd/missions.db").exists()
    assert not (tmp_path / "usage.json").exists()


def test_oversized_usage_refuses_instead_of_truncating_valid_cost(tmp_path):
    database(tmp_path, rows=257)
    with pytest.raises(ValueError, match="exceeds bound"):
        load().capture_terminal_counters(tmp_path)
    assert not (tmp_path / "usage.json").exists()
