"""One offline public SDK selection drives native path and mission mode."""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from rosclaw.agentd.cli import _resume_target_mode, _select_continuation_target
from rosclaw.storage.migrations import MigrationRunner

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "packages/rosclaw-agent/dist/src/main.js"
SDK = ROOT / "packages/rosclaw-agent/node_modules/@earendil-works/pi-coding-agent/dist/index.js"


def recorded(home, cwd, provider):
    code = """import {SessionManager} from SDK;
const m=SessionManager.create(CWD,DIR);
m.appendMessage({role:'user',content:'offline',timestamp:Date.now()});
m.appendMessage({role:'assistant',content:[{type:'text',text:'offline'}],api:'anthropic-messages',provider:PROVIDER,model:'k3',usage:{input:1,output:1,cacheRead:0,cacheWrite:0,totalTokens:2,cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}},stopReason:'stop',timestamp:Date.now()});
console.log(JSON.stringify({id:m.getSessionId(),path:m.getSessionFile()}));"""
    code = (
        code.replace("SDK", json.dumps(SDK.as_uri()))
        .replace("CWD", json.dumps(str(cwd)))
        .replace("DIR", json.dumps(str(home / "agent/sessions")))
        .replace("PROVIDER", json.dumps(provider))
    )
    return json.loads(
        subprocess.check_output(
            [shutil.which("node"), "--input-type=module", "-e", code],
            text=True,
            env=dict(os.environ, PI_OFFLINE="1"),
            timeout=10,
        )
    )


def test_actual_metadata_target_drives_mode_even_with_newer_corrupt_file(tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    cwd = repo / "nested/stage"
    (repo / ".git").mkdir(parents=True)
    cwd.mkdir(parents=True)
    (home / "agent/sessions").mkdir(parents=True)
    first = recorded(home, cwd, "kimi-coding")
    latest = recorded(home, repo, "openai-codex")
    os.utime(first["path"], (1000, 1000))
    os.utime(latest["path"], (2000, 2000))
    corrupt = home / "agent/sessions/bad.jsonl"
    corrupt.write_text("invalid header\n")
    os.utime(corrupt, (3000, 3000))
    (home / "agentd").mkdir()
    conn = sqlite3.connect(home / "agentd/missions.db")
    MigrationRunner().apply(conn, "sqlite")
    for record, mode in [(first, "SIMULATION"), (latest, "REAL")]:
        mission = "m_" + record["id"]
        conn.execute(
            "INSERT INTO missions(mission_id,owner_principal,goal_json,body_id,effective_body_hash,mode,state,budgets_json,authorization_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (mission, "private", "{}", "sim/ur5e", "h", mode, "RUNNING", "{}", "{}", "now", "now"),
        )
        conn.execute(
            "INSERT INTO pi_session_bindings(binding_id,pi_session_id,mission_id,created_at,created_by) VALUES(?,?,?,?,?)",
            ("b_" + record["id"], record["id"], mission, "now", "test"),
        )
    conn.commit()
    conn.close()
    target = _select_continuation_target(shutil.which("node"), str(ENTRY), home)
    assert target["id"] == latest["id"] and target["path"] == latest["path"]
    args = argparse.Namespace(continue_last=True, resume=None, _continuation_native_id=target["id"])
    assert _resume_target_mode(home, args) == "REAL"
    assert not (home / "agent/workspace.json").exists()


def test_no_history_actual_metadata_query_fails_without_new_session_or_workspace(tmp_path):
    home = tmp_path / "private"
    (home / "agent/sessions").mkdir(parents=True)
    with pytest.raises(subprocess.CalledProcessError):
        _select_continuation_target(shutil.which("node"), str(ENTRY), home)
    assert list((home / "agent/sessions").iterdir()) == []
    assert not (home / "agent/workspace.json").exists()


def test_metadata_target_outside_private_session_directory_rejected(tmp_path, monkeypatch):
    home = tmp_path / "private"
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **k: json.dumps(
            {"id": "id", "cwd": "/tmp", "path": str(tmp_path / "outside.jsonl")}
        ),
    )
    with pytest.raises(ValueError, match="OUTSIDE_SESSION_DIRECTORY"):
        _select_continuation_target("node", "entry", home)


def test_target_identity_is_shared_in_actual_cli_assembly():
    import ast

    source = (ROOT / "src/rosclaw/agentd/cli.py").read_text()
    tree = ast.parse(source)
    chat = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_chat_pi")
    # Selection occurs before service construction; argv and mode reference the
    # same selected target, not a second cwd-filtered session lookup.
    text = ast.unparse(chat)
    assert text.index("_select_continuation_target") < text.index("AgentService(")
    assert "args._continuation_native_id = continuation_target['id']" in text
    assert "resume_argv = ['--resume-path', continuation_target['path']]" in text
