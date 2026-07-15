"""CLI and Dashboard surfaces over the local structured trace store."""

from __future__ import annotations

import json
import sys

from fastapi.testclient import TestClient

from rosclaw.cli import main
from rosclaw.dashboard.web_server import DashboardWebServer
from rosclaw.observability.store import TraceStore


def _write_trace(path):
    records = [
        {
            "schema_version": "rosclaw.trace.v1",
            "record_type": "span",
            "event_id": "evt-root",
            "trace_id": "trace-ui",
            "span_id": "span-root",
            "parent_span_id": None,
            "name": "runtime.execute",
            "span_kind": "MISSION",
            "source": "runtime",
            "operation": "agent.closed_loop",
            "started_at": 1.0,
            "ended_at": 1.2,
            "duration_ms": 200.0,
            "status": "OK",
            "attributes": {},
            "input": {"goal": "inspect"},
            "output": {"status": "ok"},
        },
        {
            "schema_version": "rosclaw.trace.v1",
            "record_type": "span",
            "event_id": "evt-child",
            "trace_id": "trace-ui",
            "span_id": "span-child",
            "parent_span_id": "span-root",
            "name": "provider.inference",
            "span_kind": "VLM",
            "source": "mock-vlm",
            "operation": "vlm.scene_understanding",
            "started_at": 1.02,
            "ended_at": 1.1,
            "duration_ms": 80.0,
            "status": "OK",
            "attributes": {"model": "mock"},
            "input": {"image": "artifact://frame.jpg"},
            "output": {"objects": ["door"]},
        },
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


def test_dashboard_trace_api_and_page(tmp_path):
    path = tmp_path / "live.jsonl"
    _write_trace(path)
    web = DashboardWebServer(trace_store=TraceStore(path=path))
    client = TestClient(web.app)

    listing = client.get("/api/traces").json()
    assert listing["count"] == 1
    assert listing["traces"][0]["span_count"] == 2

    trace = client.get("/api/traces/trace-ui").json()
    assert trace["tree"][0]["children"][0]["name"] == "provider.inference"
    assert client.get("/api/traces/events/evt-child").json()["span_kind"] == "VLM"
    page = client.get("/traces")
    assert page.status_code == 200
    assert "ROSClaw Trace" in page.text
    assert "Input (redacted)" in page.text


def test_trace_cli_list_show_and_export(tmp_path, capsys, monkeypatch):
    path = tmp_path / "live.jsonl"
    output = tmp_path / "export.json"
    _write_trace(path)

    monkeypatch.setattr(sys, "argv", ["rosclaw", "trace", "list", "--path", str(path)])
    assert main() == 0
    assert "trace-ui" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["rosclaw", "trace", "show", "trace-ui", "--path", str(path)],
    )
    assert main() == 0
    shown = capsys.readouterr().out
    assert "runtime.execute" in shown
    assert "provider.inference" in shown

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rosclaw",
            "trace",
            "export",
            "trace-ui",
            "--path",
            str(path),
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["span_count"] == 2
