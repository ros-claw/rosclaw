"""Provider CLI contract tests for dry-run physical-AI routing."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock


def _run_cli(monkeypatch, capsys, argv: list[str]) -> tuple[int, dict]:
    from rosclaw.cli import main

    monkeypatch.setattr(sys, "argv", ["rosclaw", *argv])
    code = main()
    captured = capsys.readouterr()
    return code, json.loads(captured.out)


def test_provider_health_json_contract(monkeypatch, capsys):
    code, payload = _run_cli(monkeypatch, capsys, ["provider", "health", "--json"])

    assert code == 0
    assert payload["ok"] is True
    assert payload["provider_count"] >= 8
    assert any(provider["name"] == "vlm" for provider in payload["providers"])


def test_provider_route_explains_vlm_scene_graph(monkeypatch, capsys):
    code, payload = _run_cli(
        monkeypatch,
        capsys,
        ["provider", "route", "--capability", "vlm.scene_graph", "--json"],
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["selected_provider"] == "vlm"
    assert payload["fallbacks"] == ["world"]
    assert "declares capability" in payload["reason"]
    assert payload["requires_guard"] is True


def test_provider_route_reports_unroutable_capability(monkeypatch, capsys):
    code, payload = _run_cli(
        monkeypatch,
        capsys,
        ["provider", "route", "--capability", "missing.capability", "--json"],
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["selected_provider"] is None


def test_provider_benchmark_dry_run_json_contract(monkeypatch, capsys):
    code, payload = _run_cli(
        monkeypatch,
        capsys,
        ["provider", "benchmark", "--dry-run", "--json"],
    )

    assert code == 0
    assert payload["dry_run"] is True
    assert payload["status"] == "dry_run"
    assert any(item["capability"] == "vlm.scene_graph" for item in payload["route_plan"])


def test_provider_benchmark_requires_dry_run(monkeypatch, capsys):
    code, payload = _run_cli(monkeypatch, capsys, ["provider", "benchmark", "--json"])

    assert code == 1
    assert payload["ok"] is False
    assert "--dry-run" in payload["error"]


def test_provider_invoke_uses_builtin_deepseek_provider(monkeypatch, tmp_path):
    from rosclaw.cli import cmd_provider_invoke
    from rosclaw.provider.builtins.deepseek import DeepSeekProvider
    from rosclaw.provider.core.response import ProviderResponse

    infer = AsyncMock(
        return_value=ProviderResponse(
            request_id="trace-provider-test",
            provider="deepseek",
            capability="llm.chat",
            result={"text": "provider path"},
            status="ok",
        )
    )
    monkeypatch.setattr(DeepSeekProvider, "infer", infer)

    output = tmp_path / "provider.json"
    args = type(
        "Args",
        (),
        {
            "provider_id": "deepseek",
            "provider_id_opt": None,
            "input": '{"message": "hello"}',
            "capability": "llm.chat",
            "question": None,
            "image_path": None,
            "output_path": str(output),
            "json": True,
            "trace_id": "trace-provider-test",
        },
    )()

    assert cmd_provider_invoke(args) == 0
    infer.assert_awaited_once()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert json.loads(payload["raw"])["text"] == "provider path"


def test_provider_invoke_returns_failure_for_provider_error(monkeypatch, tmp_path):
    from rosclaw.cli import cmd_provider_invoke
    from rosclaw.provider.builtins.deepseek import DeepSeekProvider
    from rosclaw.provider.core.response import ProviderResponse

    monkeypatch.setattr(
        DeepSeekProvider,
        "infer",
        AsyncMock(
            return_value=ProviderResponse(
                request_id="trace-provider-error",
                provider="deepseek",
                capability="llm.chat",
                result={"error": "upstream unavailable"},
                status="error",
            )
        ),
    )

    output = tmp_path / "provider-error.json"
    args = type(
        "Args",
        (),
        {
            "provider_id": "deepseek",
            "provider_id_opt": None,
            "input": '{"message": "hello"}',
            "capability": "llm.chat",
            "question": None,
            "image_path": None,
            "output_path": str(output),
            "json": True,
            "trace_id": "trace-provider-error",
        },
    )()

    assert cmd_provider_invoke(args) == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["errors"] == ["upstream unavailable"]


def test_provider_invoke_deepseek_over_real_http(monkeypatch, tmp_path):
    from rosclaw.cli import cmd_provider_invoke

    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length)))
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "ROSCLAW_PROVIDER_OK",
                            }
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")

    output = tmp_path / "provider-http.json"
    args = type(
        "Args",
        (),
        {
            "provider_id": "deepseek",
            "provider_id_opt": None,
            "input": '{"message": "reply exactly"}',
            "capability": "llm.chat",
            "question": None,
            "image_path": None,
            "output_path": str(output),
            "json": True,
            "trace_id": "trace-provider-http",
        },
    )()

    try:
        assert cmd_provider_invoke(args) == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert json.loads(payload["raw"])["text"] == "ROSCLAW_PROVIDER_OK"
    assert requests[0]["model"] == "test-model"
    assert requests[0]["messages"][0]["content"] == "reply exactly"
