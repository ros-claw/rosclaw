"""U01–U10 验收矩阵 harness（0914 PR-5，审计 §7）。

原则（审计条款）：
- 所有端到端场景从**正式安装的 CLI** 进入（wheel 构建+安装——
  不只调内部函数）；测试模型配置场景用受控 provider 测试服务；
- 结果**机器生成**（UResult → u-matrix.json + markdown 表——
  不手工填表；0912 报告 7329/7336 口径混排教训）；
- 真实模型场景（U05/U06/U10）无 key 一律 NOT_RUN——不合成冒充。
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


@dataclass
class UResult:
    scenario: str
    verdict: str  # PASS | FAIL | NOT_RUN
    wall_time_s: float
    evidence: list[str] = field(default_factory=list)
    detail: str = ""

    @classmethod
    def timed(cls, scenario: str, verdict: str, started: float, **kw) -> UResult:
        return cls(
            scenario=scenario,
            verdict=verdict,
            wall_time_s=round(time.monotonic() - started, 2),
            **kw,
        )


def write_matrix(results: list[UResult], out_dir: Path) -> Path:
    """机器生成矩阵：JSON（权威）+ markdown 表（人读）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "rosclaw.u_matrix.v1",
        "generated_by": "tests/acceptance/u_matrix (machine-generated)",
        "total": len(results),
        "pass": sum(1 for r in results if r.verdict == "PASS"),
        "fail": sum(1 for r in results if r.verdict == "FAIL"),
        "not_run": sum(1 for r in results if r.verdict == "NOT_RUN"),
        "results": [asdict(r) for r in results],
    }
    (out_dir / "u-matrix.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    lines = [
        "# U01–U10 验收矩阵（机器生成——勿手改）",
        "",
        "| 场景 | 判定 | 耗时(s) | 证据 |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.scenario} | {r.verdict} | {r.wall_time_s} | "
            f"{'; '.join(r.evidence)[:160]} |"
        )
    lines.append("")
    lines.append(
        f"合计 {payload['total']}：PASS {payload['pass']} / "
        f"FAIL {payload['fail']} / NOT_RUN {payload['not_run']}"
    )
    (out_dir / "u-matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir / "u-matrix.json"


# ---------------------------------------------------------------- 受控 provider


class FakeProvider:
    """行为可配的 OpenAI 兼容假 provider（配置场景专用）。

    模式：ok / wrong_key(401) / quota(403 配额) / rate_limited(429) /
    offline（不启动）。chat completions 返回固定文本；/models 返回
    固定模型目录。非 stream 与 stream 都支持（probe/chat 两路径）。
    """

    def __init__(self, mode: str = "ok", model: str = "fake-k3") -> None:
        if mode == "offline":
            raise ValueError("offline 模式不启动服务——直接拿不可达地址")
        self.mode = mode
        self.model = model
        self.requests: list[dict] = []
        handler = self._make_handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _make_handler(self):
        outer = self

        class _H(BaseHTTPRequestHandler):
            def log_message(self, *_a):  # 静默
                return

            def _send(self, code: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                if self.path.rstrip("/").endswith("models"):
                    if outer.mode == "wrong_key":
                        self._send(401, {"error": {"message": "invalid api key"}})
                        return
                    self._send(200, {
                        "object": "list",
                        "data": [{"id": outer.model, "object": "model"}],
                    })
                    return
                self._send(404, {"error": {"message": "not found"}})

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    body = {}
                outer.requests.append(body)
                if outer.mode == "wrong_key":
                    self._send(401, {"error": {"message": "invalid api key"}})
                    return
                if outer.mode == "quota":
                    self._send(403, {
                        "error": {"message": "quota exceeded for this billing period"}
                    })
                    return
                if outer.mode == "rate_limited":
                    self._send(429, {"error": {"message": "rate limit exceeded"}})
                    return
                if body.get("stream"):
                    # 流式 chat（init/doctor 的 probe 与 chat 真回合都
                    # 走 SSE）——固定两帧文本 + done。
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    chunk = {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": outer.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": "OK"},
                            "finish_reason": None,
                        }],
                    }
                    end = {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": outer.model,
                        "choices": [{
                            "index": 0, "delta": {}, "finish_reason": "stop",
                        }],
                    }
                    for payload in (chunk, end):
                        self.wfile.write(
                            f"data: {json.dumps(payload)}\n\n".encode()
                        )
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                self._send(200, {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "created": 1,
                    "model": outer.model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                })

        return _H

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


# ---------------------------------------------------------------- 安装 CLI 运行


def run_cli(
    prefix: Path,
    args: list[str],
    *,
    env_extra: dict | None = None,
    timeout: int = 120,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    """从正式安装前缀运行 rosclaw CLI（审计：从安装产物进入）。"""
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    env["PATH"] = f"{prefix / 'bin'}:{env.get('PATH', '')}"
    return subprocess.run(
        [str(prefix / "bin" / "rosclaw"), *args],
        env=env,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def cli_json(proc: subprocess.CompletedProcess) -> dict:
    """解析 CLI 的 JSON stdout（raw_decode——前面可有日志行，
    JSON 文档内部不再按行猜（reversed 逐行会被 JSON 内嵌行骗）。"""
    text = proc.stdout
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            doc, _end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(doc, dict):
            return doc
    raise AssertionError(f"CLI 无 JSON 输出: {proc.stdout[-400:]} {proc.stderr[-200:]}")
