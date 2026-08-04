"""ModeldGateway — ModelGateway over rosclaw-modeld (批次 D §7.2).

AgentLoop 不再直接依赖 OpenAI-compatible 协议细节：本 gateway 把
ModelTurnRequest 转发给本地 modeld（UDS + bearer token），由 modeld 经
pi-ai 完成 provider 差异、认证与流式调用。

边界与失败语义：
- token 在启动时随机生成，只经子进程环境传递；不落盘、不进日志。
- modeld 崩溃/不可达 → ModelGatewayError（诚实失败，绝不伪造成功）。
- 凭据按 profile.api_key_ref 的 env:VAR 由 *agentd 侧* 环境注入 modeld
  子进程环境（modeld 自己读 env）；api_key_ref 的 *值* 永不出现在请求体。
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from rosclaw.agentd.models.gateway import (
    ModelGatewayError,
    ModelProbeResult,
    ModelTurnRequest,
)
from rosclaw.agentd.models.policy import ModelProfile
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1, ModelUsage, ToolCall
from rosclaw.contracts.common import new_id

#: provider 名映射：ModelProfile.provider → modeld provider id
_PROVIDER_MAP = {
    "kimi_cn": "moonshot",
    "kimi_code": "kimi-code",
    "moonshot": "moonshot",
    "kimi-code": "kimi-code",
    "ollama": "ollama",
}


def _find_modeld_runtime() -> tuple[str, str] | None:
    """(node ≥22.19, modeld entry) — 与 CLI 的 TUI 探测同一语义。"""
    candidates = [shutil.which("node"), "/usr/bin/node", "/usr/local/bin/node"]
    node = None
    for candidate in filter(None, candidates):
        try:
            out = subprocess.check_output([candidate, "--version"], text=True, timeout=10).strip()
            if [int(p) for p in out.lstrip("v").split(".")] >= [22, 19, 0]:
                node = candidate
                break
        except Exception:  # noqa: BLE001
            continue
    if node is None:
        return None
    entry_env = os.environ.get("ROSCLAW_MODELD_ENTRY")
    repo_entry = (
        Path(__file__).resolve().parents[4] / "packages" / "rosclaw-modeld" / "dist" / "src" / "main.js"
    )
    entry = entry_env or (str(repo_entry) if repo_entry.exists() else None)
    if not entry or not Path(entry).exists():
        return None
    return node, entry


class ModeldGateway:
    """ModelGateway 协议实现（complete_stream/probe/close）。"""

    def __init__(self, profile: ModelProfile, *, home: Path | None = None) -> None:
        self.profile = profile
        self._home = home or (Path.home() / ".rosclaw")
        self._provider = _PROVIDER_MAP.get(profile.provider, profile.provider)
        self._token = secrets.token_urlsafe(32)
        self._proc: subprocess.Popen | None = None
        self._socket_path = ""
        self._session = None
        self._last_error: str | None = None
        self._start_lock = asyncio.Lock()

    # -- lifecycle ------------------------------------------------------------

    async def _ensure_started(self) -> None:
        if self._session is not None:
            if self._proc is not None and self._proc.poll() is not None:
                # modeld 崩溃：诚实报错，不假装修复（重启由下一次调用显式触发）。
                self._last_error = f"modeld exited with code {self._proc.returncode}"
                raise ModelGatewayError("modeld_crashed", self._last_error)
            return
        async with self._start_lock:
            if self._session is not None:
                return
            await self._start_locked()

    async def _start_locked(self) -> None:
        runtime = _find_modeld_runtime()
        if runtime is None:
            raise ModelGatewayError(
                "modeld_unavailable",
                "rosclaw-modeld runtime not found (need Node >= 22.19 and built "
                "packages/rosclaw-modeld); use the legacy backend or run rosclaw doctor",
            )
        node, entry = runtime
        home = self._home / "agentd" / "modeld"
        home.mkdir(parents=True, exist_ok=True)
        # 每实例唯一 socket：多个 gateway（failover 候选、mgmt 通道、并发
        # mission）共存时，后启动的 modeld 不得 unlink 别人的 socket。
        self._socket_path = str(home / f"modeld-{os.getpid()}-{id(self) % 0xFFFF:x}.sock")
        env = dict(os.environ)
        env["ROSCLAW_MODELD_TOKEN"] = self._token
        # profile 引用的 env key 若存在则随子进程环境传递（值不进命令行/文件）；
        # 缺失时不拦——由 modeld 诚实报告 no_credential。
        self._proc = subprocess.Popen(  # noqa: S603 - fixed entry, no shell
            [node, entry, "--socket", self._socket_path, "--home", str(home)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import aiohttp

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise ModelGatewayError(
                    "modeld_crashed", f"modeld exited during startup (code {self._proc.returncode})"
                )
            if Path(self._socket_path).exists():
                break
            await asyncio.sleep(0.05)
        else:
            raise ModelGatewayError("modeld_timeout", "modeld did not create its socket in 10s")
        connector = aiohttp.UnixConnector(path=self._socket_path)
        self._session = aiohttp.ClientSession(
            connector=connector, headers={"authorization": f"Bearer {self._token}"}
        )

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    # -- management channel (commands: /providers /login /logout /model) ---------

    async def manage(self, method: str, path: str, payload: dict | None = None) -> dict:
        """管理面调用（/v1/providers、/v1/auth、/v1/models、login/logout）。

        只返回 modeld 的公开 JSON——secret 永不经过此通道的响应。
        """
        await self._ensure_started()
        try:
            async with self._session.request(
                method, f"http://localhost{path}", json=payload, timeout=30
            ) as resp:
                return await resp.json()  # type: ignore[no-any-return]
        except ModelGatewayError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelGatewayError(
                "modeld_transport", f"management call failed: {type(exc).__name__}: {exc}"
            ) from exc

    # -- ModelGateway protocol --------------------------------------------------

    async def probe(self) -> ModelProbeResult:
        try:
            await self._ensure_started()
        except ModelGatewayError as exc:
            return ModelProbeResult(reachable=False, error=f"{exc.kind}: {exc}")
        try:
            async with self._session.post(
                "http://localhost/v1/probe",
                json={"provider": self._provider, "model": self.profile.model},
                timeout=35,
            ) as resp:
                body = await resp.json()
        except Exception as exc:  # noqa: BLE001
            return ModelProbeResult(reachable=False, error=f"transport: {exc}")
        if body.get("ok"):
            return ModelProbeResult(reachable=True, chat_ok=True, tool_call_ok=None)
        return ModelProbeResult(
            reachable=True, error=f"{body.get('error')}: {body.get('message')}"
        )

    async def complete(self, request: ModelTurnRequest) -> ModelTurnResultV1:
        return await self.complete_stream(request, None)

    async def complete_stream(
        self, request: ModelTurnRequest, on_text_delta=None
    ) -> ModelTurnResultV1:
        await self._ensure_started()
        started = time.monotonic()
        payload: dict[str, Any] = {
            "provider": self._provider,
            "model": self.profile.model,
            "system_prompt": request.system_prompt,
            "messages": request.messages,
            "max_tokens": request.max_output_tokens,
            "tools": [t.to_openai()["function"] for t in request.tools],
        }
        effort = self.profile.vendor_parameters.get("reasoning_effort")
        if effort:
            payload["reasoning_effort"] = effort
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage = ModelUsage()
        finish_reason: str | None = None
        assistant_message: dict[str, Any] = {}
        try:
            async with self._session.post(
                "http://localhost/v1/stream",
                json=payload,
                timeout=self.profile.timeout_sec,
            ) as resp:
                buffer = ""
                async for chunk in resp.content.iter_any():
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n\n" in buffer:
                        frame, buffer = buffer.split("\n\n", 1)
                        data = ""
                        for line in frame.splitlines():
                            if line.startswith("data: "):
                                data += line[6:]
                        if not data:
                            continue
                        event = json.loads(data)
                        etype = event.get("type")
                        if etype == "text.delta":
                            text = event.get("text", "")
                            content_parts.append(text)
                            if on_text_delta is not None:
                                maybe = on_text_delta(text)
                                if asyncio.iscoroutine(maybe):
                                    await maybe
                        elif etype == "tool_call":
                            tool_calls.append(
                                ToolCall(
                                    call_id=event.get("call_id", ""),
                                    name=event.get("name", ""),
                                    arguments_json=json.dumps(
                                        event.get("arguments") or {}, ensure_ascii=False
                                    ),
                                )
                            )
                        elif etype == "usage":
                            usage = ModelUsage(
                                prompt_tokens=int(event.get("input", 0)),
                                completion_tokens=int(event.get("output", 0)),
                                total_tokens=int(event.get("total", 0)),
                            )
                        elif etype == "done":
                            finish_reason = event.get("stop_reason")
                            assistant_message = event.get("assistant_message") or {}
                        elif etype == "error":
                            raise ModelGatewayError(
                                str(event.get("kind", "provider_error")),
                                str(event.get("message", "modeld stream error")),
                            )
        except ModelGatewayError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport/crash → honest error
            raise ModelGatewayError(
                "modeld_transport", f"modeld stream failed: {type(exc).__name__}: {exc}"
            ) from exc
        if finish_reason is None and not tool_calls and not content_parts:
            raise ModelGatewayError("modeld_transport", "modeld stream ended without done event")
        from rosclaw.agentd.usage import estimate_cost_microunits

        usage = usage.model_copy(
            update={
                "cost_microunits": estimate_cost_microunits(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    price_input_per_mtok=self.profile.price_input_per_mtok_microunits,
                    price_output_per_mtok=self.profile.price_output_per_mtok_microunits,
                )
            }
        )
        return ModelTurnResultV1(
            turn_id=new_id("turn"),
            mission_id=request.mission_id,
            provider=self._provider,
            model=self.profile.model,
            profile=self.profile.name,
            content="".join(content_parts),
            tool_calls=tool_calls,
            assistant_message=assistant_message,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=int((time.monotonic() - started) * 1000),
            context_id=request.context_id,
            context_revision=request.context_revision,
        )
