"""HTTP Runtime Adapter."""

import asyncio
from typing import Any

from rosclaw.provider.core.errors import RuntimeAdapterError
from rosclaw.provider.runtimes.base import RuntimeAdapter


class HTTPRuntime(RuntimeAdapter):
    """Async HTTP runtime adapter using aiohttp."""

    def __init__(
        self,
        name: str,
        endpoint: str,
        timeout_sec: float = 30.0,
        retries: int = 1,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(name, config={"endpoint": endpoint, "timeout_sec": timeout_sec})
        self.endpoint = endpoint.rstrip("/")
        self.timeout_sec = timeout_sec
        self.retries = retries
        self.headers = headers or {}
        self._session = None

    async def start(self) -> None:
        try:
            import aiohttp
        except ImportError as err:
            raise RuntimeError("aiohttp is required for HTTPRuntime. pip install aiohttp") from err
        self._session = aiohttp.ClientSession(headers=self.headers)
        self._started = True

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        self._started = False

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_started()
        if self._session is None:
            raise RuntimeAdapterError("Session not initialized", provider=self.name)

        import aiohttp

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                async with self._session.post(
                    self.endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_sec),
                ) as resp:
                    body = await resp.json()
                    if resp.status >= 400:
                        raise RuntimeAdapterError(
                            f"HTTP {resp.status}: {body}",
                            provider=self.name,
                        )
                    return body
            except Exception as e:
                last_error = e
                if attempt < self.retries:
                    await asyncio.sleep(0.5 * (attempt + 1))

        raise RuntimeAdapterError(
            f"HTTP invoke failed after {self.retries + 1} attempts: {last_error}",
            provider=self.name,
        )

    async def health(
        self,
        check_config: Any = None,
        model_id: str = "",
    ) -> dict[str, Any]:
        """Probe configured health endpoints and return a detailed status.

        If ``check_config`` is not provided, a lightweight default probe is
        performed against the runtime endpoint root.
        """
        import aiohttp

        try:
            self.ensure_started()
        except RuntimeError as exc:
            return {
                "ok": False,
                "reason": str(exc),
                "endpoint": self.endpoint,
            }
        if self._session is None:
            return {
                "ok": False,
                "reason": "session not initialized",
                "endpoint": self.endpoint,
            }

        endpoints: list[dict[str, Any]] = []
        if check_config is not None and getattr(check_config, "endpoints", None):
            for ep in check_config.endpoints:
                endpoints.append({
                    "name": ep.name,
                    "method": ep.method,
                    "url": ep.url,
                    "timeout_ms": ep.timeout_ms,
                    "optional": ep.optional,
                    "payload": ep.payload,
                })
        else:
            # Default: try a GET on the endpoint root; fallback to a minimal
            # POST on the declared endpoint itself.
            base_url = self.endpoint.rsplit("/", 1)[0] or self.endpoint
            endpoints = [
                {"name": "endpoint_root", "method": "GET", "url": base_url, "timeout_ms": 5000, "optional": True},
                {"name": "endpoint_post", "method": "POST", "url": self.endpoint, "timeout_ms": 8000, "optional": True},
            ]

        results: list[dict[str, Any]] = []
        overall_ok = True
        failure_reasons: list[str] = []
        warnings: list[str] = []

        for ep in endpoints:
            name = ep.get("name", "unknown")
            method = ep.get("method", "GET").upper()
            url = ep.get("url", "")
            timeout_ms = int(ep.get("timeout_ms", 5000))
            optional = bool(ep.get("optional", True))
            payload = ep.get("payload") or {}

            result: dict[str, Any] = {
                "name": name,
                "url": url,
                "method": method,
                "optional": optional,
            }

            if not url:
                result["status"] = "skipped"
                result["reason"] = "no url configured"
                results.append(result)
                if not optional:
                    overall_ok = False
                    failure_reasons.append(f"{name}: missing url")
                continue

            try:
                timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000.0)
                request_payload: dict[str, Any] | None = None
                if method == "POST":
                    request_payload = dict(payload)
                    if not request_payload:
                        # Minimal OpenAI-compatible completion probe.
                        request_payload = {
                            "model": model_id or "default",
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 1,
                            "temperature": 0.0,
                        }
                    async with self._session.post(url, json=request_payload, timeout=timeout) as resp:
                        body = await resp.text()
                        result["http_status"] = resp.status
                        result["response_bytes"] = len(body)
                        if 200 <= resp.status < 400:
                            result["status"] = "ok"
                        else:
                            raise RuntimeAdapterError(f"HTTP {resp.status}", provider=self.name)
                else:
                    async with self._session.get(url, timeout=timeout) as resp:
                        body = await resp.text()
                        result["http_status"] = resp.status
                        result["response_bytes"] = len(body)
                        if 200 <= resp.status < 400:
                            result["status"] = "ok"
                        else:
                            raise RuntimeAdapterError(f"HTTP {resp.status}", provider=self.name)
            except Exception as exc:
                result["status"] = "error"
                result["error"] = str(exc)
                if optional:
                    warnings.append(f"{name}: {exc}")
                else:
                    overall_ok = False
                    failure_reasons.append(f"{name}: {exc}")

            results.append(result)

            # If a required endpoint fails, stop further probes.
            if result["status"] == "error" and not optional:
                break

        return {
            "ok": overall_ok,
            "endpoint": self.endpoint,
            "results": results,
            "warnings": warnings,
            "reason": "; ".join(failure_reasons) if failure_reasons else "all required probes passed",
        }
