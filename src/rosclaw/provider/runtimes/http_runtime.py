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
        super().__init__(name, config={"endpoint": endpoint, "timeout": timeout_sec})
        self.endpoint = endpoint.rstrip("/")
        self.timeout_sec = timeout_sec
        self.retries = retries
        self.headers = headers or {}
        self._session = None

    async def start(self) -> None:
        try:
            import aiohttp
        except ImportError:
            raise RuntimeError("aiohttp is required for HTTPRuntime. pip install aiohttp")
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
