"""GenericProvider - Backend-agnostic provider backed by a RuntimeAdapter.

Given a ProviderManifest, GenericProvider:
1. Creates the appropriate RuntimeAdapter (http, python, ros2)
2. Bridges manifest-declared capabilities to runtime.invoke()
3. Handles load/unload lifecycle

This allows declarative provider registration via provider.yaml
without writing custom Provider subclasses for every backend.
"""

from typing import Any

from rosclaw.provider.core.errors import RuntimeAdapterError
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.provider import Provider
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.runtimes.base import RuntimeAdapter
from rosclaw.provider.runtimes.http_runtime import HTTPRuntime
from rosclaw.provider.runtimes.python_runtime import PythonRuntime
from rosclaw.provider.runtimes.ros2_runtime import ROS2Runtime


class GenericProvider(Provider):
    """Generic provider that delegates inference to a RuntimeAdapter.

    The RuntimeAdapter is selected from manifest.runtime.backend:
        - http   -> HTTPRuntime
        - python -> PythonRuntime
        - ros2   -> ROS2Runtime
    """

    def __init__(self, manifest: ProviderManifest):
        super().__init__(manifest)
        self._runtime: RuntimeAdapter | None = None
        self._create_runtime()

    def _create_runtime(self) -> None:
        """Instantiate the correct RuntimeAdapter from manifest."""
        runtime_spec = self.manifest.runtime
        backend = runtime_spec.backend
        name = self.manifest.name

        if backend == "http":
            self._runtime = HTTPRuntime(
                name=name,
                endpoint=runtime_spec.endpoint,
                timeout_sec=float(runtime_spec.env.get("timeout_sec", 30.0)),
                retries=int(runtime_spec.env.get("retries", 1)),
                headers=self._parse_headers(runtime_spec.env.get("headers", "")),
            )
        elif backend == "python":
            self._runtime = PythonRuntime(name=name)
            # PythonRuntime expects a callable to be bound separately
            # (e.g., by a custom loader or user code)
        elif backend == "ros2":
            self._runtime = ROS2Runtime(
                name=name,
                action_name=runtime_spec.env.get("action_name", ""),
                service_name=runtime_spec.env.get("service_name", ""),
                timeout_sec=float(runtime_spec.env.get("timeout_sec", 30.0)),
            )
        elif backend:
            raise RuntimeAdapterError(
                f"Unsupported backend: {backend}",
                provider=name,
            )
        else:
            # No runtime declared; this provider may be purely metadata
            # (e.g., a wrapper handled by custom code)
            pass

    @staticmethod
    def _parse_headers(headers_str: str) -> dict[str, str]:
        """Parse 'Key: Value\nKey2: Value2' into dict."""
        result: dict[str, str] = {}
        for line in headers_str.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                result[k.strip()] = v.strip()
        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def load(self) -> None:
        if self._runtime is not None:
            await self._runtime.start()
        self._healthy = True

    async def unload(self) -> None:
        if self._runtime is not None:
            await self._runtime.stop()
        self._healthy = False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        self._ensure_capability_supported(request.capability)

        if self._runtime is None:
            raise RuntimeAdapterError(
                "No runtime adapter configured for this provider",
                provider=self.name,
            )

        payload = {
            "capability": request.capability,
            "inputs": request.inputs,
            "context": request.context,
            "constraints": request.constraints,
        }

        try:
            raw = await self._runtime.invoke(payload)
        except RuntimeAdapterError:
            raise
        except Exception as e:
            raise RuntimeAdapterError(
                f"Runtime invoke failed: {e}",
                provider=self.name,
            ) from e

        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result=raw.get("result", raw),
            confidence=raw.get("confidence"),
            status=raw.get("status", "ok"),
            warnings=raw.get("warnings", []),
            errors=raw.get("errors", []),
        )

    async def health(self) -> dict[str, Any]:
        base = await super().health()
        if self._runtime is not None:
            base["runtime_started"] = self._runtime._started
        return base
