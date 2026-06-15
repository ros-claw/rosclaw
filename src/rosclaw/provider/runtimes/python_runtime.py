"""Python Runtime Adapter.

For backends that are directly importable Python callables
(e.g., HuggingFace Transformers pipeline, local PyTorch model).
"""

from collections.abc import Callable
from typing import Any

from rosclaw.provider.core.errors import RuntimeAdapterError
from rosclaw.provider.runtimes.base import RuntimeAdapter


class PythonRuntime(RuntimeAdapter):
    """Direct Python function call runtime."""

    def __init__(
        self,
        name: str,
        fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        super().__init__(name)
        self._fn = fn

    def bind(self, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        """Bind a callable to this runtime."""
        self._fn = fn

    async def start(self) -> None:
        if self._fn is None:
            raise RuntimeAdapterError(
                "No callable bound to PythonRuntime",
                provider=self.name,
            )
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_started()
        if self._fn is None:
            raise RuntimeAdapterError(
                "No callable bound",
                provider=self.name,
            )
        try:
            return self._fn(payload)
        except Exception as e:
            raise RuntimeAdapterError(
                f"PythonRuntime invoke failed: {e}",
                provider=self.name,
            ) from e
