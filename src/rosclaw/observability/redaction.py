"""Bounded, secret-aware trace payload capture."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from rosclaw.observability.schema import CaptureMode

_SECRET_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|password|passwd|secret|authorization|access[_-]?token|"
    r"refresh[_-]?token|bearer[_-]?token|private[_-]?key)($|[_-])",
    re.IGNORECASE,
)
_BEARER_VALUE = re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+")
_PRIVATE_REASONING_KEY = re.compile(
    r"^(cot|chain[_-]?of[_-]?thought|reasoning|reasoning[_-]?trace|internal[_-]?reasoning|"
    r"hidden[_-]?reasoning)$",
    re.IGNORECASE,
)


class TraceRedactor:
    """Convert arbitrary runtime values into safe, bounded JSON-like values."""

    def __init__(
        self,
        mode: CaptureMode | str = CaptureMode.STANDARD,
        max_text_chars: int = 16_384,
        max_collection_items: int = 256,
    ) -> None:
        self.mode = CaptureMode(mode)
        self.max_text_chars = max_text_chars
        self.max_collection_items = max_collection_items

    def redact(self, value: Any) -> Any:
        """Redact secrets and replace large/binary values with stable references."""

        try:
            if self.mode == CaptureMode.MINIMAL:
                return self._summarize(value)
            return self._convert(value, depth=0)
        except Exception as exc:  # noqa: BLE001
            return {
                "type": type(value).__name__,
                "capture_error": type(exc).__name__,
            }

    def _convert(self, value: Any, depth: int) -> Any:
        if depth > 20:
            return {"type": type(value).__name__, "truncated": "max_depth"}
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            safe = _BEARER_VALUE.sub("Bearer [REDACTED]", value)
            if len(safe) <= self.max_text_chars:
                return safe
            return {
                "text": safe[: self.max_text_chars],
                "truncated": True,
                "original_chars": len(safe),
                "sha256": hashlib.sha256(safe.encode("utf-8")).hexdigest(),
            }
        if isinstance(value, (bytes, bytearray, memoryview)):
            data = bytes(value)
            return {
                "artifact": "inline-binary-omitted",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value) and not isinstance(value, type):
            return self._convert(asdict(value), depth + 1)
        if isinstance(value, dict):
            converted: dict[str, Any] = {}
            items = list(value.items())
            for key, item in items[: self.max_collection_items]:
                key_text = str(key)
                if _SECRET_KEY.search(key_text):
                    converted[key_text] = "[REDACTED]"
                elif self.mode != CaptureMode.RESEARCH and _PRIVATE_REASONING_KEY.match(key_text):
                    converted[key_text] = "[PRIVATE_REASONING_OMITTED]"
                else:
                    converted[key_text] = self._convert(item, depth + 1)
            if len(items) > self.max_collection_items:
                converted["_trace_truncated_items"] = len(items) - self.max_collection_items
            return converted
        if isinstance(value, (list, tuple, set, frozenset)):
            items = list(value)
            converted = [
                self._convert(item, depth + 1) for item in items[: self.max_collection_items]
            ]
            if len(items) > self.max_collection_items:
                converted.append({"truncated_items": len(items) - self.max_collection_items})
            return converted

        # NumPy arrays, tensors, images, and point clouds remain artifact metadata.
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None:
            return {
                "artifact": "array-omitted",
                "type": type(value).__name__,
                "shape": list(shape),
                "dtype": str(dtype) if dtype is not None else None,
            }
        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return self._convert(value.to_dict(), depth + 1)
            except Exception:
                pass
        return {"type": type(value).__name__, "repr": repr(value)[:512]}

    def _summarize(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return {"type": "str", "chars": len(value)}
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {"type": "binary", "bytes": len(value)}
        if isinstance(value, dict):
            return {"type": "object", "keys": [str(k) for k in list(value)[:32]]}
        if isinstance(value, (list, tuple, set, frozenset)):
            return {"type": type(value).__name__, "items": len(value)}
        shape = getattr(value, "shape", None)
        if shape is not None:
            return {"type": type(value).__name__, "shape": list(shape)}
        return {"type": type(value).__name__}
