"""Structured model-facing results for observation tools.

Text remains the protocol response for the originating tool call.  Optional
images are ephemeral evidence for a VLM: callers must not journal their raw
base64 payloads or treat pixels as operator authorization.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolImage:
    """One bounded, validated image returned by an observation tool."""

    mime_type: str
    data_base64: str


@dataclass(frozen=True)
class ToolExecutionResult:
    """A textual tool response plus optional ephemeral image evidence."""

    text: str
    images: tuple[ToolImage, ...] = ()
