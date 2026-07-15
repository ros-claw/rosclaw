"""Trace context propagation across sync and async ROSClaw call paths."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True)
class TraceContext:
    """The causal identity of the operation currently executing."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None


_CURRENT_TRACE_CONTEXT: ContextVar[TraceContext | None] = ContextVar(
    "rosclaw_trace_context", default=None
)


def current_trace_context() -> TraceContext | None:
    """Return the current trace context, if one is active."""

    return _CURRENT_TRACE_CONTEXT.get()


def attach_trace_context(context: TraceContext) -> Token[TraceContext | None]:
    """Make ``context`` current and return a token used to restore the parent."""

    return _CURRENT_TRACE_CONTEXT.set(context)


def detach_trace_context(token: Token[TraceContext | None]) -> None:
    """Restore the context that preceded ``attach_trace_context``."""

    _CURRENT_TRACE_CONTEXT.reset(token)
