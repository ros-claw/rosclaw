"""Stream display filter: hide DecisionV1 blocks from streamed output.

The DecisionV1 JSON block is machine protocol, not user prose. Deltas are
held in a small lookahead buffer; once a code fence followed by the
decision marker is detected, the rest of that block is suppressed. All
other content (including ordinary code fences) passes through.
"""

from __future__ import annotations

from collections.abc import Callable

_MARKER = '"schema_version": "rosclaw.decision.v1"'
_FENCE = "```"
#: max chars after a fence that may still turn into a decision block
_LOOKAHEAD = len(_MARKER) + 16


class DecisionBlockFilter:
    def __init__(self, sink: Callable[[str], None]) -> None:
        self._sink = sink
        self._pending = ""
        self._suppressing = False

    def feed(self, piece: str) -> None:
        if self._suppressing:
            return
        self._pending += piece
        self._drain(final=False)

    def flush(self) -> None:
        """Turn end: emit whatever was held back and not suppressed."""
        if self._suppressing:
            self._pending = ""
            self._suppressing = False
            return
        # A held-back tail might itself contain a decision block; strip it.
        text = self._pending
        self._pending = ""
        fence = text.find(_FENCE)
        if fence != -1 and _MARKER in text[fence:]:
            text = text[:fence]
        if text:
            self._sink(text)

    def _drain(self, *, final: bool) -> None:
        while not self._suppressing:
            fence = self._pending.find(_FENCE)
            if fence == -1:
                # Keep a few chars back: a fence may straddle two deltas.
                keep = len(_FENCE) - 1
                emit = self._pending[:-keep] if len(self._pending) > keep else ""
                self._pending = self._pending[-keep:] if emit else self._pending
                if emit:
                    self._sink(emit)
                return
            before = self._pending[:fence]
            if before:
                self._sink(before)
            self._pending = self._pending[fence:]
            tail = self._pending
            if _MARKER in tail:
                self._suppressing = True
                self._pending = ""
                return
            if len(tail) > _LOOKAHEAD:
                # Definitely not a decision block: emit the fence region.
                self._sink(tail[: _LOOKAHEAD + 1])
                self._pending = tail[_LOOKAHEAD + 1 :]
                continue
            return  # wait for more deltas to decide
