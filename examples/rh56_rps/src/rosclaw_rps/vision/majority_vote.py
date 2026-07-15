"""Temporal majority voting for human gesture recognition."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List

from ..gesture_schema import GesturePrediction


@dataclass
class MajorityVoteBuffer:
    """Collect predictions over a window and return the most confident label.

    ``tail_size`` > 0 restricts the vote to the most recent predictions, so a
    gesture that has just settled dominates the decision instead of being
    diluted by transitional frames captured while the hand was still moving.
    """

    window_size: int = 20
    min_confidence: float = 0.65
    majority_ratio: float = 0.6
    tail_size: int = 0
    _buffer: List[GesturePrediction] = field(default_factory=list, repr=False)

    def update(self, pred: GesturePrediction) -> None:
        self._buffer.append(pred)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

    def reset(self) -> None:
        self._buffer.clear()

    def final(self) -> GesturePrediction:
        if not self._buffer:
            return GesturePrediction(label="unknown", confidence=0.0)

        entries = (
            self._buffer[-self.tail_size :] if self.tail_size > 0 else self._buffer
        )
        valid = [
            p
            for p in entries
            if p.label != "unknown" and p.confidence >= self.min_confidence
        ]
        if not valid:
            return GesturePrediction(label="unknown", confidence=0.0)

        labels = [p.label for p in valid]
        counter = Counter(labels)
        top_label, top_count = counter.most_common(1)[0]
        ratio = top_count / len(valid)
        if ratio < self.majority_ratio:
            return GesturePrediction(label="unknown", confidence=0.0)

        avg_confidence = (
            sum(p.confidence for p in valid if p.label == top_label) / top_count
        )
        return GesturePrediction(label=top_label, confidence=round(avg_confidence, 3))
