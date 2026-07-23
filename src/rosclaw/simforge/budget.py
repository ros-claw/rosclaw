"""Resource limits for evidence records and episode/run storage."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class BudgetScope(StrEnum):
    EVENT = "event"
    TRACE = "trace"
    EPISODE_SEMANTIC = "episode_semantic"
    EPISODE_RAW = "episode_raw"
    RUN = "run"
    WORKSPACE = "workspace"


class BudgetAction(StrEnum):
    ACCEPT = "accept"
    SUMMARIZE = "summarize"
    STOP_RECORDING = "stop_recording"
    ABORT_FAIL_CLOSED = "abort_fail_closed"


@dataclass(frozen=True)
class DataBudgetSpec:
    max_event_record_bytes: int = 1_048_576
    max_trace_record_bytes: int = 1_048_576
    max_nested_depth: int = 8
    max_string_length: int = 4096
    max_collection_items: int = 100_000
    max_episode_semantic_bytes: int = 67_108_864
    max_episode_raw_bytes: int = 2_147_483_648
    max_run_bytes: int = 10_737_418_240
    max_workspace_bytes: int = 214_748_364_800

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values
        ):
            raise ValueError("all data budget limits must be positive integers")
        if self.max_nested_depth > 128:
            raise ValueError("max_nested_depth cannot exceed 128")


@dataclass(frozen=True)
class BudgetDecision:
    accepted: bool
    action: BudgetAction
    reason: str
    estimated_bytes: int


class DataBudgetManager:
    """Bounded preflight accounting that avoids serializing attacker-sized data."""

    def __init__(self, spec: DataBudgetSpec | None = None) -> None:
        self.spec = spec or DataBudgetSpec()
        self._usage = dict.fromkeys(BudgetScope, 0)

    def inspect_record(self, value: Any, *, scope: BudgetScope) -> BudgetDecision:
        record_limit = self._limit(scope)
        try:
            estimated = self._bounded_size(value, record_limit)
        except _BudgetViolationError as violation:
            return BudgetDecision(
                accepted=False,
                action=self._overflow_action(scope),
                reason=str(violation),
                estimated_bytes=violation.estimated_bytes,
            )
        except (TypeError, ValueError, OverflowError, UnicodeError) as error:
            return BudgetDecision(
                accepted=False,
                action=self._overflow_action(scope),
                reason=f"record cannot be measured safely: {type(error).__name__}",
                estimated_bytes=0,
            )
        projected = self._usage[scope] + estimated
        if projected > record_limit:
            return BudgetDecision(
                accepted=False,
                action=self._overflow_action(scope),
                reason=f"{scope.value} budget would exceed {record_limit} bytes",
                estimated_bytes=estimated,
            )
        return BudgetDecision(True, BudgetAction.ACCEPT, "within_budget", estimated)

    def commit(self, decision: BudgetDecision, *, scope: BudgetScope) -> None:
        if not decision.accepted:
            raise ValueError("refusing to commit a rejected budget decision")
        projected = self._usage[scope] + decision.estimated_bytes
        if projected > self._limit(scope):
            raise RuntimeError("data budget changed between inspection and commit")
        self._usage[scope] = projected

    def account_external_bytes(self, *, scope: BudgetScope, size: int) -> BudgetDecision:
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("external byte size must be a non-negative integer")
        limit = self._limit(scope)
        projected = self._usage[scope] + size
        accepted = projected <= limit
        return BudgetDecision(
            accepted=accepted,
            action=BudgetAction.ACCEPT if accepted else self._overflow_action(scope),
            reason="within_budget" if accepted else f"{scope.value} budget exceeded",
            estimated_bytes=size,
        )

    def usage(self) -> dict[str, int]:
        return {scope.value: value for scope, value in self._usage.items()}

    def _bounded_size(self, root: Any, byte_limit: int) -> int:
        total = 0
        item_count = 0
        stack: list[tuple[Any, int]] = [(root, 0)]
        seen: set[int] = set()
        while stack:
            value, depth = stack.pop()
            item_count += 1
            if item_count > self.spec.max_collection_items:
                raise _BudgetViolationError("collection item budget exceeded", total)
            if depth > self.spec.max_nested_depth:
                raise _BudgetViolationError("nested depth budget exceeded", total)
            if isinstance(value, str):
                length = len(value.encode("utf-8"))
                if length > self.spec.max_string_length:
                    raise _BudgetViolationError("string length budget exceeded", total + length)
                total += length + 2
            elif isinstance(value, float) and not math.isfinite(value):
                raise _BudgetViolationError("non-finite numeric value", total)
            elif value is None or isinstance(value, (bool, int, float)):
                total += len(json.dumps(value, allow_nan=False))
            elif isinstance(value, dict):
                identity = id(value)
                if identity in seen:
                    raise _BudgetViolationError("recursive record detected", total)
                seen.add(identity)
                total += 2
                for key, child in value.items():
                    if not isinstance(key, str):
                        raise _BudgetViolationError(
                            "record mapping keys must be strings",
                            total,
                        )
                    stack.append((child, depth + 1))
                    stack.append((key, depth + 1))
            elif isinstance(value, (list, tuple)):
                identity = id(value)
                if identity in seen:
                    raise _BudgetViolationError("recursive record detected", total)
                seen.add(identity)
                total += 2
                for child in value:
                    stack.append((child, depth + 1))
            else:
                raise _BudgetViolationError(
                    f"unsupported record type: {type(value).__name__}", total
                )
            if total > byte_limit:
                raise _BudgetViolationError(f"record exceeds {byte_limit} bytes", total)
        return total

    def _limit(self, scope: BudgetScope) -> int:
        return {
            BudgetScope.EVENT: self.spec.max_event_record_bytes,
            BudgetScope.TRACE: self.spec.max_trace_record_bytes,
            BudgetScope.EPISODE_SEMANTIC: self.spec.max_episode_semantic_bytes,
            BudgetScope.EPISODE_RAW: self.spec.max_episode_raw_bytes,
            BudgetScope.RUN: self.spec.max_run_bytes,
            BudgetScope.WORKSPACE: self.spec.max_workspace_bytes,
        }[scope]

    @staticmethod
    def _overflow_action(scope: BudgetScope) -> BudgetAction:
        if scope in {BudgetScope.EVENT, BudgetScope.TRACE, BudgetScope.EPISODE_SEMANTIC}:
            return BudgetAction.SUMMARIZE
        if scope is BudgetScope.EPISODE_RAW:
            return BudgetAction.STOP_RECORDING
        return BudgetAction.ABORT_FAIL_CLOSED


class _BudgetViolationError(RuntimeError):
    def __init__(self, message: str, estimated_bytes: int) -> None:
        super().__init__(message)
        self.estimated_bytes = estimated_bytes


__all__ = [
    "BudgetAction",
    "BudgetDecision",
    "BudgetScope",
    "DataBudgetManager",
    "DataBudgetSpec",
]
