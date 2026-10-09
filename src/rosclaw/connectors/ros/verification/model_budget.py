"""Registered model-response release budgets for controlled experiments.

The operator buffers the entire response before calling finish(). A failed
decision must never reach Native or its tools. Input usage is known only after
inference: a single request can exceed the total, is charged honestly and fails.
This is not a hard prebilling cost cap, an action permit or causal evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ConfigDict, Field, StrictInt

from rosclaw.contracts.common import ContractModel


class RegisteredModelBudgetV1(ContractModel):
    SCHEMA = "rosclaw.model_response_budget.v1"
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["rosclaw.model_response_budget.v1"] = "rosclaw.model_response_budget.v1"
    provider: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=128)
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"]
    max_requests: StrictInt = Field(gt=0, le=512)
    max_output_tokens_per_request: StrictInt = Field(gt=0, le=128000)
    max_total_tokens: StrictInt = Field(gt=0, le=100000000)
    wall_budget_sec: StrictInt = Field(gt=0, le=7200)
    max_request_bytes: StrictInt = Field(default=2097152, gt=0, le=2097152)
    max_response_bytes: StrictInt = Field(default=8388608, gt=0, le=8388608)


@dataclass(frozen=True)
class PendingModelRequest:
    upstream_body: bytes
    original_request_sha256: str
    upstream_request_sha256: str
    output_cap: int


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _closed_json(raw: bytes):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate JSON key")
            result[name] = value
        return result

    return json.loads(raw, object_pairs_hook=unique)


class ModelResponseBudget:
    def __init__(self, registered: RegisteredModelBudgetV1, *, started_monotonic: float):
        # Revalidate even a model_copy(update=...) constructed by an operator.
        self.registered = RegisteredModelBudgetV1.model_validate(registered.model_dump())
        self._started = self._last_clock = self._valid_clock(started_monotonic)
        self._deadline = self._started + self.registered.wall_budget_sec
        self._pending: PendingModelRequest | None = None
        self._requests = self._input = self._output = 0
        self._halt = ""
        self._usage_complete = True
        self._records: list[dict] = []

    @staticmethod
    def _valid_clock(now):
        if type(now) not in (float, int) or not math.isfinite(now) or now < 0:
            raise ValueError("finite monotonic clock required")
        return now

    def _clock(self, now):
        now = self._valid_clock(now)
        if now < self._last_clock:
            raise ValueError("monotonic clock cannot move backwards")
        self._last_clock = now
        return now

    def admit(self, raw: bytes, *, now: float) -> PendingModelRequest:
        if self._halt or self._pending is not None:
            raise ValueError("halted budget or model request already in flight")
        if self._clock(now) >= self._deadline:
            self._halt = "WALL_BUDGET_EXCEEDED"
            raise ValueError(self._halt)
        if self._requests >= self.registered.max_requests:
            self._halt = "MODEL_REQUEST_BUDGET_EXCEEDED"
            raise ValueError(self._halt)
        remaining = self.registered.max_total_tokens - self._input - self._output
        if remaining <= 0:
            self._halt = "TOTAL_TOKEN_BUDGET_EXCEEDED"
            raise ValueError(self._halt)
        if type(raw) is not bytes or not 0 < len(raw) <= self.registered.max_request_bytes:
            raise ValueError("bounded original model request bytes required")
        body = _closed_json(raw)
        cap = min(self.registered.max_output_tokens_per_request, remaining)
        allowed = {
            "model",
            "store",
            "stream",
            "instructions",
            "input",
            "text",
            "include",
            "prompt_cache_key",
            "tool_choice",
            "parallel_tool_calls",
            "tools",
            "reasoning",
            "max_output_tokens",
        }
        if (
            type(body) is not dict
            or set(body) - allowed
            or body.get("model") != self.registered.model
            or body.get("store") is not False
            or body.get("stream") is not True
            or body.get("reasoning")
            != {"effort": self.registered.reasoning_effort, "summary": "auto"}
            or body.get("text") != {"verbosity": "low"}
            or body.get("parallel_tool_calls") is not True
            or body.get("tool_choice") != "auto"
            or body.get("include") != ["reasoning.encrypted_content"]
            or type(body.get("input")) is not list
            or not 1 <= len(body["input"]) <= 4000
            or any(
                k in body
                for k in ("temperature", "top_p", "seed", "service_tier", "max_completion_tokens")
            )
        ):
            raise ValueError("registered exact model and Native inference parameters required")
        if "tools" in body and (
            type(body["tools"]) is not list
            or len(body["tools"]) > 128
            or any(
                type(tool) is not dict or tool.get("type") != "function" for tool in body["tools"]
            )
        ):
            raise ValueError("bounded local function tool definitions required")
        if "max_output_tokens" in body and (
            type(body["max_output_tokens"]) is not int or body["max_output_tokens"] != cap
        ):
            raise ValueError("registered output cap cannot be overridden")
        body["max_output_tokens"] = cap
        upstream = json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        if len(upstream) > self.registered.max_request_bytes:
            raise ValueError("bounded forwarded request required")
        self._pending = PendingModelRequest(upstream, _sha(raw), _sha(upstream), cap)
        self._requests += 1
        return self._pending

    def finish(self, raw: bytes, *, http_status: int, now: float) -> dict:
        pending = self._pending
        if pending is None:
            raise ValueError("an admitted original model request is required")
        self._pending = None
        result: dict[str, Any] = {
            "request_index": self._requests - 1,
            "original_request_sha256": pending.original_request_sha256,
            "upstream_request_sha256": pending.upstream_request_sha256,
            "registered_output_cap": pending.output_cap,
            "response_sha256": _sha(raw) if type(raw) is bytes else None,
            "allow_response": False,
            "usage_verified": False,
            "status": "MODEL_RESPONSE_UNVERIFIABLE",
            "actual_usage": None,
        }
        try:
            observed = self._clock(now)
            if (
                type(http_status) is not int
                or http_status != 200
                or type(raw) is not bytes
                or not 0 < len(raw) <= self.registered.max_response_bytes
            ):
                raise ValueError("bounded original successful SSE response required")
            terminal = []
            for line in raw.splitlines():
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                event = _closed_json(line[6:])
                if type(event) is not dict:
                    raise ValueError("typed SSE event required")
                if event.get("type") in ("error", "response.failed"):
                    raise ValueError("failed response cannot release tool arguments")
                if event.get("type") in ("response.completed", "response.incomplete"):
                    terminal.append(event)
            if len(terminal) != 1:
                raise ValueError("exactly one authoritative model terminal event required")
            event = terminal[0]
            response = event.get("response")
            if type(response) is not dict or response.get("model") != self.registered.model:
                raise ValueError("exact actual model required")
            usage = response.get("usage")
            keys = ("input_tokens", "output_tokens", "total_tokens")
            if type(usage) is not dict or any(
                type(usage.get(k)) is not int or usage[k] < 0 for k in keys
            ):
                raise ValueError("authoritative nonnegative integral token usage required")
            if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
                raise ValueError("consistent authoritative total token usage required")
            self._input += usage["input_tokens"]
            self._output += usage["output_tokens"]
            result.update(usage_verified=True, actual_usage={k: usage[k] for k in keys})
            if self._input + self._output > self.registered.max_total_tokens:
                result["status"] = "TOTAL_TOKEN_BUDGET_EXCEEDED"
            elif observed >= self._deadline:
                result["status"] = "WALL_BUDGET_EXCEEDED"
            elif (
                type(response.get("max_output_tokens")) is not int
                or response["max_output_tokens"] != pending.output_cap
                or usage["output_tokens"] > pending.output_cap
            ):
                result["status"] = "MODEL_OUTPUT_CAP_NOT_HONORED"
            elif event["type"] == "response.incomplete":
                details = response.get("incomplete_details")
                result["status"] = (
                    "MODEL_OUTPUT_LIMIT"
                    if type(details) is dict and details.get("reason") == "max_output_tokens"
                    else "MODEL_RESPONSE_INCOMPLETE"
                )
            elif response.get("status") != "completed":
                result["status"] = "MODEL_TERMINAL_STATUS_MISMATCH"
            else:
                result.update(status="WITHIN_REGISTERED_RESPONSE_BUDGET", allow_response=True)
        except (ValueError, TypeError, KeyError, OverflowError):
            pass  # Unverifiable responses never reach Native or tool execution.
        if not result["usage_verified"]:
            self._usage_complete = False
        if not result["allow_response"]:
            self._halt = result["status"]
        self._records.append(deepcopy(result))
        return deepcopy(result)

    def snapshot(self) -> dict:
        total = self._input + self._output
        return {
            "schema_version": "rosclaw.model_response_budget_ledger.v1",
            "budget_scope": "RESPONSE_RELEASE_GATE_NOT_PREBILLING_COST_CAP",
            "registration": self.registered.model_dump(mode="json"),
            "requests_admitted": self._requests,
            "request_in_flight": self._pending is not None,
            "actual_input_tokens": self._input,
            "actual_output_tokens": self._output,
            "actual_total_tokens": total,
            "usage_complete": self._usage_complete and self._pending is None,
            "overshoot_tokens": max(0, total - self.registered.max_total_tokens),
            "halt_reason": self._halt,
            "records": deepcopy(self._records),
            "robot_authorization": False,
            "causal_benefit": "NOT_MEASURED",
        }
