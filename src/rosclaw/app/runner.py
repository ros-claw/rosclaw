"""Execute an App exclusively through rosclawd ActionEnvelope calls."""

from __future__ import annotations

import contextlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from rosclaw.app.schema import AppManifest
from rosclaw.kernel import (
    ActionEnvelope,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)

APP_RUN_SCHEMA_VERSION = "rosclaw.app.run.v1"
_TEMPLATE_RE = re.compile(r"\$\{([a-z][a-z0-9_.]*)\}")


class AppClient(Protocol):
    def create_session(
        self,
        *,
        session_id: str,
        actor_id: str,
        agent_framework: str,
        body_scope: list[str],
        capability_scope: list[str],
        ttl_ms: int = 30_000,
    ) -> dict[str, Any]: ...
    def request_action(self, action: ActionEnvelope) -> dict[str, Any]: ...
    def wait_for_action(self, action_id: str, *, timeout_sec: float) -> dict[str, Any]: ...
    def close_session(self, session_id: str, *, reason: str) -> dict[str, Any]: ...


@dataclass
class AppRunResult:
    app: str
    status: str
    session_id: str
    execution_mode: ExecutionMode
    receipts: list[dict[str, Any]] = field(default_factory=list)
    verification: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, str] | None = None
    schema_version: str = APP_RUN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "app": self.app,
            "status": self.status,
            "session_id": self.session_id,
            "execution_mode": self.execution_mode.value,
            "trust_level": _run_trust(self.execution_mode, self.receipts),
            "receipts": self.receipts,
            "verification": self.verification,
            "error": self.error,
        }


class AppRunner:
    def __init__(self, client: AppClient) -> None:
        self.client = client

    def run(
        self,
        manifest: AppManifest,
        *,
        body_id: str,
        body_snapshot_hash: str,
        execution_mode: ExecutionMode,
        principal_id: str = "",
        permits: dict[str, str] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> AppRunResult:
        mode = ExecutionMode(execution_mode)
        if mode in {ExecutionMode.SHADOW, ExecutionMode.REAL} and not body_snapshot_hash:
            raise ValueError("Shadow and REAL App runs require body_snapshot_hash")
        session_id = f"app_{manifest.metadata.name}_{uuid.uuid4().hex[:16]}"
        result = AppRunResult(
            app=manifest.metadata.name,
            status="running",
            session_id=session_id,
            execution_mode=mode,
        )
        context: dict[str, Any] = {"input": dict(inputs or {})}
        permits = permits or {}
        max_timeout = max(step.timeout_sec for step in manifest.workflow)
        self.client.create_session(
            session_id=session_id,
            actor_id=f"rosclaw-app:{manifest.metadata.name}",
            agent_framework="rosclaw-app",
            body_scope=[body_id],
            capability_scope=manifest.requires.capabilities,
            ttl_ms=min(3_600_000, max(10_000, int(max_timeout * 1000) + 1_000)),
        )
        try:
            for index, step in enumerate(manifest.workflow):
                arguments = _resolve_templates(step.input, context)
                permit_id = permits.get(step.call)
                authorization = AuthorizationContext(
                    principal_id=principal_id,
                    approved=permit_id is not None,
                    approval_id=permit_id,
                    scopes=[step.call] if permit_id is not None else [],
                )
                action = ActionEnvelope(
                    actor_id=f"rosclaw-app:{manifest.metadata.name}",
                    agent_framework="rosclaw-app",
                    session_id=session_id,
                    body_id=body_id,
                    body_snapshot_hash=body_snapshot_hash,
                    capability_id=step.call,
                    arguments=arguments,
                    execution_mode=mode,
                    authorization=authorization,
                    deadline_at=datetime.now(UTC) + timedelta(seconds=step.timeout_sec),
                    lease_ttl_ms=max(1_000, min(10_000, int(step.timeout_sec * 1000))),
                    renew_interval_ms=max(
                        100,
                        min(3_000, int(step.timeout_sec * 1000 / 3)),
                    ),
                    verification_policy=VerificationPolicy(
                        required_evidence=(
                            EvidenceLevel.PHYSICALLY_OBSERVED
                            if mode is ExecutionMode.REAL
                            else EvidenceLevel.REQUESTED
                        ),
                        timeout_sec=step.timeout_sec,
                    ),
                )
                ticket = self.client.request_action(action)
                status = self.client.wait_for_action(
                    str(ticket["action_id"]),
                    timeout_sec=step.timeout_sec + 1.0,
                )
                receipt = status.get("receipt")
                if not isinstance(receipt, dict):
                    raise RuntimeError(f"App step {index} returned no ExecutionReceipt")
                result.receipts.append(receipt)
                if step.save_as:
                    context[step.save_as] = receipt
                if receipt.get("final_state") not in {"COMPLETED", "DEGRADED"}:
                    result.status = "failed"
                    result.error = {
                        "code": "APP_STEP_FAILED",
                        "message": f"Capability {step.call!r} did not complete",
                    }
                    return result
            result.verification = [
                _verify_rule(rule, context) for rule in manifest.verification.require
            ]
            failed = [item for item in result.verification if item["passed"] is not True]
            result.status = "failed" if failed else "success"
            if failed:
                result.error = {
                    "code": "APP_VERIFICATION_FAILED",
                    "message": f"{len(failed)} App verification rule(s) failed",
                }
            return result
        except Exception as exc:  # noqa: BLE001 - daemon failures become App result
            result.status = "failed"
            result.error = {
                "code": "APP_RUN_FAILED",
                "message": f"{type(exc).__name__}: {exc}"[:1024],
            }
            return result
        finally:
            with contextlib.suppress(Exception):
                self.client.close_session(session_id, reason="app_run_finished")


def _resolve_templates(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_templates(child, context) for key, child in value.items()}
    if isinstance(value, list):
        return [_resolve_templates(child, context) for child in value]
    if not isinstance(value, str):
        return value
    exact = _TEMPLATE_RE.fullmatch(value)
    if exact:
        return _resolve_path(exact.group(1), context)

    def replace(match: re.Match[str]) -> str:
        resolved = _resolve_path(match.group(1), context)
        if isinstance(resolved, (dict, list)):
            return json.dumps(resolved, sort_keys=True, ensure_ascii=False)
        return str(resolved)

    return _TEMPLATE_RE.sub(replace, value)


def _resolve_path(path: str, context: dict[str, Any]) -> Any:
    current: Any = context
    for component in path.split("."):
        if isinstance(current, dict) and component in current:
            current = current[component]
        elif isinstance(current, list) and component.isdigit() and int(component) < len(current):
            current = current[int(component)]
        else:
            raise KeyError(f"App context path does not exist: {path}")
    return current


def _verify_rule(rule: str, context: dict[str, Any]) -> dict[str, Any]:
    if rule.endswith(".exists"):
        path = rule.removesuffix(".exists")
        try:
            value = _resolve_path(path, context)
            passed = value is not None and value != "" and value != [] and value != {}
        except KeyError:
            passed = False
        return {"rule": rule, "passed": passed}
    if "==" in rule:
        path, expected = (part.strip() for part in rule.split("==", 1))
        try:
            actual = _resolve_path(path, context)
        except KeyError:
            actual = None
        return {"rule": rule, "passed": str(actual) == expected, "actual": actual}
    return {"rule": rule, "passed": False, "error": "unsupported verification rule"}


def _run_trust(mode: ExecutionMode, receipts: list[dict[str, Any]]) -> str:
    if not receipts:
        return "UNAVAILABLE"
    levels = {str(receipt.get("trust_level", "UNAVAILABLE")) for receipt in receipts}
    if mode is ExecutionMode.REAL and levels == {"VERIFIED"}:
        return "VERIFIED"
    if mode is ExecutionMode.SIMULATION:
        return "SIMULATED"
    if mode is ExecutionMode.FIXTURE:
        return "SYNTHETIC"
    return "UNVERIFIED"


__all__ = ["APP_RUN_SCHEMA_VERSION", "AppClient", "AppRunResult", "AppRunner"]
