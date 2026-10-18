"""Bounded new B/C/E episode client; frozen G5 and A/D runners stay unchanged.

The host injects an already gated RskPhysicalACI. This module opens no Runtime,
ROS transport or hardware. Contract clients cannot invoke SDK/probe callbacks or
produce model/physical evidence. Actual runs require frozen original inputs,
source, client implementation and executable before the first provider call.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

import rosclaw
from benchmarks.ros_expert_ablation import host_client, isolated_client
from benchmarks.ros_expert_ablation.aci_context import EngineeringEpisodeContext, encoded
from benchmarks.ros_expert_ablation.aci_runner import EpisodeWorkbook, model_safe
from benchmarks.ros_expert_ablation.protocol import usage_from_events
from benchmarks.ros_expert_ablation.rsk_physical_aci import RskPhysicalACI
from benchmarks.ros_expert_ablation.rsk_pilot_protocol import (
    RskPilotProtocolV2,
    allocate_pilot,
    review_pilot_entry,
)
from benchmarks.ros_expert_ablation.rsk_tool_guide import build_rsk_tool_guide
from rosclaw.contracts.common import content_hash

CLIENT_BINARY = Path(
    "/home/nvidia/.codex/packages/standalone/releases/0.162.1-aarch64-unknown-linux-musl/bin/codex"
)
INSTRUCTIONS = (
    "Solve the supplied SIM ROS engineering task with one schema-valid decision per turn. "
    "Use only the offered external operations; no native tools, shell, ROS, hardware or web. "
    "Read original facts to distinguish competing hypotheses. UNKNOWN is not permission. "
    "Staging options and historical concepts are unexecuted conditional hypotheses. "
    "Recompute current parameters; cite only Memory references actually returned in context. "
    "Explicitly validate a supported proposal in the operator-owned new SDK World, or abstain. "
    "Never alter Body, geometry, speeds, leases, source clocks, freshness or permissions. "
    "Finish reports only; physical success requires independent canonical verification. "
    "Give a short evidence-based public decision summary, not private reasoning."
)


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class OriginalRef(Closed):
    path: str = Field(min_length=1, max_length=4096)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EpisodeFactsV1(Closed):
    schema_version: Literal["rosclaw.rsk_episode_facts.v1"] = "rosclaw.rsk_episode_facts.v1"
    case: OriginalRef
    model: OriginalRef
    requirements: OriginalRef
    staging: OriginalRef
    task: OriginalRef
    evaluation_episode_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_settings: OriginalRef


class FrozenProviderV1(Closed):
    schema_version: Literal["rosclaw.rsk_model_provider.v1"] = "rosclaw.rsk_model_provider.v1"
    execution_kind: Literal["ACTUAL_MODEL_CLIENT", "SYNTHETIC_CONTRACT_CLIENT"]
    model_identity: str = Field(min_length=1, max_length=128)
    effort: str = Field(min_length=1, max_length=32)
    client_code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    client_settings_code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _read(root: Path, reference: OriginalRef, *, limit=8_000_000):
    relative = Path(reference.path)
    if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
        raise ValueError("relative original reference within explicit evidence root required")
    path = root / relative
    resolved = path.resolve(strict=True)
    resolved.relative_to(root)
    if path.is_symlink():
        raise ValueError("original reference cannot be a symlink")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError("bounded original regular file required")
        raw = os.read(fd, limit + 1)
        after = os.fstat(fd)
        if (
            len(raw) != before.st_size
            or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or hashlib.sha256(raw).hexdigest() != reference.sha256
        ):
            raise ValueError("original input changed or SHA mismatch")
        return raw
    finally:
        os.close(fd)


def _provider_pin(provider, protocol, *, contract_client):
    if provider.model_identity != protocol.model_identity:
        raise ValueError("provider identity differs from frozen protocol")
    if provider.execution_kind == "SYNTHETIC_CONTRACT_CLIENT":
        if contract_client is None:
            raise ValueError("explicit synthetic contract client required")
        return contract_client
    if contract_client is not None:
        raise ValueError("synthetic callback cannot impersonate an actual model client")
    for module, expected in (
        (host_client, provider.client_code_sha256),
        (isolated_client, provider.client_settings_code_sha256),
    ):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen model client implementation changed")
    if (
        provider.model_identity != isolated_client.MODEL
        or provider.effort != isolated_client.EFFORT
        or hashlib.sha256(CLIENT_BINARY.read_bytes()).hexdigest() != provider.executable_sha256
    ):
        raise ValueError("actual model settings or executable differ from frozen provider")
    repo = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if (
        head != protocol.harness_source
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip()
    ):
        raise ValueError("actual episode requires the exact clean frozen harness source")
    for module in (host_client, isolated_client):
        Path(module.__file__).resolve().relative_to(repo)
    Path(rosclaw.__file__).resolve().relative_to(repo / "src")
    return host_client.invoke


def run_rsk_typed_episode(
    task,
    aci: RskPhysicalACI,
    output,
    *,
    protocol: RskPilotProtocolV2,
    pilot_evidence,
    evidence_root,
    ordinal,
    facts_manifest: OriginalRef,
    evaluation_episode: OriginalRef,
    provider_settings: OriginalRef,
    memory_policy: OriginalRef,
    contract_client=None,
    clock=time.monotonic,
):
    """One registered B/C/E allocation, max300s/12 decisions, never a pilot launcher."""
    if not isinstance(aci, RskPhysicalACI) or type(ordinal) is not int:
        raise ValueError("typed RSK interface and registered allocation ordinal required")
    root = Path(evidence_root).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    source_clock = clock
    last_clock = None

    def measured_clock():
        nonlocal last_clock
        value = source_clock()
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (last_clock is not None and value < last_clock)
        ):
            raise ValueError("finite nondecreasing host monotonic clock required")
        last_clock = value
        return value

    clock = measured_clock
    started = clock()
    if not math.isfinite(started):
        raise ValueError("finite monotonic episode clock required")
    base = aci.base
    entry = review_pilot_entry(protocol, pilot_evidence, root=root)
    result = {
        "schema_version": "rosclaw.rsk_typed_episode.v1",
        "status": "NO_GO",
        "protocol_hash": protocol.protocol_hash,
        "harness_source": protocol.harness_source,
        "entry_review": entry,
        "actual_model_calls": 0,
        "actual_model_client_invocations": 0,
        "synthetic_contract_calls": 0,
        "new_Worlds_requested": 0,
        "verified_success": False,
        "current_authority": "NONE",
        "causal_Memory_benefit": "NOT_ESTABLISHED",
        "old_G5": "UNCHANGED",
    }
    calls, tools = [], []
    usage: dict[str, int] = {}
    reported_usage: dict[str, int] = {}
    provider = None
    try:
        if entry["status"] != "ELIGIBLE_FOR_BOUNDED_PILOT_ONLY":
            raise ValueError("original pilot entry is NO_GO; no model or SDK calls")
        allocation = next(row for row in allocate_pilot(protocol) if row["ordinal"] == ordinal)
        if allocation["arm"] != aci.arm:
            raise ValueError("registered allocation arm differs from the injected interface")
        case = next(row for row in protocol.cases if row.case_id == allocation["case_id"])
        if (
            facts_manifest.sha256 != case.facts_manifest_sha256
            or evaluation_episode.sha256 != case.evaluation_episode_sha256
            or provider_settings.sha256 != protocol.provider_settings_sha256
            or memory_policy.sha256 != protocol.memory_policy_sha256
        ):
            raise ValueError("original allocation/facts/episode/provider hashes differ")
        facts = EpisodeFactsV1.model_validate_json(_read(root, facts_manifest))
        _read(root, evaluation_episode)
        if facts.evaluation_episode_sha256 != evaluation_episode.sha256:
            raise ValueError("registered facts do not bind the original evaluation episode")
        for ref, value in (
            (facts.case, base.case.model_dump(mode="json")),
            (facts.model, base.model.to_dict()),
            (facts.requirements, base.requirements),
            (facts.staging, base.tools.initial_staging),
            (facts.task, task),
        ):
            if json.loads(_read(root, ref)) != value:
                raise ValueError("injected interface differs from registered original facts")
        policy = json.loads(_read(root, memory_policy))
        if (
            type(policy) is not dict
            or set(policy) != {"schema_version", "cases"}
            or policy["schema_version"] != "rosclaw.rsk_memory_policy_manifest.v1"
            or type(policy["cases"]) is not dict
            or set(policy["cases"]) != {row.case_id for row in protocol.cases}
        ):
            raise ValueError("closed frozen per-case Memory policy manifest required")
        run_id = json.loads(_read(root, facts.run_settings)).get("run_id")
        if (
            type(run_id) is not str
            or len(run_id) != 32
            or any(c not in "0123456789abcdef" for c in run_id)
        ):
            raise ValueError("original evaluation World/run identity required")
        if aci.memory_query is not None and (
            policy["cases"][case.case_id] != aci.memory_query.policy
            or evaluation_episode.sha256 not in aci.memory_query.excluded_episodes
            or run_id not in aci.memory_query.excluded_runs
        ):
            raise ValueError("Memory policy differs or does not exclude the current episode/run")
        provider = FrozenProviderV1.model_validate_json(_read(root, provider_settings))
        synthetic = provider.execution_kind == "SYNTHETIC_CONTRACT_CLIENT"
        if synthetic and (base.evaluator is not None or base.probe_evaluator is not None):
            raise ValueError("synthetic contract clients cannot use SDK/probe callbacks")
        if (
            base.proposals
            or base.results
            or base.validation_count
            or base.probe_count
            or base.finished
            or aci.trace
            or (aci.memory_query is not None and aci.memory_query.trace)
        ):
            raise ValueError("fresh episode-scoped interface and Memory context required")
        client = _provider_pin(provider, protocol, contract_client=contract_client)
        result.update(allocation=allocation, execution_kind=provider.execution_kind)
        context, workbook = EngineeringEpisodeContext(base), EpisodeWorkbook()
        # Gold case labels/operator paths never enter the public task projection.
        public_task = {
            key: task[key] for key in ("task_id", "goal", "evidence_scope", "safety") if key in task
        }
        deadline = started + 300
        state, first_proposal, investigations = "MODEL_DECISION_BUDGET_EXHAUSTED", None, 0
        for turn in range(protocol.max_model_decisions):
            if clock() >= deadline:
                state = "WALL_TIME_BUDGET_EXHAUSTED"
                break
            context.revision = turn
            frame, schema = context.frame(), aci.decision_schema()
            prompt = (
                INSTRUCTIONS
                + "\nTASK\n"
                + encoded(model_safe(public_task))
                + "\nCURRENT_STATE\n"
                + encoded(model_safe(frame))
                + "\nHOST_OPERATION_GUIDE\n"
                + encoded(build_rsk_tool_guide(schema))
                + "\nWORKBOOK\n"
                + encoded(workbook.payload())
            )
            if len(prompt.encode()) > 48000:
                state = "CONTEXT_BYTE_BUDGET_EXHAUSTED"
                break
            call_dir = output / f"model-turn-{turn:02d}"
            counter = "synthetic_contract_calls" if synthetic else "actual_model_client_invocations"
            result[counter] += 1
            (output / f"turn-{turn:02d}-client-input.json").write_text(
                json.dumps({"prompt": prompt, "response_schema": schema}, indent=2) + "\n"
            )
            mark = clock()
            try:
                outcome = client(
                    prompt, schema, call_dir, timeout_sec=min(120, max(0.001, deadline - clock()))
                )
            except Exception as exc:
                outcome = {
                    "status": "MODEL_CLIENT_EXCEPTION",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "provider_request_started": "UNKNOWN_FOR_EXCEPTION",
                }
            calls.append(
                {
                    "turn": turn + 1,
                    "outcome": outcome,
                    "wall_sec": clock() - mark,
                    "prompt_utf8_bytes": len(prompt.encode()),
                    "frame_hash": frame["state_hash"],
                }
            )
            (output / f"turn-{turn:02d}-client-outcome.json").write_text(
                json.dumps(outcome, indent=2) + "\n"
            )
            if isinstance(outcome.get("usage"), dict):
                for key, value in outcome["usage"].items():
                    if type(value) is int and value >= 0:
                        reported_usage[key] = reported_usage.get(key, 0) + value
            if outcome.get("native_tool_items"):
                state = "NATIVE_TOOL_PROTOCOL_VIOLATION"
                break
            if outcome.get("status") != "ACTUAL_MODEL_TURN_COMPLETED" or not isinstance(
                outcome.get("usage"), dict
            ):
                state = "MODEL_CLIENT_OR_USAGE_FAILURE"
                break
            measured = outcome["usage"]
            if (
                type(measured.get("input_tokens")) is not int
                or not 0 <= measured["input_tokens"] <= 64000
            ):
                state = "MODEL_USAGE_OR_CONTEXT_TOKEN_LIMIT_INVALID"
                break
            event_path = call_dir / "original-events.jsonl"
            with event_path.open("rb") as stream:
                event_raw = stream.read(16_000_001)
            if not 0 < len(event_raw) <= 16_000_000:
                raise ValueError("bounded original model events required")
            events = [json.loads(row) for row in event_raw.splitlines() if row.strip()]
            calls[-1]["model_events_original"] = {
                "path": str(event_path),
                "sha256": hashlib.sha256(event_raw).hexdigest(),
            }
            if host_client.native_tool_items(events):
                state = "NATIVE_TOOL_PROTOCOL_VIOLATION"
                break
            if not synthetic and usage_from_events(events) != measured:
                state = "ORIGINAL_MODEL_EVENT_USAGE_MISMATCH"
                break
            if not synthetic:
                result["actual_model_calls"] += 1
            for key, value in measured.items():
                if type(value) is int and value >= 0:
                    usage[key] = usage.get(key, 0) + value
            if clock() >= deadline:
                state = "WALL_TIME_BUDGET_EXHAUSTED"
                break
            messages = [
                event["item"]["text"]
                for event in events
                if event.get("type") == "item.completed"
                and event.get("item", {}).get("type") == "agent_message"
            ]
            if not messages:
                raise ValueError("original structured model decision missing")
            decision = json.loads(messages[-1])
            op = decision.get("decision", {}).get("operation")
            if op == "ros_engineer.investigate":
                if investigations >= protocol.max_read_only_probes:
                    state = "READ_ONLY_PROBE_BUDGET_EXHAUSTED"
                    break
                investigations += 1
            # Provider latency never permits changed originals, entry gates,
            # code or newly attached callbacks to authorize a host operation.
            if (
                review_pilot_entry(protocol, pilot_evidence, root=root)["status"]
                != "ELIGIBLE_FOR_BOUNDED_PILOT_ONLY"
            ):
                raise ValueError("pilot entry changed before host dispatch")
            for reference in (
                facts_manifest,
                evaluation_episode,
                provider_settings,
                memory_policy,
                facts.case,
                facts.model,
                facts.requirements,
                facts.staging,
                facts.task,
                facts.run_settings,
            ):
                _read(root, reference)
            _provider_pin(provider, protocol, contract_client=contract_client)
            if synthetic and (base.evaluator is not None or base.probe_evaluator is not None):
                raise ValueError("synthetic contract cannot acquire SDK/probe callbacks")
            tool_started = clock()
            tool_result = aci.call_model_decision(
                decision, remaining_sec=max(0.001, deadline - clock())
            )
            workbook.observe(
                tool_result, decision["decision_summary"], request=decision["decision"]
            )
            tools.append(
                {
                    "turn": turn + 1,
                    "decision": decision,
                    "result": tool_result,
                    "elapsed_sec": clock() - started,
                    "host_tool_wall_sec": clock() - tool_started,
                }
            )
            (output / f"turn-{turn:02d}-tool-result.json").write_text(
                json.dumps(tools[-1], indent=2) + "\n"
            )
            if "proposal" in tool_result and first_proposal is None:
                first_proposal = {
                    "decision": turn + 1,
                    "elapsed_sec": clock() - started,
                    "host_tool_wall_sec": clock() - tool_started,
                    "proposal_id": tool_result["proposal"]["proposal_id"],
                }
            if base.validation_count > 1 or base.probe_count > 1:
                raise ValueError("independent ACI World budget violated")
            if any(row.get("accepted_success") is True for row in base.results.values()):
                result["verified_success"] = not synthetic and clock() <= deadline
                state = (
                    "VERIFIED_CANONICAL_SIM_SUCCESS"
                    if result["verified_success"]
                    else "SYNTHETIC_OR_LATE_RESULT_NOT_STUDY_SUCCESS"
                )
                break
            if base.finished:
                state = "FINISHED_WITHOUT_ACCEPTED_SDK_RESULT"
                break
        result.update(
            status=state,
            first_valid_proposal=first_proposal,
            known_sanity_proposal_within_3_decisions=(
                None
                if allocation["stratum"] != "known_mechanism"
                else bool(
                    first_proposal
                    and first_proposal["decision"] <= protocol.first_known_valid_proposal_decisions
                )
            ),
            read_only_investigation_requests=investigations,
            new_Worlds_requested=base.probe_count + base.validation_count,
            verification_records=base.results,
            Memory_trace=[] if aci.memory_query is None else aci.memory_query.trace,
        )
    except (OSError, ValueError, TypeError, KeyError, StopIteration) as exc:
        result.update(
            status="NO_GO" if not calls else "INVALID_MODEL_DECISION_OR_ORIGINAL_INPUT",
            error=str(exc),
            error_type=type(exc).__name__,
        )
    result.update(
        actual_elapsed_sec=clock() - started,
        raw_model_calls=calls,
        tool_calls=tools,
        measured_usage=usage or None,
        facade_trace=aci.trace,
        artifact_scope="SYNTHETIC_CONTRACT_ONLY_NOT_MODEL_OR_PHYSICAL_EVIDENCE"
        if provider and provider.execution_kind == "SYNTHETIC_CONTRACT_CLIENT"
        else "REGISTERED_ACTUAL_MODEL_EPISODE_OR_NO_GO",
    )
    result["reported_usage_totals_including_partial"] = reported_usage or None
    result["physical_validation_phase"] = (
        "NOT_REQUESTED"
        if base.validation_count == 0
        else "VERIFIED_SUCCESS"
        if result["verified_success"]
        else "FAILED_OR_INCOMPLETE_OR_OUT_OF_BUDGET"
    )
    result["unconfirmed_model_client_invocations"] = (
        result["actual_model_client_invocations"] - result["actual_model_calls"]
    )
    result["actual_model_calls_definition"] = (
        "original provider usage replay verified; uncertain or failed invocations are separate"
    )
    result["result_hash"] = content_hash("rskepisode", result)
    with (output / "original-rsk-episode-result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    return result
