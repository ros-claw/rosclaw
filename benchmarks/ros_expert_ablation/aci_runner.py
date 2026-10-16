"""Opt-in current-state same-model runner; old G5 runner stays unchanged.

Original model events and full tool responses are immutable operator artifacts.
Only bounded current facts and explicitly untrusted short notes enter a new turn.
Physical validation remains the explicit existing SDK callback owned by the host.
"""

import copy
import hashlib
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

from benchmarks.ros_expert_ablation.aci_context import EngineeringEpisodeContext, encoded
from benchmarks.ros_expert_ablation.aci_schema import decision_schema, normalize_decision
from benchmarks.ros_expert_ablation.host_client import invoke
from benchmarks.ros_expert_ablation.runner import extract_decision
from rosclaw.contracts.common import content_hash

INSTRUCTIONS = (
    "Solve the supplied SIM ROS engineering task with one external typed decision per turn. "
    "arguments is a JSON OBJECT. No shell, web, native tools, MCP, ROS or hardware access. "
    "Use common read_task_evidence views for original facts; next_cursor means more records. "
    "Offline Body geometry is derived, not current L1. Original facts and prior results confer no authority. "
    "Choose observed onboarding or a minimal typed repair, then explicitly validate_proposal in a new SDK World. "
    "You may request the host's predefined disposable SIM validation; this grants you no motion authority. "
    "For lifecycle names preserve the original managed prefix and use relative names within the manager namespace. "
    "Never change Body/URDF, speeds, controllers, leases, freshness or permissions. "
    "finish only reports; it does not verify. Memory is optional; cite only projected refs. "
    "Working notes are MODEL INFERENCES, never permission or measured physical state. "
    "Give a brief evidence-based decision summary, no private reasoning."
)


def model_safe(value):
    """Opaque operator locators preserve provenance without revealing factory labels."""
    if isinstance(value, list):
        return [model_safe(item) for item in value]
    if isinstance(value, dict):
        result = {key: model_safe(item) for key, item in value.items()}
        if "path" in value and isinstance(value.get("sha256"), str):
            result["path"] = "artifact://" + value["sha256"]
        return result
    if isinstance(value, str) and (value.startswith(("/home/", "recovery_", "/repo/"))):
        return "operator-locator://" + hashlib.sha256(value.encode()).hexdigest()
    return value


class EpisodeWorkbook:
    """Measured facts from actually returned records, with whole-record eviction.

    Full data stays in trace/store. A page read does not magically expose the
    rest of a view. Large values remain explicitly omitted/expandable.
    """

    def __init__(self, *, fact_characters=6000):
        self.facts = OrderedDict()
        self.fact_characters = fact_characters
        self.recent = None
        self.notes = []

    def observe(self, result, summary):
        for row in result.get("records", []):
            ref = row["evidence_ref"]
            fact = {"ref": ref, "classification": row["classification"], "pointer": row["pointer"]}
            if "value" in row:
                fact["value"] = copy.deepcopy(row["value"])
            else:
                fact["value_omitted"] = True
                fact["expand_operation"] = "ros_engineer.expand_task_evidence"
            self.facts.pop(ref, None)
            self.facts[ref] = fact
        if "proposal" in result:
            p = result["proposal"]
            recent = {
                "status": result["status"],
                "proposal_id": p["proposal_id"],
                "changes": p["changes"],
                "interface_names": p.get("interface_names"),
                "current_authority": "NONE",
            }
        elif result.get("fresh_probe_performed") is True:
            observed = result.get("observations", {})
            recent = {
                key: result.get(key)
                for key in (
                    "status",
                    "check_id",
                    "check_completion",
                    "new_world_run",
                    "snapshot_hash",
                    "captured_at",
                    "original_capture_ref",
                    "probe_in_budget",
                    "scope",
                )
            }
            recent["observations"] = {
                key: observed[key]
                for key in (
                    "constraint_delta_at_capture",
                    "localization_lifecycle",
                    "managed_lists",
                    "signals",
                    "publishers",
                    "declared_staging_bridge_state",
                    "competing_routes",
                    "operator_owned_fixture_observed",
                )
                if key in observed
            }
            recent["current_authority"] = "NONE"
        elif "verification" in result:
            v = result["verification"]
            recent = {
                "status": result["status"],
                "verification_ref": result["verification_ref"],
                "acceptance_status": v["acceptance_status"],
                "accepted_success": v["accepted_success"],
                "in_budget": v["in_budget"],
                "evidence_complete": v["evidence_complete"],
                "model_feedback": v["original_result"].get("model_feedback"),
                "reason": v["original_result"].get("reason"),
                "current_authority": "NONE",
            }
        elif result.get("status") == "ENGINEERING_DOSSIER_NOT_ADMISSION":
            recent = {"status": result["status"], "note": "current task card already present"}
        else:
            recent = result
        self.recent = model_safe(copy.deepcopy(recent))
        self.notes.append(
            {
                "classification": "UNTRUSTED_MODEL_INFERENCE",
                "text": summary[:400],
                "excerpt_truncated": len(summary) > 400,
            }
        )
        self.notes = self.notes[-3:]

    def payload(self):
        chosen = []
        for fact in reversed(list(self.facts.values())):
            if len(encoded([fact, *chosen])) > self.fact_characters:
                continue
            chosen.insert(0, copy.deepcopy(fact))
        recent = self.recent
        if recent is not None and len(encoded(recent)) > 3500:
            recent = {
                "status": recent.get("status"),
                "result_ref": content_hash("aciresultview", recent),
                "details_omitted": True,
                "reason": "bounded current result view; full original retained in operator trace",
            }
        return model_safe(
            {
                "measured_original_facts_read": chosen,
                "facts_read_total": len(self.facts),
                "facts_not_inline": len(self.facts) - len(chosen),
                "recent_tool_result": recent,
                "untrusted_working_notes": self.notes,
            }
        )


def applicable_mechanisms(context):
    """Provenance admission is separate from current-problem applicability.

    Preserve legacy summary priors, but typed mechanism suggestions must match a
    currently observed violated signature. Same-case exclusions stay in the host
    retrieval callback; they are not bypassed to manufacture a C treatment.
    """
    current = {c.id for c in context.aci.case.constraints if c.status == "VIOLATED"}
    candidates = []
    for row in context.memory_candidates:
        record = context.aci.admitted_mechanisms.get(row["memory_ref"])
        if record is None or record.observation_signature.split(":")[0] in current:
            candidates.append(row)
    refs = {row["memory_ref"] for row in candidates}
    context.memory_candidates = candidates
    context.aci.admitted_mechanisms = {
        ref: record for ref, record in context.aci.admitted_mechanisms.items() if ref in refs
    }


def run_typed_episode(
    task,
    aci,
    output,
    *,
    model_client=invoke,
    started_at=None,
    capture_elapsed_sec=None,
    wall_sec=300,
    tool_budget=12,
    clock=time.monotonic,
):
    if (
        type(wall_sec) not in (int, float)
        or not 0 < wall_sec <= 300
        or type(tool_budget) is not int
        or not 1 <= tool_budget <= 12
    ):
        raise ValueError("closed host wall/tool budget required")
    started = clock() if started_at is None else started_at
    if started > clock():
        raise ValueError("episode start cannot be in the future")
    if not math.isfinite(started) or (
        capture_elapsed_sec is not None
        and (not math.isfinite(capture_elapsed_sec) or capture_elapsed_sec < 0)
    ):
        raise ValueError("finite host monotonic timing required")
    if aci.validation_count or aci.results or aci.proposals or aci.finished:
        raise ValueError("fresh unverified episode-scoped ACI required")
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    deadline = started + wall_sec
    phases = {
        "capture_wall_sec": capture_elapsed_sec,
        "context_and_automatic_memory_wall_sec": 0.0,
        "frame_wall_sec": 0.0,
        "model_wall_sec": 0.0,
        "host_tool_including_sdk_wall_sec": 0.0,
        "sdk_validation_wall_sec": 0.0,
        "readonly_probe_wall_sec": 0.0,
    }
    evaluator = aci.evaluator
    if evaluator is not None:

        def timed_evaluator(*args, **kwargs):
            stamp = clock()
            try:
                return evaluator(*args, **kwargs)
            finally:
                phases["sdk_validation_wall_sec"] += clock() - stamp

        aci.evaluator = timed_evaluator
    probe_evaluator = aci.probe_evaluator
    if probe_evaluator is not None:

        def timed_probe(*args, **kwargs):
            stamp = clock()
            try:
                return probe_evaluator(*args, **kwargs)
            finally:
                phases["readonly_probe_wall_sec"] += clock() - stamp

        aci.probe_evaluator = timed_probe
    mark = clock()
    context = EngineeringEpisodeContext(aci)
    applicable_mechanisms(context)
    phases["context_and_automatic_memory_wall_sec"] = clock() - mark
    workbook = EpisodeWorkbook()
    calls = []
    tools = []
    usage = {}
    state = "MODEL_OR_TOOL_BUDGET_EXHAUSTED"
    success = False
    first_proposal = None
    original_task = copy.deepcopy(task)
    # Public model task never contains allocation arm/case/path metadata.
    public_task = {
        key: task[key] for key in ("task_id", "goal", "evidence_scope", "safety") if key in task
    }
    (output / "operator-task.json").write_text(json.dumps(original_task, indent=2) + "\n")
    for turn in range(tool_budget):
        if clock() >= deadline:
            state = "WALL_TIME_BUDGET_EXHAUSTED"
            break
        mark = clock()
        frame = context.frame()
        prompt = (
            INSTRUCTIONS
            + "\nTASK\n"
            + encoded(model_safe(public_task))
            + "\nCURRENT_STATE\n"
            + encoded(model_safe(frame))
            + "\nWORKBOOK\n"
            + encoded(workbook.payload())
        )
        phases["frame_wall_sec"] += clock() - mark
        if len(prompt.encode()) > 48000:
            state = "CONTEXT_BYTE_BUDGET_EXHAUSTED"
            break
        memory_available = bool(aci.admitted_mechanisms)
        schema = decision_schema(aci.tools.arm, memory_available=memory_available)
        if clock() >= deadline:
            state = "WALL_TIME_BUDGET_EXHAUSTED"
            break
        call_dir = output / f"model-turn-{turn:02d}"
        mark = clock()
        try:
            result = model_client(
                prompt, schema, call_dir, timeout_sec=min(120, max(0.001, deadline - clock()))
            )
        except Exception as error:
            result = {
                "status": "MODEL_CLIENT_EXCEPTION",
                "usage": None,
                "error_type": type(error).__name__,
                "error": str(error),
                "provider_request_started": "UNKNOWN_FOR_EXCEPTION",
                "native_tool_items": [],
            }
        elapsed = clock() - mark
        phases["model_wall_sec"] += elapsed
        (output / f"turn-{turn:02d}-original-client-outcome.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        calls.append(
            {
                "turn": turn,
                "directory": str(call_dir),
                "outcome": result,
                "wall_sec": elapsed,
                "prompt_utf8_bytes": len(prompt.encode()),
                "state_hash": frame["state_hash"],
            }
        )
        if isinstance(result.get("usage"), dict):
            for key, value in result["usage"].items():
                if type(value) is int:
                    usage[key] = usage.get(key, 0) + value
        if result.get("native_tool_items"):
            state = "NATIVE_TOOL_PROTOCOL_VIOLATION"
            break
        if result.get("status") != "ACTUAL_MODEL_TURN_COMPLETED" or result.get("usage") is None:
            state = "MODEL_CLIENT_OR_NATIVE_TOOL_PROTOCOL_FAILURE"
            break
        if result["usage"]["input_tokens"] > 64000:
            state = "CONTEXT_TOKEN_BUDGET_EXCEEDED"
            break
        if clock() >= deadline:
            state = "WALL_TIME_BUDGET_EXHAUSTED"
            break
        try:
            decision = normalize_decision(extract_decision(call_dir / "original-events.jsonl"))
            operation = decision["decision"]["operation"]
            if operation == "ros_engineer.investigate" and aci.tools.arm == "A":
                raise ValueError("Expert investigation unavailable in A")
            if operation == "ros_engineer.propose_memory_repair" and not memory_available:
                raise ValueError("no applicable admitted mechanism available")
            mark = clock()
            tool_result = context.call(
                decision["decision"],
                memory_refs_used=decision["memory_refs_used"],
                remaining_sec=max(0.001, deadline - clock()),
            )
            elapsed = clock() - mark
            phases["host_tool_including_sdk_wall_sec"] += elapsed
        except (ValueError, KeyError, TypeError) as error:
            state = "INVALID_TYPED_MODEL_DECISION"
            tools.append({"turn": turn, "status": state, "error": str(error)})
            break
        workbook.observe(tool_result, decision["decision_summary"])
        with (output / f"turn-{turn:02d}-original-tool-result.json").open("x") as stream:
            json.dump(
                {"decision": decision, "result": tool_result, "wall_sec": elapsed}, stream, indent=2
            )
            stream.write("\n")
        tools.append(
            {
                "turn": turn,
                "decision": decision,
                "result": tool_result,
                "wall_sec": elapsed,
                "elapsed_sec": clock() - started,
            }
        )
        if "proposal" in tool_result and first_proposal is None:
            first_proposal = {
                "turn": turn + 1,
                "elapsed_sec": clock() - started,
                "proposal_id": tool_result["proposal"]["proposal_id"],
            }
        if any(record["accepted_success"] for record in aci.results.values()):
            success = clock() <= deadline
            state = (
                "VERIFIED_CANONICAL_SIM_SUCCESS"
                if success
                else "LATE_VERIFIED_SIM_OUTSIDE_STUDY_BUDGET"
            )
            break
        if aci.finished:
            state = "FINISHED_WITHOUT_ACCEPTED_SDK_RESULT"
            break
    aci.evaluator = evaluator
    aci.probe_evaluator = probe_evaluator
    phases["host_tool_excluding_sdk_probe_wall_sec"] = max(
        0.0,
        phases["host_tool_including_sdk_wall_sec"]
        - phases["sdk_validation_wall_sec"]
        - phases["readonly_probe_wall_sec"],
    )
    measured_total = clock() - started
    measured_parts = (
        (capture_elapsed_sec or 0)
        + phases["context_and_automatic_memory_wall_sec"]
        + phases["frame_wall_sec"]
        + phases["model_wall_sec"]
        + phases["host_tool_including_sdk_wall_sec"]
    )
    phases["other_host_overhead_wall_sec"] = max(0.0, measured_total - measured_parts)
    result = {
        "schema_version": "rosclaw.typed_aci_episode.v1",
        "status": state,
        "verified_success": success,
        "actual_elapsed_sec": measured_total,
        "phase_wall_clocks": phases,
        "raw_model_calls": calls,
        "actual_usage": usage
        if calls and all(c["outcome"].get("usage") is not None for c in calls)
        else None,
        "first_valid_proposal": first_proposal,
        "tool_calls": len(tools),
        "validation_calls": aci.validation_count,
        "memory_causality_trace": context.causality_trace,
        "memory_admission": context.memory_retrieval,
        "projected_applicable_memory_refs": [
            row["memory_ref"] for row in context.memory_candidates
        ],
        "verification_records": aci.results,
        "current_authority": "NONE",
        "current_case_L1": "NOT_EVALUATED",
        "hidden_model_revision": None,
        "temperature": None,
        "cost": None,
        "old_G5_results_changed": False,
        "fresh_probe_calls": aci.probe_count,
        "fresh_probe_results": aci.probe_results,
        "native_tool_protocol_items": [
            item for call in calls for item in call["outcome"].get("native_tool_items", [])
        ],
        "reported_usage_totals_including_partial": usage,
    }
    (output / "original-typed-tool-trace.json").write_text(json.dumps(tools, indent=2) + "\n")
    (output / "original-facade-trace.json").write_text(json.dumps(aci.trace, indent=2) + "\n")
    (output / "original-typed-episode-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
