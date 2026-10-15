"""Original same-model turns with externally mediated, budgeted task tools."""

import copy
import json
import time
from pathlib import Path

from benchmarks.ros_expert_ablation.host_client import invoke
from benchmarks.ros_expert_ablation.protocol import canonical_sha


def response_schema(operations):
    return {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": operations},
            "arguments": {"type": "string"},
            "decision_summary": {"type": "string"},
            "memory_refs_used": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["operation", "arguments", "decision_summary", "memory_refs_used"],
        "additionalProperties": False,
    }


def extract_decision(path):
    events = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    messages = [
        e["item"]["text"]
        for e in events
        if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "agent_message"
    ]
    if not messages:
        raise ValueError("original structured model decision missing")
    return json.loads(messages[-1])


def run_episode(
    task,
    tools,
    output,
    evaluator,
    *,
    model_client=invoke,
    wall_time_sec=300,
    tool_budget=12,
    max_tests=2,
    clock=time.monotonic,
):
    """Evaluator owns disposable staging/Worlds; this function never opens ROS."""
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    started = clock()
    deadline = started + wall_time_sec
    calls = []
    trace = []
    tests = []
    usage = {}
    unsafe = 0
    state = "MODEL_OR_TOOL_BUDGET_EXHAUSTED"
    success = False
    transcript = []
    instructions = (
        "Solve the supplied ROS engineering task using only the external structured tool protocol. "
        "Do not call native CLI tools, shell, web, MCP, ROS or hardware interfaces. "
        "Return one schema-valid action each turn; arguments is a JSON object encoded as a string. "
        "Give only a brief evidence-based decision summary, no private reasoning trace. "
        "Use list_files/read_file for raw files (RFC6901 pointer, offset/limit for long data). "
        "propose_patch takes changes:[{file,pointer,value}] in declared staging fields; "
        "select_interfaces takes interface_names with four observed ROS roles. "
        "run_tests requests independent validation in a fresh World. Discovery PROPOSED is unbound. "
        "No successful result follows from a proposal, historical snapshot or recalled Memory. "
        "Only reference Memory IDs actually admitted during this episode. "
        "Available operations: " + ", ".join(tools.operations)
    )
    (output / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    for turn in range(tool_budget):
        if clock() >= deadline:
            state = "WALL_TIME_BUDGET_EXHAUSTED"
            break
        prompt = (
            instructions
            + "\nTASK\n"
            + json.dumps(task, ensure_ascii=False)
            + "\nTRANSCRIPT\n"
            + json.dumps(transcript, ensure_ascii=False)
        )
        if len(prompt.encode()) > 48000:
            state = "CONTEXT_BYTE_BUDGET_EXHAUSTED"
            break
        call_dir = output / f"model-turn-{turn:02d}"
        result = model_client(
            prompt,
            response_schema(tools.operations),
            call_dir,
            timeout_sec=min(120, max(1, deadline - clock())),
        )
        calls.append({"directory": str(call_dir), "outcome": result})
        if result.get("status") != "ACTUAL_MODEL_TURN_COMPLETED" or result.get("usage") is None:
            state = "ACTUAL_MODEL_CALL_FAILED_OR_USAGE_MISSING"
            break
        for key, value in result["usage"].items():
            if type(value) is int:
                usage[key] = usage.get(key, 0) + value
        if result["usage"]["input_tokens"] > 64000:
            state = "CONTEXT_TOKEN_BUDGET_EXCEEDED"
            break
        if clock() >= deadline:
            state = "WALL_TIME_BUDGET_EXHAUSTED"
            break
        try:
            decision = extract_decision(call_dir / "original-events.jsonl")
            op = decision["operation"]
            args = json.loads(decision["arguments"])
            if op not in tools.operations or type(args) is not dict:
                raise ValueError("invalid external operation")
        except (ValueError, KeyError, TypeError):
            state = "INVALID_ORIGINAL_MODEL_DECISION"
            break
        record = {"turn": turn, "original_decision": decision, "elapsed_sec": clock() - started}
        if op in {"run_tests", "finish"}:
            if len(tests) >= max_tests:
                state = "FRESH_WORLD_TEST_BUDGET_EXHAUSTED"
                break
            tested_digest = canonical_sha(
                {"staging": tools.staging, "selected_interfaces": tools.selected_interfaces}
            )
            evaluated = evaluator(
                copy.deepcopy(tools.staging),
                copy.deepcopy(tools.selected_interfaces),
                len(tests),
                remaining_sec=max(0, deadline - clock()),
            )
            tests.append({"proposal_sha256": tested_digest, "original_evaluation": evaluated})
            record["tool_result"] = evaluated
            trace.append(record)
            if clock() >= deadline:
                state = "FRESH_WORLD_VALIDATION_EXCEEDED_EPISODE_BUDGET"
                break
            if evaluated.get("verified_success") is True and not (
                evaluated.get("verification_source") == "INDEPENDENT_ORIGINAL_SDK_REVIEW"
                and all(
                    evaluated.get(k)
                    for k in [
                        "new_world_run",
                        "fresh_l1_ref",
                        "canonical_receipts",
                        "independent_stop_ref",
                    ]
                )
            ):
                state = "SUCCESS_REJECTED_MISSING_INDEPENDENT_PHYSICS_EVIDENCE"
                break
            if evaluated.get("verified_success") is True:
                success = True
                state = "VERIFIED_CANONICAL_SIM_SUCCESS"
                break
            if op == "finish":
                state = "UNVERIFIED_ORIGINAL_MODEL_FINISH"
                break
        else:
            try:
                record["tool_result"] = tools.call(op, args)
            except (ValueError, KeyError, TypeError) as rejected:
                unsafe += int(op in {"propose_patch", "select_interfaces"})
                record["tool_result"] = {
                    "status": "REJECTED_ORIGINAL_PROPOSAL_OR_TOOL",
                    "error": str(rejected),
                }
            trace.append(record)
        raw = json.dumps(record["tool_result"], ensure_ascii=False)
        if len(raw) > 16000:
            name = f"tool-result-{turn:02d}.json"
            tools.files[name] = raw
            shown = {
                "status": "ORIGINAL_TOOL_RESULT_STORED_READ_WITH_READ_FILE",
                "file": name,
                "characters": len(raw),
            }
        else:
            shown = record["tool_result"]
        transcript.extend([{"model": decision}, {"tool_result": shown}])
        (output / "original-tool-trace.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False) + "\n"
        )
    result = {
        "status": state,
        "verified_success": success,
        "elapsed_sec": clock() - started,
        "raw_model_calls": calls,
        "actual_usage": usage or None,
        "tool_calls": len(trace),
        "unsafe_proposals_blocked": unsafe,
        "patch_trace": tools.patch_trace,
        "memory_causality_trace": tools.memory_trace,
        "tests": tests,
        "claim_scope": "SIMULATION_ONLY",
        "hardware_verified": False,
        "v1_done": False,
    }
    (output / "original-tool-trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False) + "\n"
    )
    (output / "original-episode-result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    return result
