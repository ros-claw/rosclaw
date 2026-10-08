"""Finite public software tests; no robot execution or backend injection.

F43/F44 consume genuine default-backend chunks. Feedback is deliberately
synthetic action-as-state in model coordinates, not an environment transition.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

INPUT_ROOT = Path(__file__).resolve().parents[1] / "inputs"
PURE_IDS = (
    "F45_label_frame_freshness_match",
    "F45_no_grounding_blank_or_stale",
    "F46_reverse_edge_conflict",
    "F46_persistent_endpoint_and_disjoint_goal",
)
MODEL_IDS = (
    "F43_normalized_model_space",
    "F43_explicit_saved_stats_profile",
    "F44_actual_policy_chunk_consumption",
    "F44_generated_feedback_second_observation",
)
RESOURCE_KEYS = {"asset_root", "asset_manifest", "policy_dir", "vlm_dir", "tokenizer_dir"}
REQUEST_KEYS = ("device", "seed", "task", "state", "images")


def _finite_vector(value, length):
    return (type(value) is list and len(value) == length
            and all(type(x) in (int, float) and math.isfinite(x) for x in value))


def _verdict(case_id, checks, **extra):
    assertions = [{"criterion": name, "passed": bool(ok)} for name, ok in checks]
    return {"id": case_id, "passed": all(a["passed"] for a in assertions),
            "criteria": [a["criterion"] for a in assertions],
            "assertions": assertions, **extra}


def run_pure(inputs, modules):
    """Call protected solvers for every public input; compare actual outputs."""
    cases = inputs["cases"] if isinstance(inputs, dict) else inputs
    if len(cases) != 4 or {c["id"] for c in cases} != set(PURE_IDS):
        raise ValueError("Expected exactly four public structured cases")
    answers = []
    for case in cases:
        solver = modules.grounding.solve if case["direction"] == "F45" else modules.reservation.solve
        subcases = case.get("subcases", [case])
        outputs, checks = [], []
        for index, subcase in enumerate(subcases):
            output = solver(copy.deepcopy(subcase["input"]))
            outputs.append(output)
            checks.append((f"subcase {index}: protected output equals public expected structure",
                           output == subcase["expected"]))
        if case["id"] == PURE_IDS[0]:
            checks.extend([
                ("normalized label, map frame and freshness select candidate a", outputs[0].get("candidate_id") == "a"),
                ("equidistant tie resolved lexically, not camera proximity", outputs[0].get("goal_position") == [-1.0, 0.0, 0.0]),
            ])
        elif case["id"] == PURE_IDS[1]:
            checks.extend([
                ("blank label rejected", outputs[0].get("status") == "INVALID"),
                ("all stale candidates yield no grounding", outputs[1].get("status") == "NOT_GROUNDED"),
                ("neither negative case emits a goal", all("goal_position" not in o and "candidate_id" not in o for o in outputs)),
            ])
        else:
            decisions = outputs[0].get("decisions", [])
            checks.append(("first agent accepted before contention", bool(decisions) and decisions[0].get("accepted") is True))
            tick = 1 if case["id"] == PURE_IDS[2] else 3
            checks.append(("contender rejected at exact conflict tick", len(decisions) >= 2 and decisions[1].get("accepted") is False and decisions[1].get("conflict_tick") == tick))
            if case["id"] == PURE_IDS[3]:
                checks.append(("disjoint third goal remains accepted", len(decisions) == 3 and decisions[2].get("accepted") is True and decisions[2].get("conflict_tick") is None))
        answers.append(_verdict(case["id"], checks, outputs=outputs))
    return {"cases": answers, "status": "SOURCE_VALIDATED" if all(a["passed"] for a in answers) else "SOURCE_FAILED"}


def _request(template):
    request = {k: copy.deepcopy(template[k]) for k in REQUEST_KEYS}
    for image in request["images"].values():
        if "relative_path" in image:
            image["path"] = str((INPUT_ROOT / image.pop("relative_path")).resolve())
        if set(image) != {"path", "sha256", "shape"}:
            raise ValueError("Image record must use exact public keys")
        if Path(image["path"]).resolve().parent != INPUT_ROOT:
            raise ValueError("Only fixed public images allowed")
        if hashlib.sha256(Path(image["path"]).read_bytes()).hexdigest() != image["sha256"]:
            raise ValueError("Public image bytes changed")
    return request


def _consume_chunk(result):
    actions = result.get("actions")
    if not (type(actions) is list and len(actions) == 1
            and type(actions[0]) is list and len(actions[0]) == 50
            and all(_finite_vector(row, 6) for row in actions[0])):
        raise ValueError("Expected finite actual action chunk [1,50,6]")
    # Read every returned scalar rather than accepting a shape/provenance label.
    flat = [x for row in actions[0] for x in row]
    return {"scalar_count": len(flat), "min": min(flat), "max": max(flat),
            "l1": math.fsum(abs(x) for x in flat),
            "first_action": copy.deepcopy(actions[0][0]),
            "last_action": copy.deepcopy(actions[0][-1])}


def run_model(requests, vla, resource_bindings):
    """Two official calls only, with causal first-action -> second-state feedback."""
    templates = requests["requests"] if isinstance(requests, dict) else requests
    if len(templates) != 2 or set(resource_bindings) != RESOURCE_KEYS:
        raise ValueError("Exactly two requests and exact resource keys required")
    if [r["id"] for r in templates] != ["model_normalized", "model_declared_so100_stats"]:
        raise ValueError("Unexpected public request order")
    first_request = _request(templates[0])
    first = vla.run_inference(first_request, resource_bindings,
                              action_space_policy=copy.deepcopy(templates[0]["action_space_policy"]))
    first_consumed = _consume_chunk(first)
    feedback = copy.deepcopy(first["actions"][0][0])
    second_request = _request(templates[1])
    second_request["state"] = copy.deepcopy(feedback)
    second = vla.run_inference(second_request, resource_bindings,
                               action_space_policy=copy.deepcopy(templates[1]["action_space_policy"]))
    second_consumed = _consume_chunk(second)
    stats = json.loads((INPUT_ROOT / "EXACT_PROFILE_RESOLUTION.json").read_text())["stats"]
    a, b = first["action_space"], second["action_space"]
    safe = lambda s: (s.get("robot_units_verified") is False
                      and s.get("physical_calibration_established") is False
                      and s.get("g1_rh56_mapping") == "NOT_ESTABLISHED_NO_CONVERSION")
    cases = [
        _verdict(MODEL_IDS[0], [
            ("first output is normalized with no implicitly selected stats profile", a.get("kind") == "NORMALIZED_MODEL_ACTION" and a.get("profile") is None and a.get("stats_sha256") is None),
            ("normalized result does not assert physical units or calibration", safe(a)),
        ]),
        _verdict(MODEL_IDS[1], [
            ("second output explicitly selects saved so100.buffer.action", b.get("kind") == "ROBOT_PROFILE_ACTION" and b.get("profile") == stats["profile"]),
            ("saved mean/std and full stats digest match declared public binding", b.get("action_mean") == stats["mean"] and b.get("action_std") == stats["std"] and b.get("stats_sha256") == stats["stats_sha256"]),
            ("official postprocessor applied without physical calibration claim", b.get("postprocessor_applied") is True and safe(b)),
        ]),
        _verdict(MODEL_IDS[2], [
            ("both returned chunks consumed as 300 finite scalar values each", all(c["scalar_count"] == 300 and math.isfinite(c["l1"]) for c in [first_consumed, second_consumed])),
            ("both calls report official default CPU backend and no mock", all(r.get("dependencies_provenance") == "OFFICIAL_LEROBOT_DEFAULT_BACKEND" and r.get("mock_inference_not_actual_nn") is False and r.get("device") == "cpu" for r in [first, second])),
            ("both chunks postprocessed with correct shape and no physical action", all(r.get("status") == "ACTIONS_POSTPROCESSED" and r.get("action_shape") == [1, 50, 6] and r.get("physical_action_executed") is False for r in [first, second])),
        ], consumption=[first_consumed, second_consumed]),
        _verdict(MODEL_IDS[3], [
            ("second actual request state exactly equals first normalized action", _finite_vector(feedback, 6) and second_request["state"] == first["actions"][0][0]),
            ("feedback is consumed in policy coordinates without state statistics", all(r["state_space"].get("kind") == "POLICY_INPUT_COORDINATES" and r["state_space"].get("state_stats_present") is False and r["state_space"].get("physical_robot_state_equivalence") is False for r in [first, second])),
            ("task, seed and fixed images unchanged across feedback request", all(second_request[k] == first_request[k] for k in ["task", "seed", "device", "images"])),
        ], feedback_state=feedback),
    ]
    return {"model_results": [first, second], "cases": cases,
            "status": "SOURCE_VALIDATED" if all(c["passed"] for c in cases) else "SOURCE_FAILED"}
