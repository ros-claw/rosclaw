"""Provider-friendly structured decision schema; host types remain authoritative.

Root is an object, unions use nested anyOf, and optional defaults are represented
as required nullable fields. Only those declared defaults may be omitted again.
No new model invocation or alternate execution path is introduced here.
"""

import copy

from benchmarks.ros_expert_ablation.physical_aci import CALL_ADAPTER

COMMON = {
    "InspectCall",
    "EvidenceCall",
    "ExpandCall",
    "RepairCall",
    "OnboardingCall",
    "ValidateCall",
    "ExplainCall",
    "FinishCall",
}


def _project(value):
    if isinstance(value, list):
        return [_project(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        if key in {"title", "discriminator", "default"}:
            continue
        if key == "const":
            out["enum"] = [item]
        else:
            out["anyOf" if key == "oneOf" else key] = _project(item)
    if out.get("type") == "object":
        props = out.get("properties", {})
        old_required = value.get("required", [])
        for key in props:
            if key not in old_required:
                props[key] = {"anyOf": [props[key], {"type": "null"}]}
        out["required"] = list(props)
        out["additionalProperties"] = False
    return out


def decision_schema(arm, *, memory_available=False):
    if arm not in {"A", "B", "C"}:
        raise ValueError("closed preregistered arm required")
    schema = CALL_ADAPTER.json_schema()
    allowed = (
        COMMON
        | ({"InvestigateCall"} if arm in {"B", "C"} else set())
        | ({"MemoryRepairCall"} if arm == "C" and memory_available else set())
    )
    choices = [ref for ref in schema["oneOf"] if ref["$ref"].rsplit("/", 1)[1] in allowed]
    definitions = schema["$defs"]
    needed = set()

    def follow(value):
        if isinstance(value, dict):
            if "$ref" in value:
                name = value["$ref"].rsplit("/", 1)[1]
                if name not in needed:
                    needed.add(name)
                    follow(definitions[name])
            for key, item in value.items():
                if key != "$ref":
                    follow(item)
        elif isinstance(value, list):
            for item in value:
                follow(item)

    follow(choices)
    raw = {
        "type": "object",
        "properties": {
            "decision": {"anyOf": choices},
            "decision_summary": {
                "type": "string",
                "description": "brief evidence-based decision; no private reasoning",
            },
            "memory_refs_used": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["decision", "decision_summary", "memory_refs_used"],
        "additionalProperties": False,
        "$defs": {name: definitions[name] for name in sorted(needed)},
    }
    if arm != "C" or not memory_available:
        raw["properties"]["memory_refs_used"]["maxItems"] = 0
    return _project(raw)


def normalize_decision(value):
    """Remove provider nulls only where the authoritative schema declares a default."""
    if not isinstance(value, dict) or set(value) != {
        "decision",
        "decision_summary",
        "memory_refs_used",
    }:
        raise ValueError("closed structured decision envelope required")
    if type(value["decision_summary"]) is not str or len(value["decision_summary"]) > 2000:
        raise ValueError("bounded brief decision summary required")
    if type(value["memory_refs_used"]) is not list or any(
        type(ref) is not str for ref in value["memory_refs_used"]
    ):
        raise ValueError("actual admitted Memory reference list required")
    request = copy.deepcopy(value["decision"])
    if (
        not isinstance(request, dict)
        or type(request.get("operation")) is not str
        or not isinstance(request.get("arguments"), dict)
    ):
        raise ValueError("operation arguments must be an object")
    defaults = {
        "ros_engineer.read_task_evidence": {"cursor"},
        "ros_engineer.expand_task_evidence": {"offset", "limit"},
    }.get(request.get("operation"), set())
    for key in defaults:
        if request["arguments"].get(key, object()) is None:
            del request["arguments"][key]
    CALL_ADAPTER.validate_python(request)
    return {**value, "decision": request}
