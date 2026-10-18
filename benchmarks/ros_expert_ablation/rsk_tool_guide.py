"""New RSK operation effects without changing frozen G5 operation descriptions."""

from copy import deepcopy

from benchmarks.ros_expert_ablation.aci_context import encoded
from benchmarks.ros_expert_ablation.aci_tool_guide import build_tool_guide

RSK_DESCRIPTIONS = {
    "ros_engineer.query_cross_body_memory": (
        "READ_ONLY_HISTORICAL_CONCEPTS_NO_AUTHORITY",
        "Query the frozen provenance-filtered Memory and insert returned abstract concepts into context. Historical parameters and permissions do not transfer. Starts no World.",
    ),
    "ros_engineer.propose_cross_body_memory_repair": (
        "REGISTER_CURRENT_STAGING_ONLY",
        "Select an actually returned memory_ref. Host rechecks current conditions and independently compiles the current minimal diff. May abstain. Registers a conditional proposal only; starts no World.",
    ),
}


def build_rsk_tool_guide(schema):
    legacy = deepcopy(schema)
    definitions = schema["$defs"]
    choices, extra = [], []
    for choice in schema["properties"]["decision"]["anyOf"]:
        call = definitions[choice["$ref"].rsplit("/", 1)[1]]["properties"]
        operation = call["operation"]["enum"][0]
        if operation not in RSK_DESCRIPTIONS:
            choices.append(choice)
            continue
        arguments = call["arguments"]
        if "$ref" in arguments:
            arguments = definitions[arguments["$ref"].rsplit("/", 1)[1]]
        effect, description = RSK_DESCRIPTIONS[operation]
        extra.append(
            {
                "operation": operation,
                "argument_fields": list(arguments["properties"]),
                "host_effect": effect,
                "description": description,
            }
        )
    legacy["properties"]["decision"]["anyOf"] = choices
    guide = build_tool_guide(legacy)
    guide["operations_available_this_turn"].extend(extra)
    if len(encoded(guide)) > 6500:
        raise ValueError("bounded new RSK operation guide exceeded")
    return guide
