"""New RSK operation effects without changing frozen G5 operation descriptions."""

from copy import deepcopy

from benchmarks.ros_expert_ablation.aci_context import encoded
from benchmarks.ros_expert_ablation.aci_tool_guide import build_tool_guide

RSK_DESCRIPTIONS = {
    "ros_engineer.inspect_operational_graph": (
        "READ_ONLY_SEALED_OPERATIONAL_FINDINGS",
        "Page current original operational findings by cursor/limit. QoS/time/TF findings are distinct from five-mechanism repair checks. UNKNOWN remains unknown; no L1 or action authority.",
    ),
    "ros_engineer.plan_diagnostic_experiment": (
        "REGISTER_READ_ONLY_DIAGNOSTIC_PLAN_NOT_EXECUTED",
        "Select actual graph issue_ids and a bounded budget. The current Core proposes competing hypotheses, discriminating measurements and explicit unknown/deferred scope; no probe or repair runs implicitly.",
    ),
    "ros_engineer.run_diagnostic_probe": (
        "REQUEST_ONE_GATED_OWNED_READ_ONLY_PROBE",
        "C/E only: select an actual experiment_hash and probe_id. The registered owned SDK provider uses one shared probe slot and must supply original source/run/container/Native/zero-command evidence. The sensor-only provider starts no physics World; the source-only Nav2 provider costs one new physics World and grants no Body admission. A new replica is independently rebound internally; the old case is not updated. Observational falsification is not causal, navigation, L1 or Memory success.",
    ),
    "files.list": (
        "READ_ONLY_ORIGINAL_FILE_CATALOG",
        "List the same original task and application inputs plus the declared staging whitelist. No Expert diagnosis, compiled repair or generated application answer is returned.",
    ),
    "files.read": (
        "READ_ONLY_ORIGINAL_FILE_SLICE",
        "Read a bounded character slice from an actual allowlisted original file, using file/offset/limit. No host paths or native filesystem access are exposed.",
    ),
    "staging.propose_patch": (
        "REGISTER_GENERIC_MODEL_AUTHORED_STAGING_ONLY",
        "Supply file/pointer/value_json changes yourself. The existing whitelist and typed field safety checks apply. No implicit diagnosis or repair compiler chooses values. Explicit independent validation is still required.",
    ),
    "staging.propose_application": (
        "REGISTER_GENERIC_MODEL_AUTHORED_APPLICATION_ONLY",
        "Supply four file strings yourself. Host checks semantic equality to the frozen Core contract with exact scalar types and list order; whitespace/key order need not match. Host seals the optional navigation_requirements_hash and serializes equivalent files. Mismatch returns no generated answer. Stage only, not SDK or Body admission.",
    ),
    "ros_engineer.read_application_input": (
        "READ_ONLY_ORIGINAL_APPLICATION_INPUT",
        "Read a bounded slice of the same frozen reusable software template, installed component manifests or application request available to the generic baseline. No graph diagnosis, proposal, World or permission is created.",
    ),
    "ros_engineer.propose_application_bootstrap": (
        "REGISTER_FOUR_FILE_APPLICATION_ONLY",
        "Compile a planned navigation application from the host-pinned current driver observations, original URDF, installed component manifests and explicit Body limits. Missing drivers/dependencies abstain. Four configuration files and rollback are proposals only; validate_proposal requires the existing fresh-World SDK and independent canonical proof.",
    ),
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
