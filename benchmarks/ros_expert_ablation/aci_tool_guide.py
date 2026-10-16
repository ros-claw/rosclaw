"""Effect-aware descriptions for the operations in the current response schema.

A response schema is a type contract. This separate guide explains host effects;
it reads no robot data, adds no case answer and grants no execution authority.
"""

from benchmarks.ros_expert_ablation.aci_context import encoded

DESCRIPTIONS = {
    "ros_engineer.inspect_task": (
        "READ_ONLY_CURRENT_CARD",
        "Return the task card already present in CURRENT_STATE. No new observation or World.",
    ),
    "ros_engineer.read_task_evidence": (
        "READ_ONLY_ORIGINAL_PAGE",
        "Read one original view page. Copy next_cursor to continue that view. Repeating a page adds no new observation.",
    ),
    "ros_engineer.expand_task_evidence": (
        "READ_ONLY_ORIGINAL_SLICE",
        "Expand a returned omitted evidence_ref. Use next_unread_offset from the access workbook and limit up to 1600 characters. Partial text is not a complete value or current physical proof.",
    ),
    "ros_engineer.propose_repair": (
        "REGISTER_STAGING_ONLY",
        "Submit one typed semantic repair. The host checks original readbacks and restricted diff. Registers a proposal only; starts no World and changes no live configuration.",
    ),
    "ros_engineer.propose_compound_repair": (
        "REGISTER_STAGING_ONLY",
        "Submit 2-5 distinct typed repairs against the same original staging baseline. Host rejects overlaps or unsupported changes. No World starts.",
    ),
    "ros_engineer.select_staging_options": (
        "REGISTER_STAGING_ONLY",
        "Select distinct option_refs from CURRENT_STATE task staging_options. Host recompiles the bound original-input diffs. This is a conditional hypothesis selection, not physical admission or execution; starts no World.",
    ),
    "ros_engineer.propose_onboarding": (
        "REGISTER_STAGING_ONLY",
        "Register the four unique actually observed interface names as a proposal. No configuration repair, Body binding or World is performed.",
    ),
    "ros_engineer.propose_memory_repair": (
        "REGISTER_STAGING_ONLY",
        "Select an actually admitted applicable memory_ref. Host instantiates current-input values; historical literals and proof do not grant current authority. No World starts.",
    ),
    "ros_engineer.investigate": (
        "HOST_READ_ONLY_CHECK_MAY_START_ZERO_GOAL_REPLICA",
        "Use a named check_id. If available, host may capture a disposable zero-goal replica within preparation budget. Otherwise returns NEEDS_OPERATOR. Replica facts never refresh the original case or L1.",
    ),
    "ros_engineer.validate_proposal": (
        "EXPLICIT_HIGH_COST_FRESH_SDK_VALIDATION",
        "Request the existing host SDK to test an actual registered proposal_id in a fresh World. Requires available evaluator and remaining budget. Only full independent L1, canonical receipts, stop and source proof can establish success. No callback means rejection, not success.",
    ),
    "ros_engineer.explain_result": (
        "READ_ONLY_RECORDED_RESULT",
        "Read the recorded verdict and evidence for an actual verification_ref. Does not rerun or promote a failed result.",
    ),
    "finish": (
        "END_REPORT_ONLY",
        "End and report current evidence or missing support. Starts no World, applies no repair and never counts as physical verification. Appropriate when no supported hypothesis or required host callback is available.",
    ),
}


def build_tool_guide(schema):
    """Describe precisely the phase/arm operations present in the supplied schema."""
    definitions = schema["$defs"]
    entries = []
    for choice in schema["properties"]["decision"]["anyOf"]:
        call = definitions[choice["$ref"].rsplit("/", 1)[1]]["properties"]
        operation = call["operation"]["enum"][0]
        effect, description = DESCRIPTIONS[operation]
        arguments = call["arguments"]
        if "$ref" in arguments:
            arguments = definitions[arguments["$ref"].rsplit("/", 1)[1]]
        entries.append(
            {
                "operation": operation,
                "argument_fields": list(arguments.get("properties", {})),
                "host_effect": effect,
                "description": description,
            }
        )
    guide = {
        "scope": "EXTERNAL_HOST_OPERATIONS_NOT_NATIVE_ROS_OR_ACTION_AUTHORITY",
        "arguments_contract": "JSON object; actual current case/reference enums in response schema remain authoritative",
        "staging_vs_physics": "Host-accepted staging registration needs no current motion authority. It is an unexecuted hypothesis. Source MISMATCH, unsupported changes or unbound refs are rejected. Current physical proofs require explicit available SDK validation.",
        "operations_available_this_turn": entries,
    }
    if len(encoded(guide)) > 6500:
        raise ValueError("bounded operation guide exceeded; review new operation descriptions")
    return guide
