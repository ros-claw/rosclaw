"""Pure report assertions and proposal-only effects remain separate from success."""

import pytest

from benchmarks.ros_expert_ablation.closeout_aci import CloseoutACI
from tests.benchmarks.test_physical_aci import fixture


def envelope(operation, arguments):
    return {
        "decision": {"operation": operation, "arguments": arguments},
        "decision_summary": "Evidence-based public report",
        "memory_refs_used": [],
    }


def report(aci, **overrides):
    args = {
        "case_hash": aci.base.case.case_hash,
        "decision": "NEEDS_EVIDENCE",
        "reason": "MISSING_OR_STALE_EVIDENCE",
        "evidence_refs": [next(iter(aci.visible_refs))],
        "proposal_id": None,
    }
    args.update(overrides)
    return envelope("ros_engineer.report_decision", args)


def test_report_is_terminal_unverified_model_assertion_and_exposes_no_gold():
    aci = CloseoutACI(fixture("tf", max_validations=1))
    schema = aci.decision_schema()
    operations = {
        schema["$defs"][c["$ref"].split("/")[-1]]["properties"]["operation"]["enum"][0]
        for c in schema["properties"]["decision"]["anyOf"]
    }
    assert "ros_engineer.report_decision" in operations
    assert "ros_engineer.validate_proposal" not in operations
    result = aci.call_model_decision(report(aci))
    assert result["status"] == "MODEL_REPORT_NOT_VERIFICATION"
    assert result["verified_success"] is False and result["current_authority"] == "NONE"
    assert not aci.base.proposals and aci.base.validation_count == aci.base.probe_count == 0
    with pytest.raises(ValueError, match="terminal"):
        aci.call_model_decision(report(aci))


@pytest.mark.parametrize("fault", ["forged_ref", "forged_proposal", "stale_case", "extra_permit"])
def test_forged_reports_or_permission_fields_fail_without_dispatch(fault):
    aci = CloseoutACI(fixture("tf", max_validations=1))
    value = report(aci)
    args = value["decision"]["arguments"]
    if fault == "forged_ref":
        args["evidence_refs"] = ["invented"]
    elif fault == "forged_proposal":
        args.update(decision="PROPOSE", proposal_id="invented")
    elif fault == "stale_case":
        args["case_hash"] = "stale"
    else:
        args["permit"] = True
    with pytest.raises(ValueError):
        aci.call_model_decision(value)
    assert not aci.trace and aci.report is None and not aci.base.proposals


def test_actual_typed_proposal_and_report_never_apply_or_validate():
    base = fixture("tf", max_validations=1)
    aci = CloseoutACI(base)
    original = dict(base.tools.files)
    result = aci.call_model_decision(
        envelope(
            "ros_engineer.propose_repair",
            {"case_hash": base.case.case_hash, "repair": {"kind": "enable_localization_tf"}},
        )
    )
    proposal = result["proposal"]
    aci.call_model_decision(
        report(
            aci,
            decision="PROPOSE",
            reason="CONDITIONAL_PROPOSAL",
            proposal_id=proposal["proposal_id"],
        )
    )
    assert proposal["stage_only"] and proposal["admission"] == "NOT_ADMITTED"
    assert base.tools.files == original and base.validation_count == base.probe_count == 0


@pytest.mark.parametrize("callback", ["evaluator", "probe_evaluator"])
def test_callback_injection_after_creation_cannot_execute(callback):
    base = fixture("tf", max_validations=1)
    aci = CloseoutACI(base)
    setattr(base, callback, lambda *a, **kw: pytest.fail("no physical callback"))
    with pytest.raises(ValueError, match="cannot attach"):
        aci.call_model_decision(report(aci))


@pytest.mark.parametrize(
    "operation",
    ["ros_engineer.validate_proposal", "ros_engineer.investigate", "ros2", "request_action"],
)
def test_report_facade_does_not_create_an_execution_or_probe_entry(operation):
    aci = CloseoutACI(fixture("tf", max_validations=1))
    with pytest.raises(ValueError, match="allowlist"):
        aci.call_model_decision(envelope(operation, {}))
