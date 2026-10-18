"""New conditional transfer contracts only; synthetic fixtures are not SIM proof."""

import hashlib
from copy import deepcopy

import pytest

from rosclaw.connectors.ros.engineering import compile_engineering_case
from rosclaw.connectors.ros.engineering.mechanism import distill_mechanism
from rosclaw.connectors.ros.engineering.mechanism_transfer import propose_tf_mechanism_transfer
from rosclaw.connectors.ros.engineering.repair_compiler import proposal_staging
from tests.benchmarks.test_mechanism_memory import record_fixture
from tests.benchmarks.test_physical_aci import fixture, repair
from tests.connectors.ros.test_engineering_case_compiler import NOW


def historical_tf(tmp_path):
    *_, proof = record_fixture(tmp_path)
    aci = fixture("tf")
    before = deepcopy(aci.tools.staging)
    proposal = repair(aci, kind="enable_localization_tf")["proposal"]
    after = proposal_staging(proposal, before)
    return distill_mechanism(aci.case, before, after, before, proof)


def target():
    aci = fixture("tf")
    raw = aci.tools.files["robot.urdf"].replace("0.4 0.2 0.1", "0.7 0.2 0.1").encode()
    assert raw != aci.tools.files["robot.urdf"].encode()
    for description in aci.model.observations["urdf_descriptions"].values():
        description.update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    aci.model.seal()
    aci.tools.files["robot.urdf"] = raw.decode()
    return aci, raw


def compile_current(aci, raw):
    case = compile_engineering_case(
        aci.model,
        raw,
        aci.requirements,
        source_ref="fixture:different-physical-body",
        now=NOW,
        evidence_domain="SEALED_REPLAY",
    )
    return case


def test_tf_concept_crosses_body_without_copying_geometry_or_authority(tmp_path):
    record = historical_tf(tmp_path)
    aci, raw = target()
    case = compile_current(aci, raw)
    before = deepcopy(aci.tools.staging)
    plan = propose_tf_mechanism_transfer(record, case, aci.model, aci.requirements, before)
    assert plan["status"] == "CONDITIONAL_CURRENT_TF_TRANSFER_PROPOSAL"
    assert plan["cross_body_source"] is True
    assert plan["historical_parameters_reused"] is False
    assert plan["current_authority"] == "NONE"
    assert plan["current_l1"] == "NOT_EVALUATED"
    assert plan["causal_Memory_benefit"] == "NOT_ESTABLISHED"
    assert plan["proposal"]["current_authority"] == "NONE"
    assert before == aci.tools.staging


@pytest.mark.parametrize("cause", ["enabled", "inactive", "stale", "missing"])
def test_wrong_current_cause_does_not_inherit_historical_repair(tmp_path, cause):
    record = historical_tf(tmp_path)
    aci, raw = target()
    node = aci.requirements["lifecycle_nodes"]["localization"]
    if cause == "enabled":
        aci.model.observations["node_parameters"][node]["tf_broadcast"] = True
    elif cause == "inactive":
        next(row for row in aci.model.lifecycle if row.name == node).state = "INACTIVE"
    elif cause == "stale":
        aci.model.observations["parameter_captured_at"][node] = "2026-10-01T00:00:00Z"
    else:
        del aci.model.observations["node_parameters"][node]
    aci.model.seal()
    plan = propose_tf_mechanism_transfer(
        record, compile_current(aci, raw), aci.model, aci.requirements, aci.tools.staging
    )
    assert plan["status"] in {"NOT_APPLICABLE", "NEEDS_READ_ONLY_PROBE"}
    assert "proposal" not in plan
    assert plan["current_authority"] == "NONE"


def test_other_mechanisms_do_not_enter_tf_transfer_path(tmp_path):
    _, record, *_ = record_fixture(tmp_path)
    aci, raw = target()
    plan = propose_tf_mechanism_transfer(
        record, compile_current(aci, raw), aci.model, aci.requirements, aci.tools.staging
    )
    assert plan["status"] == "NOT_APPLICABLE"
