"""Every arm has equal original facts; oversized values stay expandable."""

import json
from copy import deepcopy

import pytest

from benchmarks.ros_expert_ablation.common_evidence import encoded
from benchmarks.ros_expert_ablation.physical_aci import PhysicalACI
from benchmarks.ros_expert_ablation.task_tools import TaskTools
from tests.benchmarks.test_physical_aci import fixture


def read(aci, view, cursor=0):
    return aci.call(
        {
            "operation": "ros_engineer.read_task_evidence",
            "arguments": {
                "case_hash": aci.case.case_hash,
                "view": view,
                "cursor": cursor,
            },
        }
    )


def test_a_b_c_common_projection_is_identical_and_a_can_read_actual_repair_facts():
    projections = []
    for arm in ("A", "B", "C"):
        aci = fixture(fault="tf")
        aci.tools.arm = arm
        catalog = aci.call({"operation": "ros_engineer.inspect_task", "arguments": {}})[
            "common_evidence"
        ]
        result = read(aci, "navigation_parameters")
        projections.append((catalog, result))
        assert any(
            r.get("parameter") == "tf_broadcast" and r["value"] is False for r in result["records"]
        )
        assert not any(
            key in encoded(result)
            for key in ('"hypotheses"', '"repair_candidates"', '"fault_case_id"')
        )
    assert projections[0] == projections[1] == projections[2]


def test_actual_onboarding_interfaces_lifecycle_geometry_and_task_available_to_a():
    aci = fixture(fault="healthy")
    aci.tools.arm = "A"
    assert {r["value"]["name"] for r in read(aci, "task_interfaces")["records"]} >= {
        "/sample/go",
        "/sample/scan2",
    }
    assert read(aci, "lifecycle")["records"]
    geometry = read(aci, "body_geometry")["records"][0]
    assert geometry["classification"] == "INFERRED_OFFLINE_URDF"
    assert geometry["value"]["physical_radius_m"] == aci.tools.minimum_radius_m
    assert read(aci, "task_requirements")["records"][0]["value"] == aci.requirements


def test_common_refs_resolve_exact_original_values_and_pages_do_not_drop_records():
    aci = fixture()
    for view, expected in aci.common_evidence.groups.items():
        rows = []
        cursor = 0
        while True:
            page = read(aci, view, cursor)
            assert page["status"] == "SEALED_ORIGINAL_EVIDENCE_NOT_ADMISSION"
            assert len(encoded(page["records"])) <= 2000
            rows.extend(page["records"])
            if page["next_cursor"] is None:
                break
            assert page["next_cursor"] > cursor
            cursor = page["next_cursor"]
        assert [r["evidence_ref"] for r in rows] == [r["evidence_ref"] for r in expected]
        for record in expected:
            if record["source_ref"].startswith("sealed-model:"):
                value = aci.model.to_dict()
                for key in record["pointer"].split("/")[1:]:
                    key = key.replace("~1", "/").replace("~0", "~")
                    value = value[int(key)] if isinstance(value, list) else value[key]
                assert value == record["value"]


def test_oversized_record_is_explicitly_omitted_and_can_be_reconstructed_losslessly():
    aci = fixture()
    files = dict(aci.tools.files)
    nav = json.loads(files["nav2.yaml"])
    nav["large_original_note"] = "large original fact" * 1000
    files["nav2.yaml"] = json.dumps(nav)
    tools = TaskTools(
        files,
        arm="B",
        reviewed_at=aci.tools.reviewed_at,
        minimum_radius_m=aci.tools.minimum_radius_m,
    )
    aci = PhysicalACI(aci.case, aci.model, tools)
    entry = aci.common_evidence.groups["staging"][0]
    page = read(aci, "staging")
    assert page["records"][0]["value_omitted"] is True
    parts = []
    offset = 0
    while True:
        result = aci.call(
            {
                "operation": "ros_engineer.expand_task_evidence",
                "arguments": {
                    "case_hash": aci.case.case_hash,
                    "evidence_ref": entry["evidence_ref"],
                    "offset": offset,
                    "limit": 1600,
                },
            }
        )
        parts.append(result["text"])
        if result["next_offset"] is None:
            break
        offset = result["next_offset"]
    assert json.loads("".join(parts)) == entry


def test_reader_never_changes_staging_or_starts_validation_and_returns_copies():
    calls = []
    aci = fixture(evaluator=lambda *a, **kw: calls.append(a))
    before = deepcopy(aci.tools.staging)
    result = read(aci, "navigation_parameters")
    result["records"][0]["value"] = "tampered"
    assert read(aci, "navigation_parameters")["records"][0]["value"] != "tampered"
    assert aci.tools.staging == before and calls == [] and aci.validation_count == 0


@pytest.mark.parametrize(
    "arguments",
    [
        {"view": "private_ledger"},
        {"view": "signals", "cursor": -1},
        {"view": "signals", "cursor": True},
        {"view": "signals", "cursor": 999},
        {"view": "signals", "permit": True},
    ],
)
def test_closed_typed_views_reject_bad_or_escalating_arguments(arguments):
    aci = fixture()
    result = aci.call(
        {
            "operation": "ros_engineer.read_task_evidence",
            "arguments": {"case_hash": aci.case.case_hash, **arguments},
        }
    )
    assert result["status"] == "REJECTED_TYPED_ACI_REQUEST"


def test_changed_originals_or_stale_case_cannot_reuse_common_view():
    aci = fixture()
    aci.tools.files["robot.urdf"] += "changed"
    assert read(aci, "body_geometry")["status"] == "REJECTED_TYPED_ACI_REQUEST"


def test_mutated_compiled_evidence_is_rejected_instead_of_inheriting_original_hash():
    aci = fixture()
    aci.common_evidence.groups["signals"][0]["value"]["last_message_age_ms"] = 0
    assert read(aci, "signals")["status"] == "REJECTED_TYPED_ACI_REQUEST"


def test_absent_records_are_unknown_and_never_healthy():
    aci = fixture()
    # An exhausted page is still PRESENT; missing view facts alone are not health.
    page = read(aci, "signals", len(aci.common_evidence.groups["signals"]))
    assert page["records"] == [] and page["observation_state"] == "PRESENT"
    assert page["current_authority"] == "NONE"
