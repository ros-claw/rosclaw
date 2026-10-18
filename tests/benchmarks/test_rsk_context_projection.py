"""New context projections preserve original artifacts and authority boundaries."""

from copy import deepcopy

from benchmarks.ros_expert_ablation.aci_runner import EpisodeWorkbook
from benchmarks.ros_expert_ablation.rsk_context_projection import rsk_workbook_projection
from rosclaw.contracts.common import content_hash


def test_whole_file_proposal_context_is_bounded_without_altering_the_original():
    original = {
        "status": "CONDITIONAL_APPLICATION_STAGING_PROPOSAL_NOT_EXECUTED",
        "proposal": {
            "proposal_id": "bound-proposal",
            "changes": [
                {
                    "file": "nav2.yaml",
                    "pointer": "",
                    "operation": "REPLACE_WHOLE_ORIGINAL_FILE",
                    "before": "a" * 100000,
                    "value": "b" * 100000,
                }
            ],
            "interface_names": {"navigation.navigate_to_pose": "/new/navigate_to_pose"},
        },
    }
    saved = deepcopy(original)
    projected = rsk_workbook_projection(original)
    assert original == saved and projected is not original
    assert projected["full_original_result_hash"] == content_hash("rskfulltoolresult", original)
    c = projected["proposal"]["changes"][0]
    assert c["before"]["original_utf8_bytes"] == c["value"]["original_utf8_bytes"] == 100000
    book = EpisodeWorkbook()
    book.observe(projected, "synthetic model summary")
    assert len(str(book.payload())) < 5000
    assert book.recent["proposal_id"] == "bound-proposal"
    assert "a" * 1000 not in str(book.payload())


def test_normal_small_repair_or_original_read_remains_unchanged():
    original = {"status": "ORIGINAL_FILE_SLICE_NOT_DIAGNOSIS", "text": "actual original"}
    assert rsk_workbook_projection(original) is original


def test_diagnostic_observation_survives_old_workbook_fresh_probe_projection():
    original = {
        "status": "INDEPENDENT_ORIGINAL_PROBE_OBSERVATIONAL_REVIEW_NOT_ADMISSION",
        "fresh_probe_performed": True,
        "assessment": {
            "status": "OBSERVATIONAL_UPDATE",
            "reasons": [],
            "withdrawn_hypothesis_ids": ["qos-hypothesis"],
            "hypotheses": [
                {
                    "id": "qos-hypothesis",
                    "predicate": "QOS_CONFLICT",
                    "status": "CONTRADICTED_OBSERVATION",
                    "supporting_evidence": [],
                    "contradicting_evidence": ["real-ref"],
                }
            ],
        },
    }
    saved = deepcopy(original)
    projected = rsk_workbook_projection(original)
    book = EpisodeWorkbook()
    book.observe(projected, "synthetic")
    assert original == saved
    assert book.recent["hypotheses"][0]["status"] == "CONTRADICTED_OBSERVATION"
    assert book.recent["causal_verification"] == "NOT_ESTABLISHED"
    assert book.recent["current_case_L1"] == "NOT_EVALUATED"
    assert book.recent["current_authority"] == "NONE"
