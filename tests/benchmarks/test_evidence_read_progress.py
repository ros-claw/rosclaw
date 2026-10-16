"""Access metadata tests; synthetic inputs are not actual model/World evidence."""

import json

from benchmarks.ros_expert_ablation.aci_runner import EpisodeWorkbook
from benchmarks.ros_expert_ablation.evidence_progress import EvidenceReadProgress
from tests.benchmarks.test_physical_aci import fixture


def slice_result(offset, text, *, ref="evidence", digest="original", total=100):
    return {
        "status": "ORIGINAL_SERIALIZED_RECORD_SLICE_NOT_NEW_OBSERVATION",
        "evidence_ref": ref,
        "record_sha256": digest,
        "offset": offset,
        "text": text,
        "total_characters": total,
    }


def test_actual_page_cursor_survives_another_tool_and_repeat_is_counted():
    aci = fixture()
    book = EpisodeWorkbook()
    request = {
        "operation": "ros_engineer.read_task_evidence",
        "arguments": {"case_hash": aci.case.case_hash, "view": "navigation_parameters"},
    }
    original = aci.call(request)
    book.observe(original, "read", request=request)
    book.observe({"status": "OTHER_RESULT"}, "not evidence", request={})
    book.observe(original, "repeat", request=request)
    access = book.payload()["original_evidence_access"]
    assert access["repeated_page_reads"] == 1 and access["page_reads"] == 2
    assert access["progress"][0]["next_cursor"] == original["next_cursor"]
    assert access["progress"][0]["returned_records"] == len(original["records"])
    assert aci.validation_count == 0 and not aci.proposals


def test_expansion_tracks_union_and_first_unread_gap_without_claiming_full_value():
    progress = EvidenceReadProgress()
    for offset, text in [(20, "b" * 20), (0, "a" * 10), (5, "a" * 5)]:
        progress.observe(slice_result(offset, text), {})
    result = progress.payload()
    entry = result["progress"][0]
    assert entry["returned_character_ranges"] == [[0, 10], [20, 40]]
    assert entry["next_unread_offset"] == 10
    assert entry["serialized_record_fully_returned"] is False
    assert result["overlapping_serialized_characters_returned"] == 5
    progress.observe(slice_result(10, "c" * 90), {})
    entry = progress.payload()["progress"][0]
    assert entry["returned_character_ranges"] == [[0, 100]]
    assert entry["next_unread_offset"] is None
    assert entry["serialized_record_fully_returned"] is True
    assert "value" not in entry and "authority" not in entry


def test_original_slice_text_survives_tool_switch_and_is_copied():
    book = EpisodeWorkbook()
    result = slice_result(0, '{"unfinished":')
    book.observe(result, "not a parsed value", request={})
    result["text"] = "MUTATED"
    book.observe({"status": "OTHER_RESULT"}, "switch", request={})
    access = book.payload()["original_evidence_access"]
    assert access["original_serialized_slices"][0]["text"] == '{"unfinished":'
    assert "NOT_COMPLETE_VALUE" in access["original_serialized_slices"][0]["scope"]
    assert book.payload()["measured_original_facts_read"] == []


def test_different_original_record_hashes_never_merge_coverage():
    progress = EvidenceReadProgress()
    progress.observe(slice_result(0, "a" * 50, digest="first"), {})
    progress.observe(slice_result(50, "b" * 50, digest="second"), {})
    entries = progress.payload()["progress"]
    assert len(entries) == 2
    assert all(not entry["serialized_record_fully_returned"] for entry in entries)
    assert entries[0]["next_unread_offset"] == 50
    assert entries[1]["next_unread_offset"] == 0


def test_whole_progress_rows_and_slices_are_bounded_and_omissions_explicit():
    progress = EvidenceReadProgress()
    for i in range(12):
        progress.observe(slice_result(0, "x" * 1600, ref=str(i), total=2000), {})
    access = progress.payload()
    assert len(json.dumps(access["progress"], separators=(",", ":"))) <= 2000
    assert len(json.dumps(access["original_serialized_slices"], separators=(",", ":"))) <= 2000
    assert access["progress_entries_not_inline"] > 0
    assert access["slice_entries_not_inline"] > 0
    assert all(len(row["text"]) == 1600 for row in access["original_serialized_slices"])


def test_rejected_requests_do_not_create_read_progress():
    progress = EvidenceReadProgress()
    progress.observe({"status": "REJECTED_TYPED_ACI_REQUEST", "error": "missing original"}, {})
    assert progress.payload()["progress"] == [] and progress.payload()["expand_reads"] == 0


def test_metadata_never_replaces_raw_tool_result_or_model_notes():
    book = EpisodeWorkbook()
    book.observe(slice_result(0, "original bytes"), "model says permitted", request={})
    payload = book.payload()
    assert payload["recent_tool_result"]["text"] == "original bytes"
    assert payload["untrusted_working_notes"][0]["classification"] == "UNTRUSTED_MODEL_INFERENCE"
    assert "NOT_LIVE_STATE_OR_AUTHORITY" in payload["original_evidence_access"]["scope"]
