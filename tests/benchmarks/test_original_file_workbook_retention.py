"""Cross-turn original access regressions; no model or SDK is invoked."""

import hashlib
import json

import pytest

from benchmarks.ros_expert_ablation.aci_runner import EpisodeWorkbook
from benchmarks.ros_expert_ablation.evidence_progress import EvidenceReadProgress


def catalog(text, *, scope="IDENTICAL_ORIGINAL_FILE_ACCESS_NO_DIAGNOSIS_OR_AUTHORITY"):
    return {
        "scope": scope,
        "files": [
            {
                "file": "input.json",
                "utf8_bytes": len(text.encode()),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        ],
        "staging_fields": [{"file": "input.json", "pointer": "/enabled"}],
        "current_authority": "NONE",
    }


def read(book, text, offset, limit):
    request = {"operation": "files.read", "arguments": {"file": "input.json"}}
    result = {
        "status": "ORIGINAL_FILE_SLICE_NOT_DIAGNOSIS",
        "file": "input.json",
        "offset": offset,
        "total_characters": len(text),
        "text": text[offset : offset + limit],
    }
    book.observe(result, "public inference only", request=request)
    return result


@pytest.mark.parametrize(
    "scope",
    [
        "IDENTICAL_ORIGINAL_FILE_ACCESS_NO_DIAGNOSIS_OR_AUTHORITY",
        "SAME_MODEL_GENERIC_ORIGINAL_FILES_NOT_CODEX_PRODUCT",
    ],
)
def test_catalog_and_file_content_survive_tool_switch_without_promoting_authority(scope):
    book = EpisodeWorkbook()
    text = '{"enabled":false,"说明":"原始字节"}'
    original = catalog(text, scope=scope)
    book.observe(original, "catalog", request={"operation": "files.list"})
    returned = read(book, text, 0, len(text))
    original["files"].clear()
    returned["text"] = "changed outside the workbook"
    book.observe({"status": "OTHER_RESULT"}, "model claims permission", request={})
    payload = book.payload()
    access = payload["original_evidence_access"]["original_file_access"]
    assert access["catalog"]["files"][0]["utf8_bytes"] == len(text.encode())
    assert access["catalog"]["staging_fields"][0]["pointer"] == "/enabled"
    assert access["file_slices"][0]["text"] == text
    assert access["catalog"]["current_authority"] == "NONE"
    assert payload["measured_original_facts_read"] == []
    assert payload["recent_tool_result"] == {"status": "OTHER_RESULT"}
    assert all(
        n["classification"] == "UNTRUSTED_MODEL_INFERENCE"
        for n in payload["untrusted_working_notes"]
    )
    access["file_slices"][0]["text"] = "caller mutation"
    assert (
        book.payload()["original_evidence_access"]["original_file_access"]["file_slices"][0]["text"]
        == text
    )


def test_file_ranges_merge_overlaps_and_preserve_character_not_byte_offsets():
    book = EpisodeWorkbook()
    text = "原始文件内容" * 10
    book.observe(catalog(text), "catalog", request={"operation": "files.list"})
    read(book, text, 0, 20)
    read(book, text, 10, 20)
    read(book, text, 40, 10)
    entry = book.payload()["original_evidence_access"]["progress"][0]
    assert entry["kind"] == "original_file_ranges"
    assert entry["returned_character_ranges"] == [[0, 30], [40, 50]]
    assert entry["next_unread_offset"] == 30
    assert entry["serialized_record_fully_returned"] is False
    assert (
        book.payload()["original_evidence_access"]["overlapping_serialized_characters_returned"]
        == 10
    )


def test_different_catalog_digests_and_uncatalogued_reads_never_merge():
    book = EpisodeWorkbook()
    read(book, "a" * 30, 0, 10)
    for text, offset in [("a" * 30, 10), ("b" * 30, 20)]:
        book.observe(catalog(text), "catalog", request={"operation": "files.list"})
        read(book, text, offset, 10)
    entries = book.payload()["original_evidence_access"]["progress"]
    assert len(entries) == 3
    assert {e["original_file_sha256"] for e in entries} == {
        None,
        hashlib.sha256(b"a" * 30).hexdigest(),
        hashlib.sha256(b"b" * 30).hexdigest(),
    }
    assert not any(e["serialized_record_fully_returned"] for e in entries)


def test_complete_file_hash_mismatch_is_rejected():
    book = EpisodeWorkbook()
    book.observe(catalog("original"), "catalog", request={"operation": "files.list"})
    with pytest.raises(ValueError, match="differs from its original file digest"):
        read(book, "tampered", 0, 8)


def test_empty_beyond_eof_and_empty_file_do_not_invent_ranges():
    book = EpisodeWorkbook()
    read(book, "original", 100, 10)
    entry = book.payload()["original_evidence_access"]["progress"][0]
    assert entry["returned_character_ranges"] == []
    assert entry["next_unread_offset"] == 0
    read(book, "", 100, 10)
    entry = book.payload()["original_evidence_access"]["progress"][0]
    assert entry["returned_character_ranges"] == []
    assert entry["serialized_record_fully_returned"] is True


def test_large_file_and_catalog_omissions_are_explicit_and_whole():
    book = EpisodeWorkbook()
    text = "x" * 16000
    oversized = catalog(text)
    oversized["staging_fields"] = [{"file": "input.json", "pointer": "/" + "x" * 4000}]
    book.observe(oversized, "catalog", request={"operation": "files.list"})
    read(book, text, 0, len(text))
    access = book.payload()["original_evidence_access"]["original_file_access"]
    assert access["catalog"] is None and access["catalog_omitted"]
    assert access["file_slices"] == [] and access["file_slices_not_inline"] == 1
    assert len(json.dumps(access)) < 1000


def test_original_too_large_for_recent_result_is_still_visible_after_tool_switch():
    book = EpisodeWorkbook()
    text = "parameter: false\n" * 500
    book.observe(catalog(text), "catalog", request={"operation": "files.list"})
    read(book, text, 0, len(text))
    assert book.payload()["recent_tool_result"]["details_omitted"] is True
    book.observe({"status": "OTHER_RESULT"}, "switch", request={})
    access = book.payload()["original_evidence_access"]["original_file_access"]
    assert access["file_slices"][0]["text"] == text
    assert access["file_slices_not_inline"] == 0


def test_file_slice_context_limit_counts_utf8_bytes():
    book = EpisodeWorkbook()
    text = "原" * 6000
    read(book, text, 0, len(text))
    access = book.payload()["original_evidence_access"]["original_file_access"]
    assert access["file_slices"] == [] and access["file_slices_not_inline"] == 1
    progress = book.payload()["original_evidence_access"]["progress"][0]
    # Host return coverage is distinct from inline model-context coverage.
    assert progress["serialized_record_fully_returned"] is True


def test_unsuccessful_or_mismatched_operations_do_not_claim_file_access():
    progress = EvidenceReadProgress()
    progress.observe(catalog("x"), {"operation": "other"})
    progress.observe({"status": "REJECTED_TYPED_ACI_REQUEST"}, {"operation": "files.list"})
    result = {
        "status": "ORIGINAL_FILE_SLICE_NOT_DIAGNOSIS",
        "file": "input.json",
        "offset": 0,
        "total_characters": 1,
        "text": "x",
    }
    progress.observe(result, {"operation": "files.read", "arguments": {"file": "other"}})
    assert "original_file_access" not in progress.payload()


def application_read(book, text, offset, limit, *, digest=None):
    response = {
        "status": "ORIGINAL_APPLICATION_INPUT_SLICE_NOT_DIAGNOSIS",
        "file": "input.json",
        "offset": offset,
        "total_characters": len(text),
        "text": text[offset : offset + limit],
        "source_sha256": digest or hashlib.sha256(text.encode()).hexdigest(),
    }
    book.observe(
        response,
        "public application summary",
        request={
            "operation": "ros_engineer.read_application_input",
            "arguments": {"file": "input.json"},
        },
    )


def test_application_input_survives_tool_switch_with_its_actual_host_digest():
    book = EpisodeWorkbook()
    text = '{"application_input":"' + "a" * 4100 + '"}'
    application_read(book, text, 0, len(text))
    book.observe({"status": "OTHER_RESULT"}, "switch", request={})
    access = book.payload()["original_evidence_access"]["original_file_access"]
    assert access["catalog"] is None and access["slice_reads"] == 1
    assert access["file_slices"][0]["text"] == text
    assert (
        access["file_slices"][0]["original_file_sha256"]
        == hashlib.sha256(text.encode()).hexdigest()
    )
    assert access["file_slices"][0]["file_hash_origin"] == "HOST_APPLICATION_INPUT_RESPONSE"


def test_common_file_and_application_reader_merge_only_the_same_original():
    book = EpisodeWorkbook()
    text = "same original" * 10
    book.observe(catalog(text), "catalog", request={"operation": "files.list"})
    read(book, text, 0, 20)
    application_read(book, text, 20, len(text))
    progress = book.payload()["original_evidence_access"]["progress"]
    assert len(progress) == 1
    assert progress[0]["returned_character_ranges"] == [[0, len(text)]]
    assert progress[0]["serialized_record_fully_returned"] is True


@pytest.mark.parametrize("digest", ["invalid", "A" * 64, "f" * 64])
def test_application_whole_read_requires_valid_matching_host_file_digest(digest):
    book = EpisodeWorkbook()
    with pytest.raises(ValueError, match="host file digest|differs from its original file digest"):
        application_read(book, "original", 0, 8, digest=digest)


def test_application_partial_read_cannot_change_catalog_identity():
    book = EpisodeWorkbook()
    book.observe(catalog("original"), "catalog", request={"operation": "files.list"})
    with pytest.raises(ValueError, match="differs from its original file catalog"):
        application_read(book, "other", 0, 2)
