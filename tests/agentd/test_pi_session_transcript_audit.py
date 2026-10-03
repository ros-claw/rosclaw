"""Transcript auditing must distinguish structure from physical evidence."""

import json

import pytest

from scripts.audit_pi_session import audit_session


@pytest.mark.parametrize(
    "entry,kind",
    [
        ({"id": ["private"], "type": "message"}, "invalid_entry_id"),
        ({"id": "a", "parentId": {"private": 1}}, "invalid_parent_id"),
        ({"id": "a", "type": []}, "invalid_entry_type"),
        ({"id": "a", "message": {"role": []}}, "invalid_message_role"),
        ({"id": "a", "message": {"role": "assistant", "content": 4}}, "invalid_content"),
        ({"id": "a", "message": {"role": "assistant", "usage": "private"}}, "invalid_usage"),
        (
            {"id": "a", "message": {"role": "assistant", "usage": {"input": "private"}}},
            "invalid_usage_tokens",
        ),
        (
            {"id": "a", "message": {"role": "assistant", "usage": {"cacheRead": -1}}},
            "invalid_usage_tokens",
        ),
        ({"id": "a", "type": "context_edit", "targetId": []}, "invalid_context_target"),
    ],
)
def test_malformed_field_is_reported_without_crash_or_sensitive_value(tmp_path, entry, kind):
    report = audit_session(write_session(tmp_path / "malformed.jsonl", [entry]))
    assert report["structural_status"] == "REVIEW_REQUIRED"
    assert kind in [issue["kind"] for issue in report["structural_issues"]]
    assert "private" not in json.dumps(report)


def write_session(path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return path


def test_false_success_text_never_becomes_physical_pass(tmp_path):
    report = audit_session(
        write_session(
            tmp_path / "s.jsonl",
            [
                {
                    "id": "a",
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "stopReason": "stop",
                        "content": [
                            {"type": "text", "text": "全部 PASS，100% 成功，SECRET_SENTINEL"}
                        ],
                    },
                }
            ],
        )
    )
    assert report["completion_claim_lines_to_review"] == [1]
    assert report["semantic_and_physical_verdict"] == "NOT_EVALUATED_BY_THIS_SCRIPT"
    assert "SECRET_SENTINEL" not in json.dumps(report)


def test_corrupt_line_and_missing_parent_detected(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        "not json\n" + json.dumps({"id": "a", "parentId": "missing", "type": "message"}) + "\n"
    )
    report = audit_session(p)
    assert report["structural_status"] == "REVIEW_REQUIRED"
    assert [x["kind"] for x in report["structural_issues"]] == [
        "invalid_json",
        "missing_prior_parent",
    ]


def test_canceled_call_is_flagged_without_fabricating_failure(tmp_path):
    report = audit_session(
        write_session(
            tmp_path / "s.jsonl",
            [
                {
                    "id": "a",
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "stopReason": "toolUse",
                        "content": [
                            {
                                "type": "toolCall",
                                "id": "tool1",
                                "name": "bash",
                                "arguments": {"command": "sleep 600"},
                            }
                        ],
                    },
                },
                {
                    "id": "b",
                    "parentId": "a",
                    "type": "message",
                    "message": {"role": "assistant", "stopReason": "aborted", "content": []},
                },
            ],
        )
    )
    assert report["unmatched_tool_call_lines"] == [1]
    assert report["bash_without_explicit_timeout_lines"] == [1]
    assert report["error_or_abort_lines"] == [2]
    assert report["structural_status"] == "PASS"


def test_wrong_order_and_orphan_result_detected(tmp_path):
    report = audit_session(
        write_session(
            tmp_path / "s.jsonl",
            [
                {
                    "id": "a",
                    "type": "message",
                    "message": {"role": "toolResult", "toolCallId": "t", "content": []},
                },
                {
                    "id": "b",
                    "parentId": "a",
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "toolCall", "id": "t", "name": "read", "arguments": {}}
                        ],
                    },
                },
                {
                    "id": "c",
                    "parentId": "b",
                    "type": "message",
                    "message": {"role": "toolResult", "toolCallId": "orphan", "content": []},
                },
            ],
        )
    )
    assert report["orphan_tool_result_lines"] == [3]
    assert report["structural_issues"] == [{"line": 1, "kind": "result_before_call"}]


def test_context_accounting_and_redaction(tmp_path):
    report = audit_session(
        write_session(
            tmp_path / "s.jsonl",
            [
                {
                    "id": "a",
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "stopReason": "error",
                        "errorMessage": "SECRET_SENTINEL",
                        "usage": {"input": 4, "cacheRead": 100, "cacheWrite": 20, "output": 999},
                        "content": [],
                    },
                },
                {
                    "id": "b",
                    "parentId": "a",
                    "type": "context_edit",
                    "targetId": "a",
                    "replacement": None,
                },
            ],
        )
    )
    assert report["max_input_context_tokens"] == 124
    assert report["context_edits"][0]["target_present"] is True
    assert "SECRET_SENTINEL" not in json.dumps(report)
