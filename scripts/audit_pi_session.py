#!/usr/bin/env python3
"""Read every PI JSONL entry and audit transcript structure without emitting secrets.

This is a historical transcript audit, not an execution or physical-success oracle.
Canceled, retried and branched histories can legitimately have unmatched tools;
these are reported for review rather than labeled as physical failures.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


def audit_session(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    entries: list[tuple[int, dict[str, Any]]] = []
    issues: list[dict[str, Any]] = []
    ids: set[str] = set()
    types: collections.Counter[str] = collections.Counter()
    roles: collections.Counter[str] = collections.Counter()
    tools: collections.Counter[str] = collections.Counter()
    calls: dict[str, list[int]] = collections.defaultdict(list)
    results: dict[str, list[int]] = collections.defaultdict(list)
    stops: collections.Counter[str] = collections.Counter()
    user_lines: list[int] = []
    claim_lines: list[int] = []
    error_lines: list[int] = []
    bash_without_timeout: list[int] = []
    context_edits: list[dict[str, Any]] = []
    max_context_tokens = 0
    for line, source in enumerate(raw.splitlines(), 1):
        try:
            entry = json.loads(source)
        except (json.JSONDecodeError, UnicodeDecodeError):
            issues.append({"line": line, "kind": "invalid_json"})
            continue
        if not isinstance(entry, dict):
            issues.append({"line": line, "kind": "non_object_entry"})
            continue
        entries.append((line, entry))
        entry_id = entry.get("id")
        if entry_id:
            if entry_id in ids:
                issues.append({"line": line, "kind": "duplicate_entry_id"})
            parent = entry.get("parentId")
            if parent and parent not in ids:
                issues.append({"line": line, "kind": "missing_prior_parent"})
            ids.add(entry_id)
        kind = entry.get("type", "unknown")
        types[kind] += 1
        if kind == "context_edit":
            context_edits.append(
                {
                    "line": line,
                    "target_present": entry.get("targetId") in ids,
                    "replacement_removed": entry.get("replacement") is None,
                }
            )
        message = entry.get("message") or {}
        if not isinstance(message, dict):
            issues.append({"line": line, "kind": "invalid_message"})
            continue
        role = message.get("role")
        if role:
            roles[role] += 1
        if role == "user":
            user_lines.append(line)
        if role == "assistant":
            stops[str(message.get("stopReason", "unknown"))] += 1
            if message.get("stopReason") in ("error", "aborted"):
                error_lines.append(line)
            usage = message.get("usage") or {}
            # Output is not part of the incoming context.
            max_context_tokens = max(
                max_context_tokens,
                int(usage.get("input", 0))
                + int(usage.get("cacheRead", 0))
                + int(usage.get("cacheWrite", 0)),
            )
        if role == "toolResult":
            results[str(message.get("toolCallId", ""))].append(line)
        content = message.get("content") or []
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for block in content:
            if not isinstance(block, dict):
                issues.append({"line": line, "kind": "non_object_content"})
                continue
            if block.get("type") == "toolCall":
                name = str(block.get("name", "unknown"))
                tools[name] += 1
                calls[str(block.get("id", ""))].append(line)
                args = block.get("arguments") or {}
                if (
                    name == "bash"
                    and isinstance(args, dict)
                    and not any(key in args for key in ("timeout_sec", "timeout", "timeoutMs"))
                ):
                    bash_without_timeout.append(line)
            elif role == "assistant" and block.get("type") == "text":
                text = str(block.get("text", ""))
                if any(
                    marker in text for marker in ("全部 PASS", "全 PASS", "100% 成功", "全部通过")
                ):
                    claim_lines.append(line)
    unmatched_calls = sorted(
        {line for key, lines in calls.items() if key not in results for line in lines}
    )
    orphan_results = sorted(
        {line for key, lines in results.items() if key not in calls for line in lines}
    )
    for key, lines in results.items():
        if key in calls and min(lines) < min(calls[key]):
            issues.append({"line": min(lines), "kind": "result_before_call"})
    return {
        "schema_version": "rosclaw.pi_session_audit.v1",
        "source": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "parsed_entries": len(entries),
        "entry_types": dict(types),
        "message_roles": dict(roles),
        "tool_calls": dict(tools),
        "assistant_stop_reasons": dict(stops),
        "max_input_context_tokens": max_context_tokens,
        "user_message_lines": user_lines,
        "error_or_abort_lines": error_lines,
        "completion_claim_lines_to_review": sorted(set(claim_lines)),
        "bash_without_explicit_timeout_lines": sorted(set(bash_without_timeout)),
        "unmatched_tool_call_lines": unmatched_calls,
        "orphan_tool_result_lines": orphan_results,
        "reused_tool_call_id_count": sum(len(lines) > 1 for lines in calls.values()),
        "context_edits": context_edits,
        "structural_issues": issues,
        "structural_status": "PASS" if not issues else "REVIEW_REQUIRED",
        "semantic_and_physical_verdict": "NOT_EVALUATED_BY_THIS_SCRIPT",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = audit_session(args.session)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "entries": report["parsed_entries"],
                "structural_status": report["structural_status"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["structural_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
