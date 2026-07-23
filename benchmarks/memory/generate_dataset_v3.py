#!/usr/bin/env python3
"""Generate the expanded v3 benchmark dataset (数据库优化v3 §10.2).

Corpus: active memory_items from a distilled knowledge sqlite
(≥1000 after full-history distillation).

Queries: ≥300 with structured labels, in the §10.2 proportions:

    20% CJK query -> CJK memory
    15% CJK query -> EN memory
    15% EN query -> CJK memory
    15% 中英混合查询
    15% 机器人错误码/代码符号
    10% 同症状不同根因
    10% Hard Negative（middle vs thumb_rot, left vs right）

Labels are derived from STRUCTURED fields (body_id / gesture_name /
failure_type / memory_type) — never from a model's own ranking output,
so the benchmark measures the world, not the model's agreement with
itself.  Hard-negative queries carry a ``forbidden`` list: memories
that must NOT rank top-1 (wrong joint/body).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

ZH_GESTURE = {"rock": "石头", "paper": "布", "scissors": "剪刀", "ready": "准备", "ok": "OK"}
ZH_HAND = {"left": "左手", "right": "右手"}
EN_GESTURE = {"rock": "rock", "paper": "paper", "scissors": "scissors", "ready": "ready"}
EXPECTED_KIND_COUNTS = {
    "cjk_to_cjk": 60,
    "cjk_to_en": 45,
    "en_to_cjk": 45,
    "mixed": 45,
    "error_code": 45,
    "same_symptom_diff_cause": 30,
    "hard_negative_body": 30,
}


def load_memories(sqlite_path: str) -> list[dict]:
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, memory_type, robot_id, body_id, practice_id, task_id,"
            " failure_type, joint_name, gesture_name, title, document, outcome,"
            " evidence_refs, metadata, event_time"
            " FROM memory_items WHERE status='active'"
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        metadata = d.pop("metadata", None)
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        if isinstance(metadata, dict):
            d["root_cause"] = metadata.get("root_cause")
        d["doc_is_cjk"] = any("一" <= ch <= "鿿" for ch in (d.get("document") or ""))
        out.append(d)
    return out


def _hand_of(mem: dict) -> str | None:
    body = mem.get("body_id") or ""
    if "left" in body:
        return "left"
    if "right" in body:
        return "right"
    return None


def _label_for(
    query_mem: dict,
    corpus: list[dict],
    *,
    fields: tuple[str, ...] = ("body_id", "gesture_name", "failure_type"),
) -> dict[str, int]:
    """Relevance by an explicit structured signature.

    A missing field is not silently treated as a match.  This prevents a
    left-hand query from labeling right-hand rows relevant merely because
    both contain ``joint_not_reached``.
    """
    required = {field: query_mem.get(field) for field in fields}
    if any(value in (None, "") for value in required.values()):
        raise ValueError(f"source {query_mem.get('id')} lacks label fields {required}")
    labels: dict[str, int] = {}
    for mem in corpus:
        if all(mem.get(field) == value for field, value in required.items()):
            labels[mem["id"]] = 3 if mem["id"] == query_mem["id"] else 2
    return labels


def build_queries(memories: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    failures = [m for m in memories if m["memory_type"] == "failure"]
    rng.shuffle(failures)
    queries: list[dict] = []

    def require(lane: str, rows: list[Any], count: int) -> list[Any]:
        if len(rows) < count:
            raise ValueError(
                f"lane {lane!r} requires {count} valid source memories, found {len(rows)}; "
                "do not fill a benchmark lane with unrelated fallback rows"
            )
        return rows[:count]

    def add(
        kind: str,
        text: str,
        mem: dict,
        *,
        labels: dict[str, int] | None = None,
        label_fields: tuple[str, ...] = ("body_id", "gesture_name", "failure_type"),
        forbidden: list[str] | None = None,
    ) -> None:
        relevance = labels if labels is not None else _label_for(mem, memories, fields=label_fields)
        if mem["id"] not in relevance:
            raise ValueError(
                f"source memory {mem['id']} is not relevant to generated query {text!r}"
            )
        queries.append(
            {
                "id": f"q_{len(queries):04d}",
                "kind": kind,
                "text": text,
                "labels": relevance,
                "forbidden": forbidden or [],
                "source_memory": mem["id"],
            }
        )

    structured = [
        mem
        for mem in failures
        if _hand_of(mem) in ("left", "right")
        and mem.get("gesture_name")
        and mem.get("failure_type")
    ]
    cjk_mems = [m for m in structured if m["doc_is_cjk"]]
    en_mems = [m for m in structured if not m["doc_is_cjk"]]

    # 20% CJK -> CJK
    for mem in require("cjk_to_cjk", cjk_mems, 60):
        hand = _hand_of(mem) or "right"
        gesture = (mem.get("gesture_name") or "rock").replace("left_", "")
        add(
            "cjk_to_cjk",
            f"{ZH_HAND.get(hand, hand)}{ZH_GESTURE.get(gesture, gesture)}手势失败原因",
            mem,
        )

    # 15% CJK -> EN (queries in Chinese over English-document memories)
    for mem in require("cjk_to_en", en_mems, 45):
        hand = _hand_of(mem) or "right"
        gesture = (mem.get("gesture_name") or "rock").replace("left_", "")
        add(
            "cjk_to_en",
            f"{ZH_HAND[hand]}{ZH_GESTURE.get(gesture, gesture)} {mem['failure_type']}",
            mem,
        )

    # 15% EN -> CJK
    for mem in require("en_to_cjk", cjk_mems[60:], 45):
        hand = _hand_of(mem) or "right"
        gesture = (mem.get("gesture_name") or "rock").replace("left_", "")
        add(
            "en_to_cjk",
            f"{hand} hand {EN_GESTURE.get(gesture, gesture)} gesture failed joint not reached",
            mem,
        )

    # 15% mixed
    for mem in require("mixed", structured[105:], 45):
        hand = _hand_of(mem) or "right"
        add(
            "mixed",
            f"RH56 {ZH_HAND.get(hand, hand)} {mem.get('failure_type') or 'joint_not_reached'} 失败",
            mem,
        )

    # 15% error codes / symbols
    error_queries = [
        ("EIO -110 串口", re.compile(r"\bEIO\b|-110", re.I)),
        ("camera wedge 无帧", re.compile(r"camera[_ -]?wedge|camera.*无帧", re.I)),
        ("joint_not_reached 未到位", re.compile(r"joint_not_reached", re.I)),
        ("overcurrent 过流", re.compile(r"overcurrent|过流", re.I)),
        ("USB disconnect 掉线", re.compile(r"usb[_ -]?disconnect|USB.*掉线", re.I)),
    ]
    for i in range(45):
        text, pattern = error_queries[i % len(error_queries)]
        matches = [
            mem
            for mem in failures
            if pattern.search(f"{mem.get('title', '')} {mem.get('document', '')}")
        ]
        if not matches:
            raise ValueError(f"lane 'error_code' has no real memory matching {pattern.pattern!r}")
        mem = matches[(i // len(error_queries)) % len(matches)]
        labels = {
            candidate["id"]: (3 if candidate["id"] == mem["id"] else 2) for candidate in matches
        }
        add("error_code", text, mem, labels=labels)

    # 10% same symptom different root cause
    causal = [mem for mem in structured if mem.get("root_cause")]
    causal_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for mem in causal:
        causal_groups[(str(mem.get("body_id")), str(mem.get("failure_type")))].add(
            str(mem["root_cause"])
        )
    causal = [
        mem
        for mem in causal
        if len(causal_groups[(str(mem.get("body_id")), str(mem.get("failure_type")))]) >= 2
    ]
    for mem in require("same_symptom_diff_cause", causal, 30):
        hand = _hand_of(mem) or "right"
        add(
            "same_symptom_diff_cause",
            f"{hand} hand {mem['failure_type']} root cause {mem['root_cause']}",
            mem,
            label_fields=("body_id", "failure_type", "root_cause"),
            forbidden=[
                candidate["id"]
                for candidate in causal
                if candidate.get("body_id") == mem.get("body_id")
                and candidate.get("failure_type") == mem.get("failure_type")
                and candidate.get("root_cause") != mem.get("root_cause")
            ],
        )

    # 10% hard negatives (joint / body confusion)
    hard_negative_sources = []
    for mem in structured:
        hand = _hand_of(mem)
        other_body = "rh56_right_01" if hand == "left" else "rh56_left_01"
        forbidden = [
            candidate["id"]
            for candidate in structured
            if candidate.get("body_id") == other_body
            and candidate.get("gesture_name") == mem.get("gesture_name")
            and candidate.get("failure_type") == mem.get("failure_type")
        ][:5]
        if forbidden:
            hard_negative_sources.append((mem, forbidden))
    for mem, forbidden in require("hard_negative_body", hard_negative_sources, 30):
        hand = _hand_of(mem)
        gesture = (mem.get("gesture_name") or "rock").replace("left_", "")
        add(
            "hard_negative_body",
            f"{ZH_HAND.get(hand, '右手')} {ZH_GESTURE.get(gesture, gesture)} 未到位",
            mem,
            forbidden=forbidden,
        )
    if len(queries) != 300:
        raise AssertionError(f"expected 300 queries, generated {len(queries)}")
    validate_queries(memories, queries)
    validate_benchmark_shape(queries)
    return queries


def validate_queries(memories: list[dict], queries: list[dict]) -> None:
    """Reject self-inconsistent labels before an expensive model run."""
    by_id = {mem["id"]: mem for mem in memories}
    if len(by_id) != len(memories):
        raise ValueError("corpus memory IDs must be unique")
    query_ids = [query.get("id") for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique")
    for query in queries:
        source_id = query["source_memory"]
        if source_id not in by_id or query["labels"].get(source_id, 0) <= 0:
            raise ValueError(f"query {query['id']} has an invalid source/label binding")
        unknown_labels = set(query["labels"]) - set(by_id)
        if unknown_labels:
            raise ValueError(f"query {query['id']} labels unknown rows: {unknown_labels}")
        forbidden = set(query.get("forbidden") or [])
        unknown_forbidden = forbidden - set(by_id)
        if unknown_forbidden:
            raise ValueError(
                f"query {query['id']} references unknown forbidden rows: {unknown_forbidden}"
            )
        overlap = forbidden & set(query["labels"])
        if overlap:
            raise ValueError(f"query {query['id']} labels forbidden rows as relevant: {overlap}")
        if query["kind"] == "hard_negative_body":
            source_body = by_id[source_id].get("body_id")
            if any(by_id[row_id].get("body_id") == source_body for row_id in query["forbidden"]):
                raise ValueError(
                    f"query {query['id']} hard negatives are not from the opposite body"
                )


def validate_benchmark_shape(
    queries: list[dict], expected: dict[str, int] | None = None
) -> None:
    """Enforce the declared lane distribution.

    ``expected`` defaults to EXPECTED_KIND_COUNTS; a dataset dir may carry
    a ``LANES.json`` sidecar declaring an honest variant (e.g. a lane
    reported not_applicable on a bilingual corpus).
    """
    counts: dict[str, int] = defaultdict(int)
    for query in queries:
        counts[str(query.get("kind"))] += 1
    want = expected if expected is not None else EXPECTED_KIND_COUNTS
    if dict(counts) != want:
        raise ValueError(
            f"invalid benchmark lane distribution: expected {want}, got {dict(counts)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    os.umask(0o077)

    out = Path(args.out_dir).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if out.is_relative_to(repo_root):
        parser.error("--out-dir must point outside the source repository")

    memories = load_memories(args.sqlite)
    print(f"corpus: {len(memories)} memories")
    queries = build_queries(memories)
    print(f"queries: {len(queries)}")

    out.mkdir(parents=True, exist_ok=True)
    with (out / "dataset.jsonl").open("w", encoding="utf-8") as fh:
        for mem in memories:
            fh.write(json.dumps(mem, ensure_ascii=False) + "\n")
    with (out / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for query in queries:
            fh.write(json.dumps(query, ensure_ascii=False) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for query in queries:
        counts[query["kind"]] += 1
    print("kinds:", dict(counts))
    no_labels = sum(1 for query in queries if not query["labels"])
    print(f"queries without labels: {no_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
