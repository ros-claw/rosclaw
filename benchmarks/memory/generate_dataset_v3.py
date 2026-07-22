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
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

ZH_GESTURE = {"rock": "石头", "paper": "布", "scissors": "剪刀", "ready": "准备", "ok": "OK"}
ZH_HAND = {"left": "左手", "right": "右手"}
EN_GESTURE = {"rock": "rock", "paper": "paper", "scissors": "scissors", "ready": "ready"}


def load_memories(sqlite_path: str) -> list[dict]:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, memory_type, robot_id, body_id, practice_id, task_id,"
        " failure_type, joint_name, gesture_name, title, document, outcome,"
        " evidence_refs, event_time"
        " FROM memory_items WHERE status='active'"
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
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


def _label_for(query_mem: dict, corpus: list[dict]) -> dict[str, int]:
    """Relevance by shared failure signature (body+gesture+failure_type)."""
    labels: dict[str, int] = {}
    for mem in corpus:
        score = 0
        if mem["id"] == query_mem["id"]:
            score = 3
        else:
            same_gesture = mem.get("gesture_name") and mem.get("gesture_name") == query_mem.get(
                "gesture_name"
            )
            same_body = mem.get("body_id") and mem.get("body_id") == query_mem.get("body_id")
            same_failure = mem.get("failure_type") and mem.get("failure_type") == query_mem.get(
                "failure_type"
            )
            if same_body and same_gesture and same_failure:
                score = 3
            elif (same_gesture and same_failure) or (same_body and same_failure):
                score = 2
            elif same_failure:
                score = 1
        if score:
            labels[mem["id"]] = score
    return labels


def build_queries(memories: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    failures = [m for m in memories if m["memory_type"] == "failure"]
    episodes = [m for m in memories if m["memory_type"] == "episodic"]
    rng.shuffle(failures)
    queries: list[dict] = []

    def add(kind: str, text: str, mem: dict, forbidden: list[str] | None = None):
        queries.append(
            {
                "id": f"q_{len(queries):04d}",
                "kind": kind,
                "text": text,
                "labels": _label_for(mem, memories),
                "forbidden": forbidden or [],
                "source_memory": mem["id"],
            }
        )

    cjk_mems = [m for m in failures if m["doc_is_cjk"]]
    en_mems = [m for m in failures if not m["doc_is_cjk"]]

    # 20% CJK -> CJK
    for mem in cjk_mems[:60]:
        hand = _hand_of(mem) or "right"
        gesture = (mem.get("gesture_name") or "rock").replace("left_", "")
        add(
            "cjk_to_cjk",
            f"{ZH_HAND.get(hand, hand)}{ZH_GESTURE.get(gesture, gesture)}手势失败原因",
            mem,
        )

    # 15% CJK -> EN (queries in Chinese over English-document memories)
    for mem in (en_mems or cjk_mems)[:45]:
        add("cjk_to_en", "关节未到位 位置跟踪失败", mem)

    # 15% EN -> CJK
    for mem in cjk_mems[60:105]:
        hand = _hand_of(mem) or "right"
        gesture = (mem.get("gesture_name") or "rock").replace("left_", "")
        add(
            "en_to_cjk",
            f"{hand} hand {EN_GESTURE.get(gesture, gesture)} gesture failed joint not reached",
            mem,
        )

    # 15% mixed
    for mem in failures[:45]:
        hand = _hand_of(mem) or "right"
        add(
            "mixed",
            f"RH56 {ZH_HAND.get(hand, hand)} {mem.get('failure_type') or 'joint_not_reached'} 失败",
            mem,
        )

    # 15% error codes / symbols
    error_queries = [
        ("EIO -110 串口", "serial_io_error"),
        ("camera wedge 无帧", "camera_wedge"),
        ("joint_not_reached 未到位", "joint_not_reached"),
        ("overcurrent 过流", "overcurrent"),
        ("USB disconnect 掉线", "usb_disconnect"),
    ]
    for i in range(45):
        text, _ = error_queries[i % len(error_queries)]
        mem = failures[i % len(failures)] if failures else episodes[0]
        add("error_code", text, mem)

    # 10% same symptom different root cause
    for mem in failures[45:75]:
        hand = _hand_of(mem) or "right"
        add(
            "same_symptom_diff_cause",
            f"{hand} 手位置跟踪失败是温度原因还是机械原因",
            mem,
        )

    # 10% hard negatives (joint / body confusion)
    for mem in failures[75:105]:
        hand = _hand_of(mem)
        other_body = "rh56_left_01" if hand == "left" else "rh56_right_01"
        forbidden = [m["id"] for m in failures if m.get("body_id") == other_body][:5]
        gesture = (mem.get("gesture_name") or "rock").replace("left_", "")
        add(
            "hard_negative_body",
            f"{ZH_HAND.get(hand, '右手')} {ZH_GESTURE.get(gesture, gesture)} 未到位",
            mem,
            forbidden=forbidden,
        )
    return queries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    memories = load_memories(args.sqlite)
    print(f"corpus: {len(memories)} memories")
    queries = build_queries(memories)
    print(f"queries: {len(queries)}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "dataset.jsonl").open("w") as fh:
        for mem in memories:
            fh.write(json.dumps(mem, ensure_ascii=False) + "\n")
    with (out / "queries.jsonl").open("w") as fh:
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
