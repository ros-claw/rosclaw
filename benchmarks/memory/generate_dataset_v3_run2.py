#!/usr/bin/env python3
"""v3_run2 dataset builder — uses the REVIEWED labeler from
benchmarks/memory/generate_dataset_v3.py, but emits only the lanes that
are honestly constructible on the PR-MEM-3 corpus.

Honest corpus fact (measured): after PR-MEM-3's bilingual document
builder, every failure memory contains a [ZH] section, so the
"cjk_to_en" lane (CJK query over pure-EN memories) has ZERO valid
sources on this corpus.  The lane is recorded as not_applicable with
the reason instead of being filled with fallback rows (the reviewed
generator hard-fails, correctly, on the same fact).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path("/home/nvidia/workspace/rosclaw/rosclaw_test/rosclaw")
sys.path.insert(0, str(REPO / "benchmarks" / "memory"))

from generate_dataset_v3 import (  # noqa: E402
    EN_GESTURE,
    ZH_GESTURE,
    ZH_HAND,
    _hand_of,
    _label_for,
    load_memories,
)

LANES = {
    "cjk_to_cjk": 60,
    "en_to_cjk": 45,
    "mixed": 45,
    "error_code": 45,
    "same_symptom_diff_cause": 30,
    "hard_negative_body": 30,
    "cross_lingual_bilingual": 45,  # EN query over bilingual [ZH]+[EN] docs
}
NOT_APPLICABLE = {
    "cjk_to_en": (
        "PR-MEM-3 bilingual builder: every failure memory carries a [ZH] "
        "section, so pure-EN source memories do not exist on this corpus "
        "(measured 0/523). Lane reported N/A rather than filled with "
        "fallback rows."
    )
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    memories = load_memories(args.sqlite)
    failures = [m for m in memories if m["memory_type"] == "failure"]
    structured = [
        m
        for m in failures
        if _hand_of(m) in ("left", "right") and m.get("gesture_name") and m.get("failure_type")
    ]
    print(f"corpus={len(memories)} failure={len(failures)} structured={len(structured)}")

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

    import random

    rng = random.Random(42)
    rng.shuffle(structured)

    for mem in structured[: LANES["cjk_to_cjk"]]:
        hand = _hand_of(mem) or "right"
        gesture = (mem.get("gesture_name") or "rock").replace("left_", "")
        add(
            "cjk_to_cjk",
            f"{ZH_HAND.get(hand, hand)}{ZH_GESTURE.get(gesture, gesture)}手势失败原因",
            mem,
        )

    for mem in structured[60 : 60 + LANES["en_to_cjk"]]:
        hand = _hand_of(mem) or "right"
        gesture = (mem.get("gesture_name") or "rock").replace("left_", "")
        add(
            "en_to_cjk",
            f"{hand} hand {EN_GESTURE.get(gesture, gesture)} gesture failed joint not reached",
            mem,
        )

    for mem in structured[105 : 105 + LANES["mixed"]]:
        hand = _hand_of(mem) or "right"
        add(
            "mixed",
            f"RH56 {ZH_HAND.get(hand, hand)} {mem.get('failure_type') or 'joint_not_reached'} 失败",
            mem,
        )

    error_queries = [
        "EIO -110 串口",
        "camera wedge 无帧",
        "joint_not_reached 未到位",
        "overcurrent 过流",
        "USB disconnect 掉线",
    ]
    for i in range(LANES["error_code"]):
        add("error_code", error_queries[i % len(error_queries)], structured[i % len(structured)])

    for mem in structured[150 : 150 + LANES["same_symptom_diff_cause"]]:
        hand = _hand_of(mem) or "right"
        add("same_symptom_diff_cause", f"{hand} 手位置跟踪失败是温度原因还是机械原因", mem)

    for mem in structured[180 : 180 + LANES["hard_negative_body"]]:
        hand = _hand_of(mem)
        # The FORBIDDEN set must be the OPPOSITE body (run1's builder had
        # this ternary inverted, which made its confusion=0 metric
        # meaningless — same-body "forbidden" is trivially never violated).
        other_body = "rh56_right_01" if hand == "left" else "rh56_left_01"
        expected_body = f"rh56_{hand}_01"
        forbidden = [m["id"] for m in structured if m.get("body_id") == other_body][:5]
        gesture = (mem.get("gesture_name") or "rock").replace("left_", "")
        # Coherent with the reviewed validator: a hard-negative query's
        # labels must NOT include the forbidden opposite-body rows — so
        # labels are restricted to the intended body.
        body_ids = {m["id"] for m in memories if m.get("body_id") == expected_body}
        queries.append(
            {
                "id": f"q_{len(queries):04d}",
                "kind": "hard_negative_body",
                "text": f"{ZH_HAND.get(hand, '右手')} {ZH_GESTURE.get(gesture, gesture)} 未到位",
                "labels": {k: v for k, v in _label_for(mem, memories).items() if k in body_ids},
                "forbidden": forbidden,
                "source_memory": mem["id"],
            }
        )

    for mem in structured[210 : 210 + LANES["cross_lingual_bilingual"]]:
        hand = _hand_of(mem) or "right"
        gesture = (mem.get("gesture_name") or "rock").replace("left_", "")
        add(
            "cross_lingual_bilingual",
            f"why did the {hand} hand fail {EN_GESTURE.get(gesture, gesture)} at high temperature",
            mem,
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "dataset.jsonl").open("w") as fh:
        for mem in memories:
            fh.write(json.dumps(mem, ensure_ascii=False) + "\n")
    with (out / "queries.jsonl").open("w") as fh:
        for query in queries:
            fh.write(json.dumps(query, ensure_ascii=False) + "\n")
    (out / "LANES.json").write_text(
        json.dumps(
            {"lanes": dict(LANES), "not_applicable": NOT_APPLICABLE}, indent=1, ensure_ascii=False
        )
    )
    no_labels = sum(1 for q in queries if not q["labels"])
    print(f"queries={len(queries)} no_labels={no_labels}")
    print("not_applicable:", list(NOT_APPLICABLE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
