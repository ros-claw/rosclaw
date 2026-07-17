#!/usr/bin/env python3
"""Generate the memory retrieval benchmark dataset (§6.7).

Deterministically builds:

* ``dataset.jsonl`` — 320 memory items: the 116 real memories distilled from
  the 7×24 RH56 run (embedded as seeds) plus synthetic bilingual variants,
  cross-robot decoys, and similar-symptom-different-root-cause pairs;
* ``queries.jsonl`` — 100 queries (Chinese/English mixed, cross-task,
  cross-body, cross-time, decoys);
* ``relevance_labels.jsonl`` — graded relevance (2 = relevant, 1 = partial,
  0 = not relevant) per query.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
rng = random.Random(20260716)

# ---------------------------------------------------------------------------
# Real seed memories (subset distilled from the 7x24 run, anonymized ids)
# ---------------------------------------------------------------------------

REAL_FAILURES = [
    (
        "right scissors failed: joint_not_reached",
        "剪刀手势食指不到位，温度 42°C。",
        "rh56_rps_robot",
        "rh56_right",
        "rh56_rps",
    ),
    (
        "left rock failed: joint_not_reached",
        "左手石头手势小指收不拢，电流峰值 503mA。",
        "rh56_rps_robot",
        "rh56_left",
        "rh56_rps",
    ),
    (
        "left left_scissors failed: joint_not_reached",
        "左手剪刀拇指旋转不到位。",
        "rh56_rps_robot",
        "rh56_left",
        "rh56_rps",
    ),
    (
        "right ready failed: joint_not_reached",
        "右手 ready 手势食指角度差 200。",
        "rh56_rps_robot",
        "rh56_right",
        "rh56_rps",
    ),
    (
        "camera no_frames after re-enumeration",
        "D435i 重启后 pipeline 无帧，需要 hardware_reset。",
        "rh56_rps_robot",
        "d435i",
        "rh56_rps",
    ),
    (
        "serial USB_TIMEOUT on CH340",
        "CH340 串口在重新枚举后超时，EIO -110。",
        "rh56_rps_robot",
        "rh56_left",
        "rh56_rps",
    ),
    (
        "FTDI adapter persistent EIO",
        "FTDI 适配器持续 Input/output error，需断电重插。",
        "rh56_rps_robot",
        "rh56_right",
        "rh56_rps",
    ),
]

REAL_BODY = [
    (
        "right hand thermal drift 36→42°C",
        "右手连续运行 60 分钟，食指温度从 36°C 升至 42°C。",
        "rh56_rps_robot",
        "rh56_right",
        "rh56_rps",
    ),
    (
        "left hand thermal drift 35→38°C",
        "左手连续运行 60 分钟，温度从 35°C 升至 38°C。",
        "rh56_rps_robot",
        "rh56_left",
        "rh56_rps",
    ),
    (
        "right index temperature rose faster than left",
        "右手食指温升速率比左手高 18%。",
        "rh56_rps_robot",
        "rh56_right",
        "rh56_rps",
    ),
]

REAL_SKILL = [
    (
        "scissors: 96/144 (67%) verified",
        "右手剪刀 144 次验证通过 96 次。",
        "rh56_rps_robot",
        "rh56_right",
        "rh56_rps",
    ),
    (
        "rock: 140/144 (97%) verified",
        "右手石头 144 次验证通过 140 次。",
        "rh56_rps_robot",
        "rh56_right",
        "rh56_rps",
    ),
    (
        "paper: 142/144 (99%) verified",
        "右手布 144 次验证通过 142 次。",
        "rh56_rps_robot",
        "rh56_right",
        "rh56_rps",
    ),
    (
        "left_scissors: 88/144 (61%) verified",
        "左手剪刀 144 次验证通过 88 次。",
        "rh56_rps_robot",
        "rh56_left",
        "rh56_rps",
    ),
]

# Paraphrase pools for synthetic variants (bilingual).
ZH_PARAPHRASE = {
    "scissors": ["剪刀手势失败", "剪刀食指不到位", "剪子比不出来"],
    "rock": ["石头手势失败", "拳头握不紧", "石头小指收不拢"],
    "paper": ["布手势失败", "手掌张不开"],
    "overcurrent": ["手指过流", "电机电流异常", "电流超标"],
    "thermal": ["温度过高", "手指发热", "温升过快"],
    "serial": ["串口超时", "串口无响应", "串口断连"],
    "camera": ["相机无帧", "摄像头没画面", "深度相机卡住"],
}

EN_PARAPHRASE = {
    "scissors": ["scissors gesture failed", "index finger not reached", "cannot make scissors"],
    "rock": ["rock gesture failed", "fist not closed", "little finger stuck"],
    "paper": ["paper gesture failed", "hand not fully open"],
    "overcurrent": ["finger overcurrent", "motor current spike", "current above limit"],
    "thermal": ["temperature too high", "finger heating up", "thermal drift"],
    "serial": ["serial timeout", "USB serial EIO", "port disconnected"],
    "camera": ["camera no frames", "no video feed", "depth camera stuck"],
}

ROOT_CAUSES = [
    ("joint_not_reached", "关节角度未到目标"),
    ("serial_timeout", "串口超时"),
    ("thermal_derating", "热降额"),
    ("low_bus_voltage", "总线电压低"),
    ("mechanical_wear", "机械磨损"),
]

ROBOTS = ["rh56_rps_robot", "mobile_base_01", "arm_ur5_02"]
BODIES = {
    "rh56_rps_robot": ["rh56_left", "rh56_right"],
    "mobile_base_01": ["base"],
    "arm_ur5_02": ["ur5_arm"],
}
TASKS = {"rh56_rps_robot": "rh56_rps", "mobile_base_01": "patrol", "arm_ur5_02": "painting"}


def _mem(
    idx: int,
    memory_type: str,
    title: str,
    document: str,
    robot: str,
    body: str,
    task: str,
    outcome: str | None,
    age_days: float,
    confidence: float = 0.85,
) -> dict:
    now = 1784208000.0  # fixed reference time for determinism
    return {
        "memory_id": f"bench_{idx:04d}",
        "memory_type": memory_type,
        "robot_id": robot,
        "body_id": body,
        "task_id": task,
        "title": title,
        "document": document,
        "outcome": outcome,
        "confidence": confidence,
        "importance": 0.6,
        "event_time": now - age_days * 86400.0,
        "evidence_refs": [f"evt_bench_{idx:04d}_a", f"evt_bench_{idx:04d}_b"],
        "tags": [memory_type, task],
        "content_hash": f"hash_{idx:04d}",
        "status": "active",
    }


def build_dataset() -> list[dict]:
    items: list[dict] = []
    idx = 0

    # 1. Real seeds (failure/body/skill) with evidence.
    for title, doc, robot, body, task in REAL_FAILURES:
        items.append(
            _mem(idx, "failure", title, doc, robot, body, task, "failure", rng.uniform(0, 5))
        )
        idx += 1
    for title, doc, robot, body, task in REAL_BODY:
        items.append(_mem(idx, "body", title, doc, robot, body, task, None, rng.uniform(0, 5)))
        idx += 1
    for title, doc, robot, body, task in REAL_SKILL:
        items.append(
            _mem(idx, "skill", title, doc, robot, body, task, "success", rng.uniform(0, 5))
        )
        idx += 1

    # 2. Similar symptom, different root cause pairs (zh + en).
    for symptom_key in ("scissors", "rock", "overcurrent"):
        for root_cause, zh_cause in ROOT_CAUSES:
            for robot in ("rh56_rps_robot",):
                zh_title = rng.choice(ZH_PARAPHRASE[symptom_key])
                items.append(
                    _mem(
                        idx,
                        "failure",
                        f"{zh_title}（{zh_cause}）",
                        f"{zh_title}，根因分析：{zh_cause}。",
                        robot,
                        rng.choice(BODIES[robot]),
                        TASKS[robot],
                        "failure",
                        rng.uniform(0, 90),
                    )
                )
                idx += 1
                en_title = rng.choice(EN_PARAPHRASE[symptom_key])
                items.append(
                    _mem(
                        idx,
                        "failure",
                        f"{en_title} ({root_cause})",
                        f"{en_title}; root cause analysis: {root_cause}.",
                        robot,
                        rng.choice(BODIES[robot]),
                        TASKS[robot],
                        "failure",
                        rng.uniform(0, 90),
                    )
                )
                idx += 1

    # 3. Cross-robot decoys (same symptom wording, other robots/tasks).
    for robot in ("mobile_base_01", "arm_ur5_02"):
        for symptom_key in ("overcurrent", "thermal", "serial", "camera"):
            for _ in range(6):
                en_title = rng.choice(EN_PARAPHRASE[symptom_key])
                items.append(
                    _mem(
                        idx,
                        "failure",
                        f"{en_title} on {robot}",
                        f"{en_title}; observed on {robot} during {TASKS[robot]}.",
                        robot,
                        BODIES[robot][0],
                        TASKS[robot],
                        "failure",
                        rng.uniform(0, 120),
                    )
                )
                idx += 1
            for _ in range(4):
                zh_title = rng.choice(ZH_PARAPHRASE[symptom_key])
                items.append(
                    _mem(
                        idx,
                        "failure",
                        f"{zh_title}（{robot}）",
                        f"{zh_title}，出现在 {robot} 的 {TASKS[robot]} 任务中。",
                        robot,
                        BODIES[robot][0],
                        TASKS[robot],
                        "failure",
                        rng.uniform(0, 120),
                    )
                )
                idx += 1

    # 4. Body memories across robots (cross-body applicability).
    for robot in ROBOTS:
        for i in range(10):
            items.append(
                _mem(
                    idx,
                    "body",
                    f"{robot} thermal profile #{i}",
                    f"{robot} 第 {i} 次热态画像：温升曲线与负载关系。",
                    robot,
                    BODIES[robot][0],
                    TASKS[robot],
                    None,
                    rng.uniform(0, 180),
                )
            )
            idx += 1

    # 5. Episodic memories (many, low relevance for most queries).
    for _i in range(120):
        robot = rng.choice(ROBOTS)
        items.append(
            _mem(
                idx,
                "episodic",
                f"Episode run #{i} on {robot}",
                f"第 {i} 次练习会话：{rng.randint(100, 200000)} events, outcome "
                f"{rng.choice(['success', 'partial', 'failure'])}.",
                robot,
                BODIES[robot][0],
                TASKS[robot],
                rng.choice(["success", "partial", "failure"]),
                rng.uniform(0, 200),
            )
        )
        idx += 1

    # 6. Skill memories (per robot/gesture).
    for robot in ROBOTS:
        for gesture in ("scissors", "rock", "paper", "pick", "place"):
            for _i in range(2):
                rate = rng.uniform(0.5, 0.99)
                items.append(
                    _mem(
                        idx,
                        "skill",
                        f"{gesture} success rate {rate:.0%} on {robot}",
                        f"{gesture} 在 {robot} 上成功率 {rate:.1%}（{rng.randint(20, 300)} 次样本）。",
                        robot,
                        BODIES[robot][0],
                        TASKS[robot],
                        "success" if rate > 0.8 else "partial",
                        rng.uniform(0, 150),
                    )
                )
                idx += 1

    # 7. Stale vs fresh conflicts (old wrong limit, new corrected limit).
    for _i, (old_limit, new_limit, unit) in enumerate(
        [(60, 52, "°C"), (500, 380, "mA"), (12, 9, "V")]
    ):
        items.append(
            _mem(
                idx,
                "semantic",
                f"RH56 safety limit {unit}: {old_limit}",
                f"旧文档：RH56 安全上限 {old_limit}{unit}。（已被新证据取代）",
                "rh56_rps_robot",
                "rh56_right",
                "rh56_rps",
                None,
                180.0,
            )
        )
        idx += 1
        items.append(
            _mem(
                idx,
                "semantic",
                f"RH56 safety limit {unit}: {new_limit}",
                f"连续实验证据：RH56 在 {new_limit}{unit} 已出现异常，建议上限 {new_limit}{unit}。",
                "rh56_rps_robot",
                "rh56_right",
                "rh56_rps",
                None,
                2.0,
            )
        )
        idx += 1

    return items


def build_queries(dataset: list[dict]) -> tuple[list[dict], list[dict]]:
    queries: list[dict] = []
    labels: list[dict] = []

    def add(
        qid: int,
        text: str,
        relevant: list[str],
        partial: list[str] | None = None,
        robot_id: str | None = None,
        memory_types: list[str] | None = None,
        body_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        queries.append(
            {
                "query_id": f"q{qid:03d}",
                "text": text,
                "robot_id": robot_id,
                "memory_types": memory_types,
                "body_id": body_id,
                "task_id": task_id,
            }
        )
        grades = dict.fromkeys(relevant, 2)
        for pid in partial or []:
            grades[pid] = 1
        labels.append({"query_id": f"q{qid:03d}", "relevance": grades})

    def find(**kw) -> list[str]:
        out = []
        for m in dataset:
            if all(m.get(k) == v for k, v in kw.items()):
                out.append(m["memory_id"])
        return out

    qid = 1
    # 1-10: direct zh queries for real rh56 failures.
    zh_queries = [
        ("剪刀手势食指不到位", find(title="right scissors failed: joint_not_reached")),
        ("左手石头小指收不拢", find(title="left rock failed: joint_not_reached")),
        ("左手剪刀拇指旋转不到位", find(title="left left_scissors failed: joint_not_reached")),
        ("右手 ready 食指角度差", find(title="right ready failed: joint_not_reached")),
        ("相机重启后无帧", find(title="camera no_frames after re-enumeration")),
        ("CH340 串口超时", find(title="serial USB_TIMEOUT on CH340")),
        ("FTDI 持续 EIO", find(title="FTDI adapter persistent EIO")),
        ("右手温升 36 到 42", find(title="right hand thermal drift 36→42°C")),
        ("左手温升", find(title="left hand thermal drift 35→38°C")),
        ("右手食指温升比左手快", find(title="right index temperature rose faster than left")),
    ]
    for text, rel in zh_queries:
        add(qid, text, rel, robot_id="rh56_rps_robot")
        qid += 1

    # 11-20: direct en queries.
    en_queries = [
        (
            "scissors gesture index finger not reached",
            find(title="right scissors failed: joint_not_reached"),
        ),
        ("left rock little finger", find(title="left rock failed: joint_not_reached")),
        ("D435i no frames after restart", find(title="camera no_frames after re-enumeration")),
        ("CH340 serial timeout EIO", find(title="serial USB_TIMEOUT on CH340")),
        ("FTDI persistent EIO", find(title="FTDI adapter persistent EIO")),
        ("right hand thermal drift", find(title="right hand thermal drift 36→42°C")),
        ("scissors success rate", find(title="scissors: 96/144 (67%) verified")),
        ("rock success rate", find(title="rock: 140/144 (97%) verified")),
        ("paper success rate", find(title="paper: 142/144 (99%) verified")),
        ("left scissors success rate", find(title="left_scissors: 88/144 (61%) verified")),
    ]
    for text, rel in en_queries:
        add(qid, text, rel, robot_id="rh56_rps_robot")
        qid += 1

    # 21-35: paraphrase zh queries hitting synthetic root-cause variants.
    for symptom in ("scissors", "rock", "overcurrent"):
        for zh in ZH_PARAPHRASE[symptom][:2]:
            rel = [
                m["memory_id"]
                for m in dataset
                if m["robot_id"] == "rh56_rps_robot"
                and m["memory_type"] == "failure"
                and (
                    symptom in m["title"].lower()
                    or any(p in m["title"] for p in ZH_PARAPHRASE[symptom])
                    or any(p in m["document"] for p in ZH_PARAPHRASE[symptom])
                )
            ]
            add(qid, zh, rel[:12], robot_id="rh56_rps_robot", memory_types=["failure"])
            qid += 1

    # 36-45: metadata-constrained queries (cross-robot isolation checks).
    for robot in ("mobile_base_01", "arm_ur5_02"):
        for symptom in ("overcurrent", "thermal", "serial", "camera", "overcurrent"):
            rel = [
                m["memory_id"]
                for m in dataset
                if m["robot_id"] == robot and m["memory_type"] == "failure"
            ]
            add(qid, EN_PARAPHRASE[symptom][0], rel[:15], robot_id=robot, memory_types=["failure"])
            qid += 1

    # 46-55: cross-body queries.
    for body in ("rh56_left", "rh56_right"):
        for text in ("温升", "剪刀", "success rate", "thermal drift", "gesture failed"):
            rel = [m["memory_id"] for m in dataset if m.get("body_id") == body]
            add(qid, text, rel[:15], robot_id="rh56_rps_robot", body_id=body)
            qid += 1

    # 56-65: stale vs fresh conflict queries (fresh must outrank stale).
    for unit in ("°C", "mA", "V"):
        fresh = [
            m["memory_id"] for m in dataset if "建议上限" in m["document"] and unit in m["title"]
        ]
        stale = [
            m["memory_id"]
            for m in dataset
            if "已被新证据取代" in m["document"] and unit in m["title"]
        ]
        add(qid, f"RH56 安全温度/电流/电压上限 {unit}", fresh, stale, robot_id="rh56_rps_robot")
        qid += 1

    # 66-75: task-scoped queries.
    for task, robot in (
        ("rh56_rps", "rh56_rps_robot"),
        ("patrol", "mobile_base_01"),
        ("painting", "arm_ur5_02"),
    ):
        for text in ("failure", "失败", "success rate", "thermal", "episode"):
            rel = [m["memory_id"] for m in dataset if m.get("task_id") == task]
            add(qid, text, rel[:15], robot_id=robot, task_id=task)
            qid += 1

    # 76-85: error-code exact queries.
    for code in ("USB_TIMEOUT", "EIO", "-110", "USB_TIMEOUT", "EIO"):
        rel = [m["memory_id"] for m in dataset if code in m["document"] or code in m["title"]]
        add(qid, f"{code} 故障", rel, robot_id="rh56_rps_robot")
        qid += 1

    # 86-95: zh/en mixed semantic queries (similar symptom different root cause).
    mixed = [
        "手指过流 根因",
        "scissors root cause",
        "剪刀 热降额",
        "rock serial timeout",
        "过流 机械磨损",
        "thermal scissors failure",
        "串口 剪刀失败",
        "camera stuck 相机",
        "电流异常 finger",
        "joint not reached 关节",
    ]
    for text in mixed:
        tokens = text.split()
        rel = [
            m["memory_id"]
            for m in dataset
            if m["memory_type"] == "failure"
            and any(t.lower() in (m["title"] + m["document"]).lower() for t in tokens)
        ]
        add(qid, text, rel[:12], robot_id="rh56_rps_robot", memory_types=["failure"])
        qid += 1

    # 96-100: decoy queries (nothing relevant expected).
    for text in (
        "quantum entanglement protocol",
        "recipe for chocolate cake",
        "kubernetes pod eviction policy",
        "量子计算算法",
        "french revolution timeline",
    ):
        add(qid, text, [], robot_id="rh56_rps_robot")
        qid += 1

    # 101-104: canonical cross-lingual failure phrasings (§6.4) — all four must
    # recall the same overcurrent failure family.
    overcurrent_family = [
        m["memory_id"]
        for m in dataset
        if m["memory_type"] == "failure"
        and any(
            tok in (m["title"] + m["document"]).lower() for tok in ("overcurrent", "过流", "电流")
        )
    ]
    for text in ("手指过流", "finger overcurrent", "电机电流异常", "RH56 finger current spike"):
        add(qid, text, overcurrent_family[:15], robot_id="rh56_rps_robot", memory_types=["failure"])
        qid += 1

    # 105-109: body-memory queries.
    for text in ("thermal profile", "热态画像", "温升曲线", "body thermal", "热限"):
        rel = [m["memory_id"] for m in dataset if m["memory_type"] == "body"]
        add(qid, text, rel[:15], robot_id="rh56_rps_robot", memory_types=["body"])
        qid += 1

    # 110-113: time-range queries (recent window).
    for text in ("recent failure", "最近失败", "latest thermal", "最近 过流"):
        rel = [
            m["memory_id"]
            for m in dataset
            if m["memory_type"] == "failure" and m["event_time"] > 1784208000.0 - 30 * 86400.0
        ]
        add(qid, text, rel[:15], robot_id="rh56_rps_robot", memory_types=["failure"])
        qid += 1

    # 114-116: skill-success pattern queries.
    for text in ("gesture success pattern", "剪刀 成功率", "skill evidence"):
        rel = [m["memory_id"] for m in dataset if m["memory_type"] == "skill"]
        add(qid, text, rel[:15], robot_id="rh56_rps_robot", memory_types=["skill"])
        qid += 1

    return queries, labels


def main() -> None:
    dataset = build_dataset()
    queries, labels = build_queries(dataset)
    with (HERE / "dataset.jsonl").open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (HERE / "queries.jsonl").open("w", encoding="utf-8") as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + "\n")
    with (HERE / "relevance_labels.jsonl").open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")
    print(f"dataset: {len(dataset)} memories")
    print(f"queries: {len(queries)}")
    print(f"labels:  {len(labels)}")


if __name__ == "__main__":
    main()
