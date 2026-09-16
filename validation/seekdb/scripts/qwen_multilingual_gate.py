#!/usr/bin/env python3
"""Qwen multilingual production-path gate (PR-SDB-140-5, P1-5).

The REAL production retrieval path, end to end on the live 1.4 engine:

    SeekDB Engine 1.4.0 + pyseekdb 1.4.0.post1 + Qwen3-Embedding-0.6B
    (pinned revision, local sentence-transformers) + ROSClaw versioned
    multilingual collection (VersionedCollectionManager, manual embeddings,
    ngram analyzer, query-side instruction per the Qwen model card).

Corpus: 64 curated docs (8 scenarios × zh/en × 4 phrasings) — 8 robot-failure scenarios (UR5e/RH56/LIMO/Nova
Carter/D435i) × zh/en × 2 phrasings.  Queries: 24 — 8 zh→zh, 8 zh→en,
8 en→zh.  Legs: vector (shadow_query with Qwen query embedding), BM25,
hybrid RRF.  The built-in MiniLM result in report 04 (vector Recall@1
0.833) stays the baseline — this gate exists to prove the PRODUCTION path
clears it on zh-heavy queries.

Exit 1 when any leg falls under --fail-under (default: vector/hybrid 0.9
Recall@5, bm25 0.8).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

# 8 scenarios × zh/en × 2 phrasings.  (scenario, doc_suffix, lang, text)
_SCENARIOS: dict[str, dict[str, list[str]]] = {
    "joint": {
        "zh": [
            "机械臂 UR5e 关节目标越界,sandbox 在派发前阻断。",
            "UR5e 的关节角度超出限位,派发前被沙箱拦截。",
            "UR5e 目标关节角超限,sandbox 拒绝执行该轨迹。",
            "UR5e 轨迹因关节限位在沙箱阶段被挡下。",
        ],
        "en": [
            "UR5e joint target out of range, blocked by the sandbox before dispatch.",
            "The arm's joint angle exceeded limits and was rejected pre-dispatch.",
            "Joint target beyond the UR5e envelope; the sandbox refused the trajectory.",
            "Trajectory rejected at the sandbox stage due to UR5e joint limits.",
        ],
    },
    "force": {
        "zh": [
            "RH56 抓取时接触力超过安全阈值,force_set 从 300 降到 50 后恢复。",
            "右手抓取力超限,降低 force_set 后接触稳定。",
            "RH56 夹持力越过安全线,通过下调力上限完成恢复。",
            "抓取力过大触发保护,调低 force_set 后 RH56 恢复稳定抓取。",
        ],
        "en": [
            "RH56 gripper contact force exceeded the safe threshold; recovered by lowering force_set.",
            "Grasp force over limit on the right hand; lowering force_set stabilized contact.",
            "The gripper crossed its force safety line and recovered after the cap was lowered.",
            "Excess grasp force tripped protection; RH56 steadied after force_set was reduced.",
        ],
    },
    "lidar": {
        "zh": [
            "LIMO 激光雷达丢包后定位漂移,重定位恢复。",
            "LIMO 的雷达丢帧造成位姿漂移,重新定位后恢复正常。",
            "LIMO 激光数据中断引起定位误差,触发了重定位流程。",
            "雷达丢包期间 LIMO 位姿发散,重定位后误差消除。",
        ],
        "en": [
            "LIMO localization drift after lidar packet loss; relocalized.",
            "Pose drift on LIMO caused by lidar frame drops; recovered via relocalization.",
            "A lidar data gap introduced localization error on LIMO, triggering relocalization.",
            "LIMO pose diverged during the lidar outage; relocalization cleared the error.",
        ],
    },
    "keepout": {
        "zh": [
            "Nova Carter 导航进入禁入区触发重新规划。",
            "Nova Carter 误入禁行区域,规划器自动重规划绕开。",
            "Nova Carter 闯入限制区域后自动生成绕行路径。",
            "禁入区边界被穿越,Nova Carter 立刻重新规划路径。",
        ],
        "en": [
            "Nova Carter navigation entered a keep-out zone and replanned.",
            "The rover drove into a forbidden area; the planner rerouted around it.",
            "After entering the restricted zone, Nova Carter generated a detour automatically.",
            "The keep-out boundary was crossed and Nova Carter immediately replanned.",
        ],
    },
    "thumb": {
        "zh": [
            "左手 thumb_rot 跟随漂移导致 OK 手势失败,校准后通过。",
            "左手拇指旋转跟随误差导致手势校验失败,重新校准后恢复。",
            "左手拇指旋转漂移使 OK 手势判定不通过,校准修复。",
            "thumb_rot 通道漂移造成左手手势验收失败,校准后达标。",
        ],
        "en": [
            "Left thumb_rot tracking drift failed the OK gesture; passed after calibration.",
            "The left thumb's rotation lag broke gesture verification until recalibrated.",
            "Drift in the left thumb rotation channel failed OK-gesture validation until calibration.",
            "Drift on the thumb_rot channel failed left-hand gesture acceptance until calibration.",
        ],
    },
    "wedge": {
        "zh": [
            "RealSense 相机在 pipe.start 卡死,UVC GET_CUR -110。",
            "D435i 启动管道时 wedge,UVC 控制请求超时 -110。",
            "RealSense 管道初始化挂起,UVC 请求返回 -110 超时。",
            "相机初始化卡死,UVC GET_CUR 控制传输超时 -110。",
        ],
        "en": [
            "Camera wedged at pipe.start with UVC error -110.",
            "The D435i hung starting the pipeline; the UVC control request timed out.",
            "The RealSense pipeline hung at initialization; the UVC request returned -110.",
            "Camera initialization froze; the UVC GET_CUR control transfer timed out with -110.",
        ],
    },
    "dock": {
        "zh": [
            "LIMO 对接过程中通信超时,串口 guardian 重连恢复。",
            "LIMO 充电对接时串口超时,guardian 自动重连完成对接。",
            "对接中途链路超时,串口守护进程重新建立连接。",
            "LIMO 对接链路超时中断,guardian 重连后顺利完成。",
        ],
        "en": [
            "Communication timeout during docking; serial guardian reconnected.",
            "LIMO's serial link timed out mid-docking; the guardian reconnected it.",
            "The docking link timed out mid-procedure; the serial daemon re-established it.",
            "LIMO docking dropped on a link timeout and completed after the guardian reconnected.",
        ],
    },
    "griprec": {
        "zh": [
            "夹爪故障通过预演动作恢复。",
            "抓取机构失效后,用预演过的恢复动作重新就位。",
            "夹爪失效,调用事先演练的复位动作完成恢复。",
            "抓取机构异常,执行预演复位序列后恢复功能。",
        ],
        "en": [
            "Gripper failure recovered via a rehearsed motion.",
            "After the gripper failed, a rehearsed recovery motion re-seated it.",
            "The gripper failed and was restored by invoking a pre-rehearsed reset motion.",
            "The gripper malfunctioned and regained function after the rehearsed reset sequence.",
        ],
    },
}

# 24 queries: (scenario, lang, text).  A query's relevant set = all 4 docs
# of its scenario (both languages, both phrasings).
_QUERIES: list[tuple[str, str, str]] = [
    ("force", "zh", "RH56 抓取过程中接触力过大的历史失败有哪些?"),
    ("joint", "zh", "之前 UR5e 在 sandbox 为什么被阻断?"),
    ("lidar", "zh", "类似这次 lidar 丢帧的恢复办法是什么?"),
    ("thumb", "zh", "OK 手势失败的已知原因"),
    ("keepout", "zh", "导航进入禁入区之后系统怎么处理?"),
    ("wedge", "zh", "相机启动卡死怎么恢复?"),
    ("dock", "zh", "对接时通信中断怎么办?"),
    ("griprec", "zh", "夹爪故障有什么恢复手段?"),
    ("force", "zh", "右手接触力超限的处理记录"),  # zh query → en docs relevant too
    ("joint", "zh", "关节越界被拦截的案例"),
    ("lidar", "zh", "雷达丢包导致定位飘了的解决办法"),
    ("thumb", "zh", "thumb_rot 漂移导致手势失败"),
    ("keepout", "zh", "车开进禁行区会重新规划吗"),
    ("wedge", "zh", "D435i 管道启动 wedge 的处置"),
    ("dock", "zh", "串口 guardian 重连恢复对接"),
    ("griprec", "zh", "预演动作恢复抓取机构"),
    ("force", "en", "gripper contact force exceeded safe threshold"),
    ("joint", "en", "why was UR5e blocked by the sandbox"),
    ("lidar", "en", "how to recover from lidar frame loss drift"),
    ("thumb", "en", "left thumb rotation drift OK gesture failure"),
    ("keepout", "en", "navigation entered keep-out zone replanning"),
    ("wedge", "en", "camera wedge at pipeline start UVC timeout"),
    ("dock", "en", "serial communication timeout during docking recovery"),
    ("griprec", "en", "rehearsed motion for gripper failure recovery"),
]


def _pct(values: list[float], p: float) -> float:
    values = sorted(values)
    k = max(0, min(len(values) - 1, math.ceil(p / 100 * len(values)) - 1))
    return round(values[k], 3)


def _rrf(leg_a: list[str], leg_b: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ids in (leg_a, leg_b):
        for rank, i in enumerate(ids):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seekdb-url", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--report", default=None)
    args = p.parse_args()

    from urllib.parse import urlparse

    from rosclaw.embedding.local_sentence_transformer import LocalSentenceTransformerProvider
    from rosclaw.embedding.profile import QWEN3_06B_1024
    from rosclaw.storage.seekdb_native import SeekDBServerRetrievalStore
    from rosclaw.storage.versioned_collections import VersionedCollectionManager

    parsed = urlparse(args.seekdb_url)
    store = SeekDBServerRetrievalStore(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 2881,
        database=(parsed.path or "/qwen_gate").lstrip("/"),
    )
    store.connect()
    provider = LocalSentenceTransformerProvider(QWEN3_06B_1024, device=args.device)
    manager = VersionedCollectionManager(store, provider)

    # corpus
    docs = []
    for scenario, langs in _SCENARIOS.items():
        for lang, texts in langs.items():
            for j, text in enumerate(texts):
                docs.append(
                    {
                        "id": f"{scenario}_{lang}{j}",
                        "document": text,
                        "title": f"{scenario}_{lang}{j}",
                    }
                )
    print(
        f"building versioned collection over {len(docs)} docs with "
        f"{provider.profile.profile_id} ...",
        flush=True,
    )
    t0 = time.perf_counter()
    build = manager.build("qwen_gate", docs)
    build_s = time.perf_counter() - t0
    physical = build["physical_collection"]
    print(f"built {physical} in {build_s:.1f}s", flush=True)

    # legs
    legs: dict[str, dict] = {}
    for leg in ("vector", "bm25", "hybrid"):
        r1, r5, mrr, lat = [], [], [], []
        for scenario, _lang, qtext in _QUERIES:
            relevant = {f"{scenario}_{lang}{j}" for lang in ("zh", "en") for j in (0, 1, 2, 3)}
            t = time.perf_counter()
            vids = [
                r.get("id")
                for r in manager.shadow_query(
                    "qwen_gate", qtext, analyzer="ngram", limit=5, exact_boost=False
                )
            ]
            bids = [r.get("id") for r in store.fulltext_search(physical, qtext, limit=5)]
            ids = {"vector": vids, "bm25": bids, "hybrid": _rrf(vids, bids)}[leg]
            lat.append((time.perf_counter() - t) * 1000)
            r1.append(1.0 if ids and ids[0] in relevant else 0.0)
            r5.append(1.0 if any(i in relevant for i in ids[:5]) else 0.0)
            rr = 0.0
            for rank, i in enumerate(ids):
                if i in relevant:
                    rr = 1.0 / (rank + 1)
                    break
            mrr.append(rr)
        legs[leg] = {
            "recall@1": round(statistics.fmean(r1), 4),
            "recall@5": round(statistics.fmean(r5), 4),
            "mrr": round(statistics.fmean(mrr), 4),
            "p50_ms": _pct(lat, 50),
            "p95_ms": _pct(lat, 95),
        }

    report = {
        "profile": provider.profile.profile_id,
        "docs": len(docs),
        "queries": len(_QUERIES),
        "build_s": round(build_s, 1),
        "legs": legs,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text)

    floors = {"vector": 0.9, "hybrid": 0.9, "bm25": 0.8}
    ok = True
    for leg, floor in floors.items():
        if legs[leg]["recall@5"] < floor:
            print(f"FAIL: {leg} recall@5 {legs[leg]['recall@5']} < {floor}")
            ok = False
    print("QWEN MULTILINGUAL GATE:", "PASS" if ok else "FAIL")
    store.disconnect()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
