"""确定性 Intent Router（总纲 §7.1，WP-P0-5 一级路由）。

只识别高价值已知任务（当前：draw_shape/五角星）——命中即产出
TaskSpec，零模型回合；未命中返回 None 交给模型（宁缺毋滥：
误路由比多一次模型请求更糟）。
"""

from __future__ import annotations

import re

#: 五角星意图：必须同时命中"绘制动词 + 星形名词"（或英文等价）。
_DRAW_VERBS = re.compile(r"画|绘|绘制|draw|trace")
_STAR_NOUNS = re.compile(r"五角星|星形|five[- ]pointed star|star", re.IGNORECASE)
_RADIUS = re.compile(r"半径\s*([0-9]+(?:\.[0-9]+)?)\s*(?:米|m(?! sec))", re.IGNORECASE)

#: 排除：有"画/星"但明显不是任务请求（故事/图片/诗歌）。
_NON_TASK = re.compile(r"故事|诗歌|图片|照片|讲讲|什么是|介绍|story|poem|picture")


def route_intent(text: str) -> dict | None:
    """自然语言 → TaskSpec（已知任务）；未知返回 None。"""
    if _NON_TASK.search(text):
        return None
    if not (_DRAW_VERBS.search(text) and _STAR_NOUNS.search(text)):
        return None
    radius = 0.10
    match = _RADIUS.search(text)
    if match:
        radius = float(match.group(1))
    return {
        "goal": "draw_shape",
        "parameters": {
            "shape": "star5",
            "center_m": [0.35, 0.25, 0.30],
            "radius_m": radius,
        },
        "matched_by": "intent_router.v1",
    }
