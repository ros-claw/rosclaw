"""Final boundary contract: fixed natural-language cases, not a benchmark."""

from __future__ import annotations

import pytest

from rosclaw.knowledge.intent_router import KnowledgeIntentRouter

CASES = [
    ("当前这台机器人有几个关节？", "body"),
    ("这台 G1 配了哪些传感器？", "body"),
    ("本机执行器状态是什么？", "body"),
    ("现在绑定的机器人型号是什么？", "body"),
    ("What are this robot's current joint limits?", "body"),
    ("上次巡检为什么失败？", "memory"),
    ("之前这台机器人撞过什么障碍？", "memory"),
    ("我记得上次标定有什么异常？", "memory"),
    ("What happened last time during docking?", "memory"),
    ("查询过去的抓取失败记录", "memory"),
    ("G1 有几个关节？", "know"),
    ("Unitree 官方 G1 规格是什么？", "know"),
    ("找一下上游 RealSense 的当前文档", "know"),
    ("有哪些开源项目实现了人形足球？", "know"),
    ("这篇论文的训练方法是什么？", "know"),
    ("这个 API 最新版本有哪些变化？", "know"),
    ("别人一般如何处理巡检定位漂移？", "know"),
    ("当前定位漂移了怎么办？", "how"),
    ("这个错误如何排查？", "how"),
    ("现在应如何适配 ROS 2 Jazzy？", "how"),
    ("摄像头超时怎么修？", "how"),
    ("Which route should I use for the current failure?", "how"),
    ("当前机器人有哪些技能？", "skill"),
    ("如何调用导航 skill？", "skill"),
    ("MCP registry 里有拍照工具吗？", "skill"),
    ("这台机器人能不能抓取？", "skill"),
    ("让小车回充。", "action"),
    ("请让机械臂抓取杯子。", "action"),
    ("Move the robot forward now.", "action"),
    ("立即停下机器人。", "action"),
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_boundary_route(text: str, expected: str) -> None:
    route = KnowledgeIntentRouter().route(text)
    assert route.destination == expected
    assert route.action_authority is False


def test_no_know_memory_or_how_action_cross_routing() -> None:
    routes = [(text, expected, KnowledgeIntentRouter().route(text).destination) for text, expected in CASES]
    assert not [item for item in routes if item[1] == "memory" and item[2] == "know"]
    assert not [item for item in routes if item[1] == "know" and item[2] == "memory"]
    assert not [item for item in routes if item[1] == "action" and item[2] == "how"]
    assert not [item for item in routes if item[1] == "how" and item[2] == "action"]


@pytest.mark.parametrize(
    "text",
    [
        "请深入调研类似项目",
        "查找这篇论文和上游实现",
        "这个 API 最新版本是什么",
        "比较外部方案",
        "research an unknown error",
    ],
)
def test_auto_research_only_for_explicit_triggers(text: str) -> None:
    route = KnowledgeIntentRouter().route(text)
    assert route.destination == "know"
    assert route.auto_research is True


def test_general_conversation_does_not_auto_research() -> None:
    assert KnowledgeIntentRouter().route("G1 有几个关节？").auto_research is False
