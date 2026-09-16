"""Task Predicate Registry v2 测试（MH12，0916 优化 §十五，红→绿）。

每个谓词返回 {predicate, ok, measured, threshold}；
自然语言条件永远不是 verifier truth。
"""

from __future__ import annotations

from rosclaw.sim.experiment.predicates import evaluate_predicates


def test_inside_near_still_work() -> None:
    observations = {"body_pose:cube": {"pos": [0.6, 0.0, 0.05], "quat": [0, 0, 0, 1]}}
    predicates = [
        {"channel": "body_pose:cube", "field": "pos", "inside": {"min": [0.5, -0.1, 0.0], "max": [0.7, 0.1, 0.2]}},
        {"channel": "body_pose:cube", "field": "pos", "near": {"target": [0.0, 0.0, 0.0], "tolerance": 0.01}},
    ]
    results = evaluate_predicates(observations, predicates)
    assert results[0]["ok"] is True and "measured" in results[0]
    assert results[1]["ok"] is False and results[1]["measured"] > 0.01


def test_contact_predicate() -> None:
    observations = {
        "contact_pairs": [
            {"geom1": "palm_g", "geom2": "cube_g", "dist": -0.001},
            {"geom1": "floor", "geom2": "other", "dist": 0.0},
        ]
    }
    hit = evaluate_predicates(
        observations,
        [{"contact": {"body1": "palm_g", "body2": "cube_g", "max_dist": 0.0}}],
    )
    assert hit[0]["ok"] is True
    assert hit[0]["measured"] == -0.001

    miss = evaluate_predicates(
        observations,
        [{"contact": {"body1": "ghost", "body2": "cube_g", "max_dist": 0.0}}],
    )
    assert miss[0]["ok"] is False
    assert miss[0]["reason"] == "no_matching_contact"


def test_joint_in_range_predicate() -> None:
    observations = {"joint_positions": [0.5, -0.2]}
    ok = evaluate_predicates(
        observations,
        [{"joint_in_range": {"index": 0, "min": 0.4, "max": 0.6}}],
    )
    assert ok[0]["ok"] is True and ok[0]["measured"] == 0.5
    not_ok = evaluate_predicates(
        observations,
        [{"joint_in_range": {"index": 0, "min": 0.6, "max": 1.0}}],
    )
    assert not_ok[0]["ok"] is False


def test_upright_predicate() -> None:
    observations = {"body_pose:obj": {"quat": [0, 0, 0, 1]}}
    ok = evaluate_predicates(
        observations,
        [{"channel": "body_pose:obj", "upright": {"max_tilt_deg": 10.0}}],
    )
    assert ok[0]["ok"] is True and ok[0]["measured"] < 1.0

    import math

    angle = math.radians(45)
    observations45 = {"body_pose:obj": {"quat": [math.sin(angle / 2), 0, 0, math.cos(angle / 2)]}}
    tilted = evaluate_predicates(
        observations45,
        [{"channel": "body_pose:obj", "upright": {"max_tilt_deg": 10.0}}],
    )
    assert tilted[0]["ok"] is False
    assert tilted[0]["measured"] == 45.0 or abs(tilted[0]["measured"] - 45.0) < 1.0


def test_speed_below_predicate() -> None:
    observations = {"joint_velocities": [0.1, -0.05]}
    ok = evaluate_predicates(observations, [{"speed_below": {"max": 0.5}}])
    assert ok[0]["ok"] is True and ok[0]["measured"] == 0.1
    not_ok = evaluate_predicates(observations, [{"speed_below": {"max": 0.01}}])
    assert not_ok[0]["ok"] is False


def test_natural_language_condition_is_not_truth() -> None:
    """§十五：'cube looks placed correctly' 类条件不可执行 → 拒绝。"""
    results = evaluate_predicates(
        {},
        [{"condition": "cube looks placed correctly"}],
    )
    assert results[0]["ok"] is False
    assert "unknown" in results[0]["reason"]
