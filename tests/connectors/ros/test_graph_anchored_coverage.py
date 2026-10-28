"""Historical cleaning follows local public graph anchors, never current odom drift."""

import numpy as np
import pytest

from rosclaw.connectors.ros.public_coverage.graph_ledger import IDENTITY, GraphAnchoredCoverage
from tests.connectors.ros.test_public_online_coverage import grid


def ledger():
    result = GraphAnchoredCoverage(odom_epoch="odom", graph_epoch="slam", brush_radius=0.15)
    result.update_graph(
        {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (5.0, 0.0), 3: (6.0, 0.0)}, graph_epoch="slam"
    )
    return result


def sample(state, stamp, point, odom=None):
    state.sample_map(
        stamp=stamp,
        map_xy=point,
        odom_xy=point if odom is None else odom,
        cleaner=True,
        odom_epoch="odom",
        odom_fresh=True,
        cleaner_fresh=True,
    )


def test_odometry_drift_does_not_relocate_historical_brush_cells():
    state = ledger()
    sample(state, 1, [0.1, 0.0])
    state.sync()
    original = set(state.anchor.cleaned)
    # Different odom coordinates map to the same independently localized pose.
    sample(state, 2, [0.1, 0.0], odom=[1.1, 0.7])
    state.sync()
    assert state.anchor.cleaned == original
    assert state.evidence_samples[-1]["public_odom_xy"] == [1.1, 0.7]


def test_nonuniform_graph_corrections_move_each_local_patch_independently():
    state = ledger()
    sample(state, 1, [0.1, 0.0])
    sample(state, 3, [5.1, 0.0])
    left = state.patches[(0, 1)]
    right = state.patches[(2, 3)]
    state.update_graph(
        {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (7.0, 1.0), 3: (7.0, 2.0)}, graph_epoch="slam"
    )
    state.sync()
    expected_left = state._keys(state.frame((0, 1)).apply(state._points(left.anchor.cleaned)))
    expected_right = state._keys(state.frame((2, 3)).apply(state._points(right.anchor.cleaned)))
    assert state.anchor.cleaned == expected_left | expected_right
    assert np.mean(state._points(expected_right), axis=0)[1] > 1


def test_missing_graph_nodes_keep_unlocalized_debt_and_award_no_credit():
    state = ledger()
    sample(state, 1, [0.1, 0.0])
    state.observe(grid(), IDENTITY)
    debt = sum(len(p.anchor.opportunities) for p in state.patches.values())
    state.update_graph({2: (5.0, 0.0), 3: (6.0, 0.0)}, graph_epoch="slam")
    result = state.status(grid(), IDENTITY)
    assert result["unlocalized_obligations"] > 0
    assert sum(len(p.anchor.opportunities) for p in state.patches.values()) == debt
    assert result["whole_house_success"] is None


def test_restart_preserves_patch_ids_local_debt_and_samples_without_interpolation(tmp_path):
    state = ledger()
    sample(state, 1, [0.1, 0.0])
    state.observe(grid(), IDENTITY)
    state.failed([[0.2, 0], [0.3, 0]], [0.1, 0], now=5)
    path = tmp_path / "state.json"
    state.save(path)
    restored = GraphAnchoredCoverage.load(path, odom_epoch="odom", graph_epoch="slam")
    import json

    assert restored.evidence_samples == json.loads(json.dumps(state.evidence_samples))
    assert restored.anchor.cleaned == state.sync().anchor.cleaned
    assert all(p.last_sample is None for p in restored.patches.values())
    with pytest.raises(ValueError, match="epoch mismatch"):
        GraphAnchoredCoverage.load(path, odom_epoch="odom", graph_epoch="other")


def test_planner_snapshot_cannot_award_synthetic_execution_credit_on_merge():
    state = ledger()
    snapshot = state.snapshot()
    snapshot.observe(grid(), IDENTITY)
    sample(snapshot, 1, [0.1, 0.0])
    state.merge_discovered(snapshot)
    state.sync()
    assert state.anchor.opportunities and not state.anchor.cleaned
    assert not state.evidence_samples


def test_current_map_from_odom_is_not_an_allowed_global_patch_correction():
    from rosclaw.connectors.ros.public_coverage.online import Transform2D

    state = ledger()
    with pytest.raises(ValueError, match="not a map gauge change"):
        state.observe(grid(), Transform2D(0.2, 0.3, 0.1))
