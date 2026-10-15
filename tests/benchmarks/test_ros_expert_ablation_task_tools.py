"""No ROS or model execution; enforce shared staging and arm boundaries."""

import copy
import json

import pytest

from benchmarks.ros_expert_ablation.task_tools import TaskTools


@pytest.fixture
def files():
    nav = {
        "/robot": {
            "amcl": {"ros__parameters": {"tf_broadcast": False}},
            "lifecycle_manager_navigation": {"ros__parameters": {"node_names": ["controller"]}},
            "local_costmap": {"local_costmap": {"ros__parameters": {"robot_radius": 0.2}}},
            "global_costmap": {"global_costmap": {"ros__parameters": {"robot_radius": 0.2}}},
        }
    }
    return {
        "nav2.yaml": json.dumps(nav),
        "bringup.json": json.dumps({"scan_bridge_state": "running"}),
        "native.json": "{}",
        "robot.urdf": "original",
        "requirements.json": "{}",
    }


def tools(files, arm="A", **kwargs):
    return TaskTools(
        files, arm=arm, reviewed_at="2026-10-15T00:00:00+00:00", minimum_radius_m=0.369, **kwargs
    )


def test_generic_arm_cannot_call_expert_or_memory(files):
    t = tools(files)
    for op in ["ros_inspect_system", "ros_diagnose", "ros_context", "query_memory"]:
        with pytest.raises(ValueError, match="forbidden"):
            t.call(op, {})


def test_expert_without_memory_cannot_query_training_store(files):
    with pytest.raises(ValueError, match="forbidden"):
        tools(files, "B").call("query_memory", {"query": "example"})


def test_memory_queries_have_actual_callback_and_trace(files):
    calls = []

    def actual(q):
        calls.append(q)
        return {"admitted": [], "rejected": ["same evaluation case"]}

    t = tools(files, "C", memory_query=actual)
    assert t.call("query_memory", {"query": "workflow"})["admitted"] == []
    assert calls == ["workflow"] and t.memory_trace[0]["query"] == "workflow"


def test_original_files_cannot_be_read_outside_task_allowlist(files):
    t = tools(files)
    with pytest.raises(ValueError, match="allowlist"):
        t.call("read_file", {"file": "/etc/passwd"})
    r = t.call("read_file", {"file": "nav2.yaml", "pointer": "/~1robot/amcl/ros__parameters"})
    assert json.loads(r["text"]) == {"tf_broadcast": False}
    assert not r["truncated"]


def test_invalid_patch_is_atomic_and_cannot_modify_body_or_tf_publisher(files):
    t = tools(files)
    before = copy.deepcopy(t.staging)
    changes = [
        {
            "file": "nav2.yaml",
            "pointer": "/~1robot/amcl/ros__parameters/tf_broadcast",
            "value": True,
        },
        {"file": "robot.urdf", "pointer": "/collision", "value": "smaller"},
    ]
    with pytest.raises(ValueError, match="allowlist"):
        t.call("propose_patch", {"changes": changes})
    assert t.staging == before and not t.patch_trace
    with pytest.raises(ValueError):
        t.call(
            "propose_patch",
            {"changes": [{"file": "nav2.yaml", "pointer": "/static_map_to_odom", "value": True}]},
        )


@pytest.mark.parametrize("radius", [0.1, 0.368, True, float("nan")])
def test_geometry_shrink_is_blocked_before_a_world_or_motion(files, radius):
    with pytest.raises(ValueError, match="typed staging"):
        tools(files).call(
            "propose_patch",
            {
                "changes": [
                    {
                        "file": "nav2.yaml",
                        "pointer": "/~1robot/local_costmap/local_costmap/ros__parameters/robot_radius",
                        "value": radius,
                    }
                ]
            },
        )


def test_minimal_actual_staging_patch_keeps_original_files(files):
    t = tools(files)
    r = t.call(
        "propose_patch",
        {
            "changes": [
                {
                    "file": "nav2.yaml",
                    "pointer": "/~1robot/amcl/ros__parameters/tf_broadcast",
                    "value": True,
                }
            ]
        },
    )
    assert r["status"].startswith("PROPOSED_STAGING")
    assert (
        json.loads(files["nav2.yaml"])["/robot"]["amcl"]["ros__parameters"]["tf_broadcast"] is False
    )
    assert t.staging["nav2.yaml"]["/robot"]["amcl"]["ros__parameters"]["tf_broadcast"] is True


def test_memory_arm_requires_actual_query_before_staging_proposal(files):
    t = tools(files, "C", memory_query=lambda q: {"admitted": []})
    patch = {
        "changes": [
            {
                "file": "nav2.yaml",
                "pointer": "/~1robot/amcl/ros__parameters/tf_broadcast",
                "value": True,
            }
        ]
    }
    with pytest.raises(ValueError, match="query provenance-bound"):
        t.call("propose_patch", patch)
    t.call("query_memory", {"query": "workflow"})
    assert t.call("propose_patch", patch)["status"].startswith("PROPOSED_STAGING")
