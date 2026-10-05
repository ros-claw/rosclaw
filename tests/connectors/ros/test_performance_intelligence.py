"""Performance evidence must not turn graph names or absent traces into proof."""

from datetime import UTC, datetime, timedelta

import pytest

from rosclaw.connectors.ros.intelligence import RosSystemModel
from rosclaw.connectors.ros.intelligence.performance import performance_graph


def system(performance=None):
    now = datetime.now(UTC)
    return RosSystemModel(
        robot_id="gpu_robot",
        snapshot_id="",
        captured_at=now,
        graph={
            "nodes": [{"name": "/cuda_camera"}, {"name": "/isaac_inference"}],
            "topics": [
                {
                    "name": "/image",
                    "publishers": ["/cuda_camera"],
                    "subscribers": ["/isaac_inference"],
                }
            ],
        },
        observations={"performance": performance or {}},
    ).seal()


def test_gpu_named_nodes_and_missing_trace_remain_unknown():
    graph = performance_graph(system())
    assert {n["buffer_backend"] for n in graph.nodes} == {"UNKNOWN"}
    assert graph.edges[0]["separate_process"] is None
    assert graph.measured_copy_bytes is None
    assert set(graph.validation_checks.values()) == {None}
    assert graph.optimization_verified is False


@pytest.mark.parametrize("age", [0, 10, -10])
def test_only_fresh_sourced_trace_contributes_copy_measurements(age):
    observation = {
        "source": "fixture_trace",
        "captured_at": (datetime.now(UTC) - timedelta(seconds=age)).isoformat(),
        "nodes": {
            "/cuda_camera": {"source": "fixture_backend_probe", "buffer_backend": "CPU", "pid": 1},
            "/isaac_inference": {
                "source": "fixture_backend_probe",
                "buffer_backend": "CUDA",
                "pid": 2,
            },
        },
        "copy_events": [{"source": "fixture_trace", "direction": "HtoD", "bytes": 4096}],
        "validation_checks": {
            "backend_type": True,
            "separate_process_transport": True,
            "cpu_fallback": True,
            "buffer_lifetime": True,
            "memory_copy": True,
        },
    }
    graph = performance_graph(system(observation))
    assert graph.measured_copy_bytes == (4096 if age == 0 else None)
    assert graph.edges[0]["separate_process"] is (True if age == 0 else None)
    assert graph.optimization_verified is False  # No measured before/after comparison.
    assert graph.edges[0]["zero_copy_verified"] is False


def test_malformed_optional_observations_do_not_claim_zero_copy():
    graph = performance_graph(system(["invalid"]))
    assert graph.measured_copy_bytes is None
    assert graph.optimization_verified is False


@pytest.mark.parametrize(
    "events",
    [
        None,
        "invalid",
        {"bytes": 4096},
        [None],
        [{}],
        [{"source": "trace", "direction": "HtoD", "bytes": True}],
        [{"source": "", "direction": "HtoD", "bytes": 4096}],
    ],
)
def test_malformed_trace_is_unknown_even_when_completeness_flag_is_true(events):
    graph = performance_graph(
        system(
            {
                "source": "trace",
                "captured_at": datetime.now(UTC).isoformat(),
                "copy_trace_complete": True,
                "copy_events": events,
            }
        )
    )
    assert graph.measured_copy_bytes is None
    assert graph.optimization_verified is False


def test_isaac_5_does_not_claim_jazzy_compatibility_or_automatic_install():
    model = system()
    model.environment = {"ros_generation": "ros2", "distro": "jazzy"}
    options = performance_graph(model).technology_options
    assert len(options) == 4
    assert all(o["status"] == "BLOCKED" for o in options)
    assert all(o["automatic_install"] is False for o in options)
    assert all(o["optimization_verified"] is False for o in options)


def test_lyrical_and_gpu_named_nodes_still_require_platform_and_measured_workload():
    model = system()
    model.environment = {"ros_generation": "ros2", "distro": "lyrical"}
    options = performance_graph(model).technology_options
    assert all(o["status"] == "UNKNOWN" for o in options)
    assert all(o["requirements"]["gpu_runtime"] is None for o in options)
