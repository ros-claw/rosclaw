"""Generic readiness derives expected identities from sealed source, not a profile."""

import importlib
from pathlib import Path

import pytest

from tests.connectors.ros.test_generic_stack_source import synthetic_stack_inputs


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("lifecycle_readiness")


def prepare_targets(tmp_path):
    directory = tmp_path / "sealed"
    importlib.import_module("generic_stack_source").prepare_generic_stack_source(
        directory, **synthetic_stack_inputs()
    )
    return importlib.import_module("generic_navigation_launch").navigation_launch_plan(directory)[
        "lifecycle_node_names"
    ]


def snapshot(module, targets):
    return {
        "schema_version": module.SCHEMA,
        "source": "actual_read_only_GetState_responses",
        "ready": True,
        "responses": {
            name: {
                "service": name + "/get_state",
                "state_id": 3,
                "state_label": "active",
                "received_monotonic_sec": 99.5,
            }
            for name in targets
        },
    }


def test_all_ten_source_nodes_required_with_exact_namespace(module, tmp_path):
    targets = prepare_targets(tmp_path)
    assert len(targets) == 10 and all(name.startswith("/") for name in targets)
    value = snapshot(module, targets)
    assert module.readiness(value, now=100, required_nodes=targets)
    assert not module.readiness(value, now=100)  # Legacy seven-node fixture is separate.
    del value["responses"][targets[-1]]
    assert not module.readiness(value, now=100, required_nodes=targets)


@pytest.mark.parametrize(
    "key,value",
    [
        ("service", "/wrong_scope/node/get_state"),
        ("state_id", True),
        ("state_id", 2),
        ("state_label", "inactive"),
        ("received_monotonic_sec", 97),
        ("received_monotonic_sec", 101),
    ],
)
def test_claimed_ready_cannot_hide_invalid_actual_reply(module, tmp_path, key, value):
    targets = prepare_targets(tmp_path)
    observed = snapshot(module, targets)
    observed["responses"][targets[0]][key] = value
    assert not module.readiness(observed, now=100, required_nodes=targets)


@pytest.mark.parametrize(
    "targets",
    [
        [],
        ["/scope/node", "scope/node"],
        ["/scope/../node"],
        ["/scope//node"],
        ["/"],
        [True],
        ["/scope/node/"],
    ],
)
def test_invalid_endpoint_targets_refused_before_client_creation(module, tmp_path, targets):
    class NoClient:
        def create_client(self, *args):
            pytest.fail("invalid source targets reached SDK client construction")

    with pytest.raises(ValueError):
        module.LifecycleProbe(NoClient(), object(), tmp_path, required_nodes=targets)


def test_service_clients_match_exact_source_identities_without_double_slash(module, tmp_path):
    targets = prepare_targets(tmp_path)
    calls = []

    class CaptureOnlyNode:
        def create_client(self, service, endpoint):
            calls.append(endpoint)
            return object()

    probe = module.LifecycleProbe(CaptureOnlyNode(), object(), tmp_path, required_nodes=targets)
    try:
        assert calls == [name + "/get_state" for name in targets]
        assert set(probe.responses) == set(targets)
    finally:
        probe.close()
