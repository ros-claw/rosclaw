"""Pure port geometry and actual host proc preflight; no DDS/socket/World."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("fixture_network")


@pytest.mark.parametrize("domain", [0, 81, 82, 100, 215, 231])
def test_reserved_domain_port_span_excludes_observed_linux_ephemeral_range(module, domain):
    result = module.validate_ros_domain(domain, ephemeral_range=(32768, 60999))
    assert result["transport_ready"] is False and result["authorization"] is False
    assert result["actual_container_network_namespace_verified"] is False


@pytest.mark.parametrize("domain", [101, 102, 201, 202, 214, 232, True, -1, 233])
def test_overlap_edge_participant_spill_or_unknown_domain_refuses_before_transport(module, domain):
    with pytest.raises(ValueError):
        module.validate_ros_domain(domain, ephemeral_range=(32768, 60999))


def test_custom_actual_ephemeral_range_does_not_use_hardcoded_safe_ids(module):
    with pytest.raises(ValueError, match="overlap"):
        module.validate_ros_domain(81, ephemeral_range=(20000, 40000))
    assert module.validate_ros_domain(202, ephemeral_range=(1000, 2000))["domain_id"] == 202
