"""Actual Core-source projection and worker file leakage contracts, no model."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.context.memory_intervention import native_memory_intervention
from rosclaw.connectors.ros.context.memory_worker_projection import (
    prepare_worker_projection,
    retrieve_worker_projection,
)
from tests.connectors.ros.test_memory_intervention_context import snapshot as snapshot


@pytest.mark.parametrize("mode", ["M0", "M1", "M2"])
def test_worker_receives_no_database_or_original_episode_and_native_uses_projection(
    snapshot, tmp_path, mode
):
    operator, declared, database, source = snapshot
    declared["mode"] = mode
    original = database.read_bytes(), source.read_bytes()
    worker = tmp_path / "worker"
    (worker / "data").mkdir(parents=True)
    projection = prepare_worker_projection(operator, "sim/ur5e", declared, worker / "data/memory")
    assert [p.name for p in (worker / "data/memory").iterdir()] == ["projection.json"]
    raw = (worker / "data/memory/projection.json").read_bytes()
    assert b"historical source advice" not in raw
    assert b"synthetic curated contract source" not in raw
    assert (b"guarded tools" in raw) is (mode == "M2")
    service = SimpleNamespace(
        _home=worker,
        _config=SimpleNamespace(raw={"agent": {"ros_expert": {"memory_intervention": projection}}}),
    )
    mission = SimpleNamespace(body_binding=SimpleNamespace(body_id="sim/ur5e"))
    result = native_memory_intervention(service, mission)
    assert result["mode"] == mode and result["authorization"] is False
    assert (not result["items"]) is (mode == "M0")
    assert (database.read_bytes(), source.read_bytes()) == original


def test_disabled_projection_never_reads_missing_operator_source(snapshot, tmp_path):
    home, declaration, database, source = snapshot
    declaration["mode"] = "M0"
    database.unlink()
    source.unlink()
    destination = tmp_path / "empty_projection"
    result = prepare_worker_projection(home, "sim/ur5e", declaration, destination)
    assert result["mode"] == "M0"
    assert json.loads((destination / "projection.json").read_bytes())["items"] == []


@pytest.mark.parametrize(
    "fault",
    [
        "changed_bytes",
        "foreign_body",
        "mode_upgrade",
        "guidance_leak",
        "extra_source_path",
        "text_leak",
    ],
)
def test_changed_or_leaking_projection_refuses_even_if_operator_digest_is_updated(
    snapshot, tmp_path, fault
):
    home, declaration, _, _ = snapshot
    declaration["mode"] = "M1"
    worker = tmp_path / "worker"
    (worker / "data").mkdir(parents=True)
    projected = prepare_worker_projection(home, "sim/ur5e", declaration, worker / "data/memory")
    path = worker / "data/memory/projection.json"
    path.chmod(0o600)
    result = json.loads(path.read_bytes())
    if fault == "foreign_body":
        projected["body_id"] = "foreign"
    elif fault == "mode_upgrade":
        projected["mode"] = "M2"
    elif fault == "guidance_leak":
        result["items"][0]["repair_pattern"] = "secret repair advice"
    elif fault == "extra_source_path":
        result["operator_database_path"] = "/host/history.sqlite"
    elif fault == "text_leak":
        result["layer_summary"] += "secret repair advice"
    else:
        result["items"][0]["facts"]["component"] = "changed"
    path.write_text(json.dumps(result))
    if fault != "changed_bytes":
        projected["projection_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        retrieve_worker_projection(worker, "sim/ur5e", projected)


def test_projection_refuses_existing_destination_without_overwriting(snapshot, tmp_path):
    home, declaration, _, _ = snapshot
    destination = tmp_path / "existing"
    destination.mkdir()
    protected = destination / "original"
    protected.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        prepare_worker_projection(home, "sim/ur5e", deepcopy(declaration), destination)
    assert protected.read_bytes() == b"keep"


@pytest.mark.parametrize("mode", ["M0", "M1", "M2"])
def test_actual_runtime_config_disables_all_other_historical_channels(monkeypatch, tmp_path, mode):
    from rosclaw.core import runtime
    from rosclaw.mcp.adapters.runtime_client import RuntimeClient

    observed = []

    class RuntimeWithoutInitialization:
        def __init__(self, config):
            observed.append(config)

        def initialize(self):
            pass

    monkeypatch.setattr(runtime, "Runtime", RuntimeWithoutInitialization)
    declared = {
        "schema_version": "rosclaw.memory_worker_runtime.v1",
        "mode": mode,
        "retrieval_source": "worker_projection_only",
    }
    client = RuntimeClient(
        project_root=tmp_path,
        robot_id="sim/ur5e",
        runtime_profile={"ros_expert_memory_experiment": declared},
        daemon_client=object(),
    )
    assert client._ensure_runtime() is not None
    cfg = observed[0]
    assert not cfg.enable_memory and not cfg.enable_knowledge and not cfg.enable_how
    assert not cfg.enable_recovery_loop and not cfg.knowledge.enabled
    assert cfg.knowledge_v2_mode == "disabled" and cfg.knowledge.url is None
    assert cfg.knowledge.api_key is None
    assert cfg.enable_firewall and cfg.enable_practice


@pytest.mark.parametrize("mode", ["M0", "M1", "M2"])
async def test_canonical_memory_tool_cannot_bypass_worker_projection(tmp_path, mode, monkeypatch):
    from rosclaw.mcp.adapters.runtime_client import RuntimeClient
    from rosclaw.mcp.schemas.common import MCPError

    client = RuntimeClient(
        project_root=tmp_path,
        robot_id="sim/ur5e",
        daemon_client=object(),
        runtime_profile={
            "ros_expert_memory_experiment": {
                "schema_version": "rosclaw.memory_worker_runtime.v1",
                "mode": mode,
                "retrieval_source": "worker_projection_only",
            }
        },
    )
    monkeypatch.setattr(
        client, "_adapters", lambda: pytest.fail("global Memory adapter was accessed")
    )
    with pytest.raises(MCPError) as error:
        await client.query_memory("past repair")
    assert error.value.code == "MEMORY_RETRIEVAL_DISABLED"


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {
            "schema_version": "rosclaw.memory_worker_runtime.v1",
            "mode": "M0",
            "retrieval_source": "global",
        },
    ],
)
def test_runtime_intervention_cannot_silently_enable_global_sources(bad):
    from rosclaw.connectors.ros.context.memory_worker_projection import worker_runtime_overrides

    with pytest.raises(ValueError):
        worker_runtime_overrides(bad)


@pytest.mark.parametrize("mode", ["M0", "M1", "M2"])
def test_worker_source_builder_freezes_same_mode_and_actual_mcp_profile(snapshot, tmp_path, mode):
    import yaml

    from rosclaw.agent.detectors import build_project_profile
    from rosclaw.connectors.ros.context.memory_worker_projection import (
        prepare_registered_worker_sources,
    )

    home, declaration, _, _ = snapshot
    declaration["mode"] = mode
    worker = tmp_path / "registered_worker"
    manifest = prepare_registered_worker_sources(home, "sim/ur5e", declaration, worker)
    profile = build_project_profile(
        project_root=worker, profile=str(worker / "runtime.yaml"), robot="sim/ur5e"
    )
    assert profile.runtime_profile["ros_expert_memory_experiment"]["mode"] == mode
    patch = manifest["native_agent_patch"]
    assert patch["agent"]["ros_expert"]["memory_intervention"]["mode"] == mode
    for path, sha in manifest["files"].items():
        assert hashlib.sha256((worker / path).read_bytes()).hexdigest() == sha
    assert manifest["requires_independent_container_mount_and_tool_route_admission"] is True
    assert manifest["complete_OS_tool_isolation_verified"] is False
    assert manifest["authorization"] is False
    assert yaml.safe_load((worker / "runtime.yaml").read_text()) == profile.runtime_profile
    all_bytes = b"\n".join(p.read_bytes() for p in worker.rglob("*") if p.is_file())
    assert b"historical source advice" not in all_bytes
    assert (b"guarded tools" in all_bytes) is (mode == "M2")
