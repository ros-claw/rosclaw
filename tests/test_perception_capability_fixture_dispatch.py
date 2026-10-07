"""Native tests for catalog registration and FIXTURE-only gateway dispatch."""

import asyncio
import hashlib
import json
import struct
from datetime import UTC, datetime, timedelta

from rosclaw.agentd.tooling.catalog import ToolCatalog
from rosclaw.agentd.tooling.native_tools import register_native_tools
from rosclaw.agentd.tooling.resolver import FilterContext, ToolResolver
from rosclaw.agentd.tools import BuiltinToolRegistry
from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.kernel import ActionEnvelope, ExecutionMode
from rosclaw.perception.capabilities import CLOUD_QUALITY_TOOL, SCAN_QUALITY_TOOL

IDS = [SCAN_QUALITY_TOOL, CLOUD_QUALITY_TOOL]


def _scan_payload():
    return {
        "record": {
            "header": {"stamp": {"sec": -1, "nanosec": 0}, "frame_id": "controlled"},
            "angle_min": 0.0,
            "angle_max": 0.25,
            "angle_increment": 0.25,
            "time_increment": 0.0,
            "scan_time": 0.0,
            "range_min": 0.1,
            "range_max": 4.0,
            "ranges": [1.0, 2.0],
            "intensities": [],
        },
        "reference_time_ns": -1_000_000_000,
    }


def _cloud_payload():
    return {
        "record": {
            "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "controlled"},
            "width": 1,
            "height": 1,
            "point_step": 12,
            "row_step": 12,
            "is_bigendian": False,
            "is_dense": True,
            "fields": [
                {"name": "x", "offset": 0, "datatype": 7, "count": 1},
                {"name": "y", "offset": 4, "datatype": 7, "count": 1},
                {"name": "z", "offset": 8, "datatype": 7, "count": 1},
            ],
            "data": list(struct.pack("<fff", 0.5, 0.25, 1.0)),
        }
    }


def _digest(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _runtime():
    config = RuntimeConfig(
        robot_id="supplied-fixture",
        enable_firewall=False,
        enable_memory=False,
        enable_practice=False,
        enable_skill_manager=False,
        enable_knowledge=False,
        enable_how=False,
        enable_auto=False,
        enable_provider=False,
        enable_sense=False,
        enable_event_persistence=False,
        enable_tracing=False,
    )
    return Runtime(config)


def test_catalog_registration_and_resolver_selection():
    catalog = ToolCatalog()
    register_native_tools(
        catalog,
        BuiltinToolRegistry(body_id="controlled", body_summary="source-only records"),
        simulation=False,
    )
    resolver = ToolResolver(catalog)
    selected = resolver.resolve(FilterContext(mode="SIMULATION"), candidates=IDS)
    assert {tool.tool_id for tool in selected.injected} == set(IDS)
    for tool_id in IDS:
        descriptor = catalog.get(tool_id)
        assert descriptor.execution_class.value == "COMPUTE"
        assert descriptor.evidence_class.value == "DERIVED"
        assert descriptor.required_body_types == []
        assert descriptor.model_callable
        assert descriptor.verifier


def test_catalog_execute_v2_genuine_computation():
    catalog = ToolCatalog()
    register_native_tools(
        catalog,
        BuiltinToolRegistry(body_id="controlled", body_summary="source-only records"),
        simulation=False,
    )
    payload = _scan_payload()
    envelope = asyncio.run(catalog.execute_v2("call-1", SCAN_QUALITY_TOOL, payload))
    assert envelope.status.value == "SUCCEEDED"
    assert envelope.value["status"] == "CLEAR"
    malformed = _scan_payload()
    malformed["extra"] = 0
    failed = asyncio.run(catalog.execute_v2("call-2", SCAN_QUALITY_TOOL, malformed))
    assert failed.status.value in {"FAILED", "BLOCKED"}
    catalog.quarantine_tool(SCAN_QUALITY_TOOL, "test")
    blocked = asyncio.run(catalog.execute_v2("call-3", SCAN_QUALITY_TOOL, payload))
    assert blocked.status.value == "BLOCKED"


def test_runtime_constructor_registers_fixture_only_executors():
    runtime = _runtime()
    registered = set(runtime.action_gateway.registered_executors)
    for tool_id in IDS:
        assert f"{tool_id}:FIXTURE" in registered
        for mode in ("REAL", "SHADOW", "SIMULATION"):
            assert f"{tool_id}:{mode}" not in registered


def test_fixture_dispatch_receipt_provenance():
    runtime = _runtime()
    for tool_id, payload in (
        (SCAN_QUALITY_TOOL, _scan_payload()),
        (CLOUD_QUALITY_TOOL, _cloud_payload()),
    ):
        action = ActionEnvelope(
            actor_id="test",
            agent_framework="test",
            session_id="test-session",
            body_id="supplied-fixture",
            capability_id=tool_id,
            arguments=payload,
            execution_mode=ExecutionMode.FIXTURE,
            deadline_at=datetime.now(UTC) + timedelta(seconds=5),
        )
        receipt = runtime.action_gateway.submit(action).to_dict()
        assert receipt["final_state"] == "DEGRADED"
        assert receipt["evidence_level"] == "SYNTHETIC"
        assert receipt["evidence_domain"] == "FIXTURE"
        observation = receipt["observations"][0]
        assert observation["source"] == "supplied_observation"
        assert observation["input_sha256"] == _digest(payload)
        assert observation["result"]["status"] in {"CLEAR", "VALID_CLOUD"}
