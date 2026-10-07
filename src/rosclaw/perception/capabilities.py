"""Catalog and FIXTURE gateway registration for supplied perception quality.

Registers the two bounded supplied-record capabilities
(``perception.scan_quality`` / ``perception.cloud_quality``) as genuine
factory-backed catalog tools (COMPUTE / no side effect / DERIVED evidence,
body-agnostic) and as private FIXTURE-only ``ActionGateway`` executors.

The catalog executors are pure compute over supplied records; declared
supported modes (SIMULATION/SHADOW/REAL) denote the resolver context in
which the pure computation may be selected, never a physical grant. The
gateway executors are registered for ``ExecutionMode.FIXTURE`` only and
produce ``ActionExecutionResult`` observations carrying the supplied-record
provenance (``source=supplied_observation`` plus the canonical input digest)
and the canonical result; evidence domain FIXTURE / level SYNTHETIC, so the
canonical receipt degrades honestly. No REAL/SHADOW/SIMULATION gateway
executor is registered here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from rosclaw.agentd.tooling.catalog import ToolCatalog
from rosclaw.contracts.agent.tool import (
    ExecutionClass,
    ToolDescriptorV2,
    ToolEvidenceClass,
)
from rosclaw.kernel import ExecutionMode
from rosclaw.kernel.contracts import (
    ActionExecutionResult,
    ActionState,
    EvidenceDomain,
    EvidenceLevel,
)
from rosclaw.perception.supplied_observations import (
    analyze_supplied_cloud,
    analyze_supplied_scan,
)

SCAN_QUALITY_TOOL = "perception.scan_quality"
CLOUD_QUALITY_TOOL = "perception.cloud_quality"

_SOURCE = "native:agentd"
_VERIFIER = "bounded-admission+canonical-result"
_SUPPORTED_MODES = ["SIMULATION", "SHADOW", "REAL"]

_SCAN_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "valid_count": {"type": "integer"},
        "invalid_count": {"type": "integer"},
        "nearest_index": {"type": ["integer", "null"]},
        "nearest_distance": {"type": ["number", "null"]},
    },
    "required": [
        "status",
        "valid_count",
        "invalid_count",
        "nearest_index",
        "nearest_distance",
    ],
    "additionalProperties": False,
}

_CLOUD_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "total_count": {"type": "integer"},
        "valid_count": {"type": "integer"},
        "invalid_count": {"type": "integer"},
        "min_xyz": {"type": ["array", "null"]},
        "max_xyz": {"type": ["array", "null"]},
        "centroid_xyz": {"type": ["array", "null"]},
    },
    "required": [
        "status",
        "total_count",
        "valid_count",
        "invalid_count",
        "min_xyz",
        "max_xyz",
        "centroid_xyz",
    ],
    "additionalProperties": False,
}

_ANALYZERS = {
    SCAN_QUALITY_TOOL: analyze_supplied_scan,
    CLOUD_QUALITY_TOOL: analyze_supplied_cloud,
}

_ABS_LIMIT = 1e12
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1

_SCAN_SCALAR_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "number", "minimum": -_ABS_LIMIT, "maximum": _ABS_LIMIT},
        {"type": "string", "enum": ["NAN", "POS_INF", "NEG_INF"]},
    ]
}

_HEADER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["stamp", "frame_id"],
    "properties": {
        "stamp": {
            "type": "object",
            "additionalProperties": False,
            "required": ["sec", "nanosec"],
            "properties": {
                "sec": {"type": "integer", "minimum": _INT32_MIN, "maximum": _INT32_MAX},
                "nanosec": {"type": "integer", "minimum": 0, "maximum": 999_999_999},
            },
        },
        "frame_id": {"type": "string", "maxLength": 128},
    },
}

_SCAN_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["record", "reference_time_ns"],
    "properties": {
        "reference_time_ns": {
            "type": "integer",
            "minimum": _INT64_MIN,
            "maximum": _INT64_MAX,
        },
        "record": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "header",
                "angle_min",
                "angle_max",
                "angle_increment",
                "time_increment",
                "scan_time",
                "range_min",
                "range_max",
                "ranges",
                "intensities",
            ],
            "properties": {
                "header": _HEADER_SCHEMA,
                "angle_min": _SCAN_SCALAR_SCHEMA,
                "angle_max": _SCAN_SCALAR_SCHEMA,
                "angle_increment": _SCAN_SCALAR_SCHEMA,
                "time_increment": _SCAN_SCALAR_SCHEMA,
                "scan_time": _SCAN_SCALAR_SCHEMA,
                "range_min": _SCAN_SCALAR_SCHEMA,
                "range_max": _SCAN_SCALAR_SCHEMA,
                "ranges": {
                    "type": "array",
                    "maxItems": 4096,
                    "items": _SCAN_SCALAR_SCHEMA,
                },
                "intensities": {
                    "type": "array",
                    "maxItems": 4096,
                    "items": _SCAN_SCALAR_SCHEMA,
                },
            },
        },
    },
}

_UINT16_SCHEMA: dict[str, Any] = {"type": "integer", "minimum": 0, "maximum": 65535}
_UINT32_SCHEMA: dict[str, Any] = {"type": "integer", "minimum": 0, "maximum": _UINT32_MAX}

_CLOUD_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["record"],
    "properties": {
        "record": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "header",
                "width",
                "height",
                "point_step",
                "row_step",
                "is_bigendian",
                "is_dense",
                "fields",
                "data",
            ],
            "properties": {
                "header": _HEADER_SCHEMA,
                "width": _UINT16_SCHEMA,
                "height": _UINT16_SCHEMA,
                "point_step": _UINT16_SCHEMA,
                "row_step": _UINT32_SCHEMA,
                "is_bigendian": {"type": "boolean"},
                "is_dense": {"type": "boolean"},
                "fields": {
                    "type": "array",
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "offset", "datatype", "count"],
                        "properties": {
                            "name": {"type": "string", "maxLength": 64},
                            "offset": _UINT32_SCHEMA,
                            "datatype": {"type": "integer", "minimum": 0, "maximum": 255},
                            "count": _UINT32_SCHEMA,
                        },
                    },
                },
                "data": {
                    "type": "array",
                    "maxItems": 65536,
                    "items": {"type": "integer", "minimum": 0, "maximum": 255},
                },
            },
        }
    },
}

_INPUT_SCHEMAS = {
    SCAN_QUALITY_TOOL: _SCAN_INPUT_SCHEMA,
    CLOUD_QUALITY_TOOL: _CLOUD_INPUT_SCHEMA,
}


def _supplied_input_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def register_perception_tools(catalog: ToolCatalog) -> None:
    """Register both supplied-perception tools with genuine async executors."""
    specs = [
        (
            SCAN_QUALITY_TOOL,
            "Analyze a bounded supplied LaserScan-like record with the "
            "canonical pure quality algorithm (no sensor, no ROS).",
            _SCAN_OUTPUT_SCHEMA,
        ),
        (
            CLOUD_QUALITY_TOOL,
            "Analyze a bounded supplied PointCloud2-like record with the "
            "canonical pure quality algorithm (no sensor, no ROS).",
            _CLOUD_OUTPUT_SCHEMA,
        ),
    ]
    for tool_id, description, output_schema in specs:
        descriptor = ToolDescriptorV2(
            tool_id=tool_id,
            source=_SOURCE,
            execution_class=ExecutionClass.COMPUTE,
            description=description,
            input_schema=_INPUT_SCHEMAS[tool_id],
            output_schema=output_schema,
            supported_modes=list(_SUPPORTED_MODES),
            required_body_types=[],
            evidence_class=ToolEvidenceClass.DERIVED,
            verifier=_VERIFIER,
            reliability=1.0,
            typical_latency_ms=1,
        )
        analyzer = _ANALYZERS[tool_id]

        async def _exec(arguments: dict[str, Any], _analyzer: Any = analyzer) -> dict[str, Any]:
            return _analyzer(arguments)

        catalog.register(descriptor, _exec)


def register_perception_fixture_executors(gateway: Any) -> None:
    """Register FIXTURE-only ActionGateway executors for both capabilities."""
    for tool_id in (SCAN_QUALITY_TOOL, CLOUD_QUALITY_TOOL):
        analyzer = _ANALYZERS[tool_id]

        def _execute(action: Any, _analyzer: Any = analyzer) -> ActionExecutionResult:
            payload = action.arguments
            result = _analyzer(payload)
            return ActionExecutionResult(
                final_state=ActionState.COMPLETED,
                evidence_level=EvidenceLevel.SYNTHETIC,
                evidence_domain=EvidenceDomain.FIXTURE,
                observations=[
                    {
                        "source": "supplied_observation",
                        "capability_id": action.capability_id,
                        "input_sha256": _supplied_input_sha256(payload),
                        "result": result,
                    }
                ],
            )

        gateway.register_executor(tool_id, ExecutionMode.FIXTURE, _execute)


__all__ = [
    "CLOUD_QUALITY_TOOL",
    "SCAN_QUALITY_TOOL",
    "register_perception_fixture_executors",
    "register_perception_tools",
]
