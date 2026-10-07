# Supplied perception quality capabilities (Stage B, FIXTURE-only)

This stage registers two bounded supplied-record perception capabilities as
pure source — no sensor, ROS, DDS, NN, physics or hardware initialization.

## Capabilities

| Capability | Adapter | Canonical analyzer |
|---|---|---|
| `perception.scan_quality` | `analyze_supplied_scan(payload)` | `rosclaw.perception.scan_quality.analyze_scan` |
| `perception.cloud_quality` | `analyze_supplied_cloud(payload)` | `rosclaw.perception.cloud_quality.analyze_cloud` |

## Bounded admission

`src/rosclaw/perception/supplied_observations.py` enforces exact plain
`dict`/`list`/`str`/`int`/`float`/`bool` types (subclass instances are
rejected), exact key sets, and list/string/numeric bounds **before** any
message-object construction, `ranges`/`fields` copy, or `bytes(data)`
conversion. Bare JSON numbers are finite with `abs <= 1e12`; non-finite scan
scalars are admitted only as the explicit string tags `NAN`, `POS_INF`,
`NEG_INF` and mapped to IEEE values after admission so the canonical
algorithm's own quality rules classify them. Malformed input raises
`TypeError`/`ValueError` before the canonical analyzer runs.

Cloud `width`/`height` are admitted as declared metadata (0..65535): a
declared 17x1 cloud reaches the canonical analyzer and yields
`QUALITY_INVALID` with `total_count == invalid_count == 17` — a quality
verdict, not a transport failure.

The canonical scan/cloud algorithms are unchanged; adapter output is exactly
the canonical result dict over admitted records.

## Registration

- `register_native_tools` (`src/rosclaw/agentd/tooling/native_tools.py`) calls
  `register_perception_tools(catalog)`: genuine factory-backed async catalog
  executors with `COMPUTE`/`NONE`/`DERIVED` descriptors, body-agnostic,
  `verifier` declared, no physical grant. Declared supported modes
  (SIMULATION/SHADOW/REAL) denote pure-compute resolver context only.
- `Runtime.__init__` (`src/rosclaw/core/runtime.py`) calls
  `register_perception_fixture_executors(self._action_gateway)` right after
  normal gateway creation: **FIXTURE-only** executors. No REAL/SHADOW/
  SIMULATION gateway executor is registered; all existing gateway
  mode/evidence/session/authorization/resource/deadline gates are preserved.
- Gateway `observations[0]` exposes `source=supplied_observation`,
  `input_sha256=SHA256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
  allow_nan=False))`, and the canonical `result`. Evidence domain FIXTURE,
  level SYNTHETIC; the canonical receipt final state is DEGRADED and App
  trust is SYNTHETIC — never physical completion.

## Example App

`examples/apps/perception-quality/app.yaml` calls the two capabilities in
order with `${input.scan}` / `${input.cloud}`. `examples/perception_capability_fixture.py`
installs it by explicit local path through the existing `AppStore` and runs it
through a real private Unix `DaemonControlPlane`/`DaemonClient`/`AppRunner`
— two terminal journal-bound FIXTURE receipts, scoped session closed.

Source correctness is demonstrated here; live sensor or physical availability
is **not** claimed.
