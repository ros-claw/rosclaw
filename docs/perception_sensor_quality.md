# ADR: Canonical Pure Perception Sensor Quality Utilities (Stage A)

## Status

Accepted (Stage A — SOURCE ONLY). This stage relocates already-tested prior
Native algorithms into canonical pure utility modules. No capability, App,
registration, dispatch, ROS graph participation, or hardware access is
introduced or claimed. Stage B registration/dispatch is a separate, later
decision.

## Context

Two protected prior Native diagnostic apps (`scan_client.py`,
`cloud_client.py`) carried genuine local generated-DDS software evidence:
their pure analyzer functions were exercised against generated
`sensor_msgs/msg/LaserScan` and `sensor_msgs/msg/PointCloud2` CDR fixtures
plus independent public invariants. That evidence is software-only: no
hardware, physical, or contact claims exist or are made here.

## Decision

Relocate the prior non-`main` algorithm functions and all literal constants
**unchanged** into canonical pure modules:

- `src/rosclaw/perception/scan_quality.py` — `analyze_scan(message, reference_time_ns)`
- `src/rosclaw/perception/cloud_quality.py` — `analyze_cloud(message)`

Only type annotations and docstrings were normalized. Thresholds
(`FRESHNESS_MAX_AGE_NS`, `FUTURE_TOLERANCE_NS`, `OBSTACLE_DISTANCE_M`,
`ANGLE_MAX_TOL`) and cloud bounds (width/height each 1..16, i.e. at most
256 decoded points; point/row step limits, datatype table, declared-count
convention) are inherited verbatim and must not be broadened in this
stage.

### Explicit reference clock

`analyze_scan` never reads a wall clock. Freshness (STALE/FUTURE) is judged
against the caller-supplied signed integer `reference_time_ns`. Signed
`Time.sec` (int32) stamps are valid; negative stamps are co-timed against
negative references.

### Resource boundary and provenance

Both analyzers are pure, deterministic functions over duck-typed message
fields, but neither guarantees real-time or resource safety for arbitrary
inputs:

- `analyze_scan` copies `list(message.ranges)` at entry; work and
  allocation are linear O(n) in the supplied scan range length. There is
  no scan-size cap and no arbitrary iterator termination guarantee.
- `analyze_cloud` accepts `width` and `height` each in 1..16, so at most
  256 points are decoded. However, the `len(bytes(msg.data))` conversion
  precedes data-length rejection, and `fields`/`name` traversal can be
  arbitrarily large for duck-typed inputs; there is no total allocation
  or work guarantee.

Bounded admission before conversion/traversal is a Stage B concern and is
not implemented here. `analyze_cloud` checks the layout envelope before
decoding any point; metadata (frame_id, is_dense) is provenance only and
does not affect the verdict.

## Offline example

`examples/perception_sensor_quality.py` invokes the canonical API on an
ordinary JSON file (nested dicts become attribute objects; cloud `data`
integer lists become bytes-compatible) and prints one JSON result:

```bash
python examples/perception_sensor_quality.py --case scan --input ABS_JSON --reference-time-ns SIGNED_INT
python examples/perception_sensor_quality.py --case cloud --input ABS_JSON
```

No middleware, network, or actuation occurs.

## Consequences

- The modules import on ordinary Python 3.11 without `rclpy` or
  `sensor_msgs` installed.
- Prior generated-CDR behavior is pinned by the integration checker; any
  algorithm change or source edit invalidates the Stage A receipt.
- Real-time safety, executor placement, and App capability registration
  remain out of scope until a separate Stage B decision with its own
  evidence.
