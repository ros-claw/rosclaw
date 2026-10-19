# Optional source-bound scan processing in owned SDK fixtures

A new camera-equipped Body passed complete Source matching and full L1, then
its first canonical navigation goal timed out. The original receipt remains
TIMED_OUT with `physical_stop_verified=false`. All 41 sink velocity messages
were zero. A separate zero-goal diagnostic World recorded 27 original scans;
each contains 12 returns inside the current source collision envelope. Own-body
returns triggering CollisionMonitor is a hypothesis, not a confirmed cause.

The optional `source_body_scan_filter` bringup declaration adds the existing
official `laser_filters/StaticLaserScanPolygonFilter` to disposable SDK fixtures.
It retains the raw scan and routes only CollisionMonitor's scan subscription to
the filtered stream. AMCL, costmaps, command routes, protection polygons,
`min_points`, sensor height limits, Body limits and Source/L1 checks are unchanged.
No Runtime, driver, goal, lease, permit or control publisher is added.

The closed declaration contains only its schema version, current expanded URDF
SHA256, base frame and scan frame. The implementation derives its 32-sided
polygon from the current complete original collision geometry, including
SHA-bound mesh originals, with zero padding. It requires a unique fixed original
scan mount and one existing GZ_TO_ROS LaserScan bridge. It accepts no user polygon,
radius, executable, package URL or control route. Driver-only bringup rejects it.

One exact arm64 Jazzy package is extracted inside the isolated container, without
host installation or an underlay upgrade:

- `ros-jazzy-laser-filters` `2.0.10-5noble.20260916.111526`
- deb SHA256 `49756059a9f5bde5416c678260feb4513dc825751ed4f4ff44c40612a52634d9`
- operator prerequisite: verify the original ROS InRelease signature, its Packages
  binding and the exact package bytes before placing
  `/assets/source-body-laser-filters.deb` in the pinned read-only asset inventory.

All filter nodes and the plugin's TF listener use simulation time. A missing TF
produces no scan; internal points become NaN rather than fabricated free space.
The stack owner supervises and terminates the filter with its other children.

## Evidence and limitations

The separately pinned stock package passed a historical replay of all 27 real
scan CDRs: exactly 12 points per scan were masked, every external finite range
was unchanged, and restoring only the ranges reproduced each original serialized
message byte for byte. The original bag was unchanged. An initial missing-TF
window produced zero outputs. This used no new physics World or navigation goal.

The production helper has source/geometry/scope/bridge failure-injection tests;
the current relevant regression passed 2317 tests, with one existing console skip
and ten default deployment/integration exclusions. Practice passed 183 tests
with nine optional-dependency skips. A production-helper SDK replay stopped at
Docker create because the previously pinned SDK image disappeared from the
shared environment. No process or World started in that failed attempt.

An enclosing collision envelope is not exact solid-body occupancy. A point
inside it is not thereby proven to be a self-return, and a concave or articulated
Body may need a more precise provider. This component is an explicit operator
SDK experiment; it grants no hardware or generic-robot acceptance. The causal
hypothesis and new-Body navigation require separately frozen fresh-World
After/Rollback observations, canonical results, independent collision and stop
checks, and full current Source/L1 continuity. A package replay or metadata
component pass cannot satisfy those obligations.

The existing model pilot remains NO_GO. These operator development cases are
not unseen model cases, and there is no demonstrated causal Memory benefit.
