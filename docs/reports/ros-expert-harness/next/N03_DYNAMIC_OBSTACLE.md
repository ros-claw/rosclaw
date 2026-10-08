# Temporal occupancy preparation (physical NOT_RUN)

This isolated branch prepares the accounting core while P0 physical pilots run.
It must not merge before the main coverage and repair optimizer PRs pass their
ordered acceptance gates. No Native/Gazebo dynamic success is claimed.

The new opt-in OccupancyAccounting adapter binds every complete occupied-cell
snapshot to frozen run/mission/geometry/frame, independent Gazebo source,
strictly contiguous sequence, identical CleaningPose SIM timestamp and fresh
source age. Integrity, missing geometry identity, reorder/time mismatch, stale
or incomplete observations latch a failure. Occupied cells never enter credit;
old valid credit survives later occupancy; removing an obstacle makes uncleaned
cells eligible again. Static accessible denominator is frozen and checked.

For dynamic accounting only, CoverageVerifier observes sampled brush footprints
without interpolation across unobserved obstacle motion. Its original gap and
speed checks remain. Old API defaults to the original interpolation and remains
compatible. Exact time pairing requires the future passive observer to produce
one occupancy snapshot for every pose from one ground-truth packet, rather than
attaching the latest asynchronously cached mask to old robot trajectories.

18 focused tests cover overlapping brush/obstacle credit, withdrawal/revisit,
later reoccupation, no interpolation through blocked cells, immutable denominator,
source/hash/clock/sequence/frame/completeness failures. Full ROS contract suite
passes342/10deselected (includes prior tests; not independent physical episodes).

Remaining before live acceptance: full timestamp/capture contract,
Gazebo model-pose and collision geometry producer, passive observer/actuator
process split, executor and canonical artifact replay integration, bounded
DEFERRED wait/revisit and actual Native D2/D4/D5/D6 episodes. Existing native
historical runs cannot substitute. The accounting adapter is not currently
wired into daemon dispatch. `v1_done=false`.

## Sealed collision projection preparation

The pure OccupancyProjector now reads exact bounded SDF bytes, binds their SHA256 and declared obstacle model set, and encloses all supported box/sphere/cylinder collision primitives (including link/collision offsets) in conservative model-centered discs. Visual-only, mesh, articulated/nested/included or ambiguous geometry fails closed. This can overestimate occupancy; it does not claim exact surface rasterization. One whole ground-truth packet must contain every declared model at the brush packet SIM timestamp in the frozen map frame. Missing/stale/duplicate/nonfinite observations and projection work over100000 cells return no partial snapshot. Both projector and accounting bind the full original grid/frame/brush geometry as well as the unchanged accessible denominator.

Latest validation:56 focused occupancy/projection tests;393 ROS contracts pass/10 integration deselected/1 existing pkg_resources deprecation warning. Two-module mypy and Ruff/format pass. These synthetic tests are not physical episodes. Producer transport, actual Gazebo source binding/scene identity, geometry capture from the actual loaded world, monotonic/capture-time contract, passive observer/actuator separation, daemon integration and fresh Native D2/D4/D5/D6 remain NOT_RUN. No occupancy adapter is wired into dispatch yet.

## Time-paired bounded recovery preparation

TimePairedRecovery separates ready/deferred/exhausted pending cells instead of deferring every cell in a mixed blocked component. It requires the latest successfully accounted occupancy sequence/time and explicit same-packet legal-route reachability; missing/unpaired evidence yields UNKNOWN with no dispatch proposal. Withdrawal restores candidates without cleaning credit, waiting consumes no attempt, canonical-attempt delivery remains idempotent, and the immutable bounded SIM deadline removes all motion proposals. The inherited unpaired propose entry is disabled. Existing MissedRegionRecovery and runtime defaults remain unchanged.

Latest full ROS contract result:407 passed/10 integration deselected/1 existing deprecation warning in8.43s;13 focused temporal recovery cases and mypy/Ruff/format pass. This is a state/proposal layer only: the actual legal-center route producer, bounded runtime wait with brush off and lease-safe stop, accounting integration, canonical blocked receipts and new Native D2/D4/D5 episodes remain unimplemented/unrun. Ordered P0 merge prerequisite is still pending.
