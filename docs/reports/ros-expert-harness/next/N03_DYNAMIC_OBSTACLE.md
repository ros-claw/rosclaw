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

Remaining before live acceptance: full timestamp/capture/geometry contract,
Gazebo model-pose and collision geometry producer, passive observer/actuator
process split, executor and canonical artifact replay integration, bounded
DEFERRED wait/revisit and actual Native D2/D4/D5/D6 episodes. Existing native
historical runs cannot substitute. The accounting adapter is not currently
wired into daemon dispatch. `v1_done=false`.
