# SDK engineering failure capture and staging repair episodes

G4 requires actual failure observations, a minimal staging patch, a new-World
L1/L2 validation and rollback evidence for five cases. This implementation is
separate from the frozen G3 source and grants no Agent ROS or Runtime authority.

The explicit isolated SDK failure-capture role preserves current original Native,
system, candidate-review and physical observations with an unadmitted zero-command
sink. It requires an UNKNOWN/BLOCKED engineering assessment, preserves the separate
original candidate result (which can remain PROPOSED), records L1 NOT_READY and
L2 NOT_RUN, and does
not compile a navigation workspace from an unready candidate. Only the actual
localization service must be active to perform the operator's initial pose setup;
that preparation condition is distinct from a readiness assessment. Ordinary
captures retain the complete ACTIVE/proposed/readiness path.

Staging can declare the Nav2 lifecycle manager's actual managed-node list. The
SDK launch respects that list rather than silently replacing an omitted node.
Unknown, duplicate or empty manager lists are rejected before launch.

The scan-stale scenario explicitly declares a bridge paused after initialization.
The SDK actor pins and pauses only its owned original scan bridge and resumes it
after failure capture. A repair must change the staging policy in a new World.
The duplicate-route scenario creates one SDK-owned advertised endpoint on the
declared original protected command topic; it publishes no messages. Both modes
require the explicit engineering role. Geometry, URDF, Body bounds and freshness
thresholds are unchanged.

Initial software checks: 24 focused role/freshness/configuration tests pass;
expanded ROS regression: 1,169 pass, 11 deselected with one existing dependency
deprecation warning. Actual before/after/rollback episodes have not yet passed.
No G4 acceptance or actual model experiment is claimed by these software tests.

The circle geometry diagnostic uses the original unique costmap publisher, fresh
parameter readback, matching base frame, explicitly empty footprint polygon and
unchanged URDF envelope. An undersized circle blocks engineering review without
promoting discovery to Body binding or bypassing the full L1 geometry checks.
The first footprint capture timeout is retained: discovery correctly remained
PROPOSED with collision geometry explicitly unverified.
