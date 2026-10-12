# Profile-free passive SIM observation and adapted held-out L1

This change lets the existing SDK witness consume separately SHA-pinned source
inputs instead of selecting a robot profile. It remains source-only: generic
mode cannot forward velocity or enable a lease. Runtime and MCP entrypoints do
not change. Native NavSat ECM components are now identified explicitly; unknown
sensor components still make independent source binding UNKNOWN.

## Scope and source lineage

The generic observer was frozen at `ff58562faff7b368c48f2d1dea0e61fa8c134083`
before selecting Clearpath Jackal J100 with the official Hokuyo UST macro.
The original known-robot World passed source binding and fresh L1. Independent
recomputation checked 3,037 original contact messages and 4,050 contact pairs,
with zero unexpected contacts. That experiment is known-fixture evidence.

Held-out selection then pinned official `clearpathrobotics/clearpath_common`,
`jazzy`, `69b6da109b14117b5fb693789747317a52ff973f`. The operator wrapper assembles
the vendor J100 and Hokuyo macros. SDK xacro export and actual `gz sdf -p`
conversion preserve collision, inertial and joint trees. Visual resource URI
staging changes no collision geometry. All six collision elements resolve; the
URDF-derived conservative radius is 0.3690000995497801 m. This is a different
vendor family from the original Waffle/Burger fixtures, not a renamed or scaled
fixture. No claim is made about foundation-model pretraining data.

The initial frozen code did **not** pass held-out acceptance. The complete
source-only capture at attempt08 returned UNKNOWN because the original ECM
observer classified the vendor GPS sensor as UNKNOWN. The vendor sensor was
retained. Commit `948e9b2583dac8422d261af49cc10cf07fd6794e` adds native NavSat
component classification, without changing source-matching criteria. The earlier
`8e4585f0` commit reproduces the frozen generic change on main `397d83f2`.
Results after this repair are adapted held-out acceptance, not zero-shot success.

## Actual SDK L1 acceptance

One repaired first-capture World (attempt09) returned source MATCHED and L1
READY_FOR_SIM_TEST. Independent original-data recomputation verified 3,577
contact messages/pairs and zero unexpected contacts.

A subsequent registered matrix created seven independent Worlds:

| Case | Fresh L1 outcome |
| --- | --- |
| Normal source-only capture | READY_FOR_SIM_TEST |
| Sensor stop | UNKNOWN |
| Clock freeze | UNKNOWN |
| TF loss | UNKNOWN |
| Lifecycle inactive | UNKNOWN |
| QoS mismatch | UNKNOWN |
| Namespace mismatch | UNKNOWN |

Independent replay reproduced Native-to-system-model and each original L1
result exactly at its original review time. Replay adds no physical runs.
All seven Worlds had zero navigation goals, zero velocity publications, zero
unexpected contacts and closed leases; all six fault actors exited with code 0
and every owned container stopped. TF injection pauses the owned Nav2 subtree
and also affects localization/costmaps; it is not a TF-only causal experiment.

Operator staging explicitly supplies `/heldout/j100`, four vendor wheel joints,
the vendor laser extrinsic, SIM time, odometry TF, a 0.2 s command timeout and a
conservative Nav2 radius of 0.40 m. Initial localization is an operator-owned
SetInitialPose setup with nondegenerate covariance, not autonomous discovery.
Two original Ogre2 rendering crashes are retained. The passing repaired runs
use the pinned SDK's Mesa software rendering; no sensor values are fabricated.
Earlier missing metadata, output-name collision, controller YAML namespace,
TF-frame/topic mix-up and mixed-clock failures remain in the original record.

## Validation and limits

On code source `948e9b25`: 1,191 tests passed, 11 skipped, 11 deselected in ROS,
MCP, architecture and Practice scopes excluding integration/acceptance/slow.
Compileall, src/tests Ruff, changed-file Ruff/format and focused mypy passed.
The whole-repository Ruff/format baseline remains failing: 290 existing Ruff
diagnostics and 606 formatting differences, with no new differences against
main `397d83f2`. SDK C++ build passed with `-Wall -Wextra -Werror`.

Local original evidence and independent reviews are under
`recovery_2026-10-12/G3-J100-L1-matrix-v1-*`; the public allowlisted archive is
`evidence/2026-10-12/G3-generic-observer-and-adapted-J100-L1-originals`.
Private runtime homes and ledgers are excluded. These paths belong to the local
acceptance workspace, not the git checkout. Failed first attempts are preserved.

This supplies local held-out L1 evidence only. Normal PR/main integration and
fresh merged-main acceptance are still separate gates. Generic L2 continuity,
leased MCP/daemon navigation, three seeds, ten execution faults, five repair
loops and actual paired A/B/C experiments are unfinished. REAL remains excluded;
the overall ROS Expert Harness v1 is not complete.
