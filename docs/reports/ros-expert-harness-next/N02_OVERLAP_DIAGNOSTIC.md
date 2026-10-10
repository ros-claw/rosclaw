# Additional main-coverage overlap diagnostic (physical NOT_RUN)

The registered five-seed pose-aware pilots retain all original results on
3c0f17d2. Their completed pairs still show many repair goals and SIM gains below
the joint30% target. Historical goal diagnostics show varying terminal-position
error; its exact controller/localization cause remains UNKNOWN. No ground-truth
position is used to compensate a dispatched target.

An additional opt-in `perimeter_stateless_overlap` preset narrows operation
spacing, leaving Body geometry, actual brush, fixed denominator, original speed,
guards, retry/cost/compute budgets and Nav2 path ownership unchanged. It retains
the existing stateless controller and nine-goal boundary pass. The existing
baseline and old candidate presets remain byte-equivalent configurations.

| Known fixture | Spacing before → candidate | Headland | Ideal main coverage before → candidate | Planned length before → candidate | Static clearance |
|---|---:|---:|---:|---:|---:|
| Waffle | .45 → .40m | .50m | 59.8714 → 68.2047% | 9.6929 → 12.0876m | .124m |
| Burger | .30 → .27m | .35m | 78.7068 → 80.6299% | 21.4282 → 23.935m | .074m |

These are fixed-image Fields2Cover predictions, not measured coverage. The
original verifier projected each ideal main path separately against the original
3576/3588-cell denominator, with zero predicted gaps. No boundary/repair physics
or actual tracking is represented. Because the main path is longer, reduced
repair demand must repay its extra cost; overall efficiency is unproven.

All23 screened plans are retained. Burger .25m was not selected: it has lower
ideal coverage and a longer path than the current .30m. Burger .20m failed the
original50mm static tracking-clearance screen. .27m was selected from17 Burger
spacing trials before any new physical run; no evaluation seed was inspected.

`n02-overlap-offline-scores.json` records all scores and exact input hashes.
Full generators, logs and plans remain in the local immutable archive
`evidence/2026-10-08/n02-overlap-offline-screen`. SDK prediction containers used
network=none and image
`sha256:c31355f34739eb4ea8b60de1414c57ce0eea8225aa9e3ce854b7ceb66ca4376e`;
no ROS node, simulator, actuator or action was invoked.

Validation:350 ROS tests passed,10 integration deselected,1 existing warning;
8 focused protocol tests and changed-file Ruff/format passed. Full-repository
Ruff has the same290 pre-existing findings as mergedmain21838614, with no added
finding; global format still reports609 pre-existing files. These global checks
are not claimed green. Next: finish and review both original five-pilot series,
then preregister new source/protocol/diagnostic seeds and run fresh pairs. Do not
replace historical trials, borrow a previous baseline or freeze evaluation on
offline predictions. Physical and Native acceptance remain NOT_RUN.
