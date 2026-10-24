# Supplemental input preparation result

Two new conditional single-TF decision inputs were collected under stationary
preparation v4 (seeds 101 and 202), each with healthy/fault/restored original
captures. All six captures were independently reconstructed from native facts,
Source reviews and original clocks. Both fault inputs compile into exactly one
stage-only false→true TF change with an exact rollback. This is operator/D
preflight, not a model proposal or a model physical success.

- Healthy/restored: all six Case constraints SATISFIED, actual Source MATCHED.
- Fault: TF VIOLATED; Scan, Lifecycle, Geometry and Route SATISFIED. Source stays
  UNKNOWN with the sole existing-binder dependency `ROS_RUNTIME:tf.map_to_odom`.
  Initial identity binding and the full original stream continuity are checked.
  No Source result is forged and no Source/Body/Lease/Permit/safety gate changes.
- V4: two stationary Worlds, zero control messages, zero goals, zero model calls.
- Earlier versions: four Worlds retained, eight canonical goal receipts (seven
  COMPLETED, one FAILED). Seed 202 failed its second task with unsafe planar
  localization covariance; physical stop was not verified by that failed receipt.
  V3's seed-101 input phases passed, but V3 overall remains failed. V4 input
  success does not establish motion stability or erase any earlier failure.

[All attempts, phases, identities and hashes](RESULTS.json).
[Protocol and version history](PROTOCOL.md). Original capture source:
`3538997483bcc1cd60c83fd383f548ea98a414aa`. These are historical sealed facts, not current L1.
The 2026-10-21 closeout and its entry NO-GO remain unchanged. A new model study
must freeze its own inputs/protocol after accepted code delivery. No full
14-case calibration or model-to-motion loop has been run by this preparation.

## Existing CLI replay of the exact new seed-101 fault input

Run from the checkout with its existing .venv. The input clock below is original,
not today's time. This command reads files and applies no configuration:

```bash
TF_INPUTS=docs/reports/ros-expert-harness/INPUT_PREPARATION_2026-10-23/demo-inputs
TF_REPLAY_HOME=$(mktemp -d)
ROSCLAW_HOME="$TF_REPLAY_HOME" PYTHONPATH=src:. .venv/bin/python -m rosclaw.entrypoint ros engineering-case \
  --snapshot "$TF_INPUTS/system.json" --urdf "$TF_INPUTS/robot.urdf" \
  --asset-originals "$TF_INPUTS/collision-asset-originals.json" \
  --requirements "$TF_INPUTS/requirements.json" --source-review "$TF_INPUTS/source-review.json" \
  --reviewed-at '2026-10-24T00:06:48.510816+00:00' --plan-staging \
  --staging-nav2 "$TF_INPUTS/nav2.yaml" --staging-bringup "$TF_INPUTS/bringup.json" --json
```

Expected: `ok=true`, authority NONE, L1 NOT_EVALUATED, exactly one conditional TF
change with rollback, and only F-SOURCE-001 remaining UNKNOWN. The fault YAML
was frozen before the World and was applied as a runtime override **after** the
healthy stack was ready; it was not the cold-start configuration. Whole local
raw evidence and both seed input sets are delivered separately with SHA256.
