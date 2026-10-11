# Source-bound navigation-only SIM Body compilation

`ros compile-navigation-body` reuses the earlier N04 `sim_workspace.py`
URDF/profile/Body/lock/reopen flow without its mandatory cleaning attachment.
It uses the existing Body compiler and navigation observation review, rather
than selecting another vendor profile, registering an executor or creating a
Runtime. It is an operator fixture compiler, not an Agent motion entry.

The input is a sealed ROS snapshot, its exact expanded URDF, explicit navigation
requirements, and a source-bound SIM fixture policy. The snapshot must still
pass the existing freshness, geometry, TF, sensors, lifecycle, localization,
costmap and command-chain review. Historical captures remain historical; the
CLI offers no override of the review time. Unknown evidence is refused before
creating the destination.

```sh
ROSCLAW_HOME=/tmp/navigation-operator-home PYTHONPATH=src \
  .venv/bin/python -m rosclaw.entrypoint ros compile-navigation-body \
  --snapshot fresh-snapshot.json --urdf expanded.urdf \
  --requirements navigation-requirements.json --policy fixture-policy.json \
  --workspace /tmp/new-navigation-workspace --json
```

The destination's parent must exist and the destination must not exist. There
is no `--output` overwrite option for this command. URDF, requirements and policy
must be bounded regular files; their symlinks are refused. The required snapshot
is replayed without opening a ROS connection. This example writes temporary
operator fixture artifacts, not a live robot workspace.

The policy contains exactly:

- `schema_version: rosclaw.navigation_sim_fixture_policy.v1`,
  `source: simulator_operator_fixture_policy`, `approved: true`, nonempty
  `policy_id` and `body_id`.
- `source_snapshot_hash`, `source_urdf_sha256`, and
  `navigation_requirements_hash`, matching the current navigation review.
- `navigation_limits`: the eight explicit limits documented in
  [navigation-only execution](NAVIGATION_ONLY_SIM_EXECUTION.md).
- `witness_declaration`: exactly `observation_topic`, `lease_service` and
  `run_id`. These are explicit independent SIM fixture declarations, not
  inferred ROS endpoints or proof that the observer exists.

The fixture policy's `approved` field declares an operator input. It is not a
daemon-issued Permit, verified publisher provenance, REAL authorization or a
claim of physical acceptance. The compiler validates the resulting immutable
SIM binding but preserves navigation's compiled capability status. It infers no
actuator provider and adds no cleaning, coverage or initial-pose write capability.

Outputs are the standard e-URDF, lock, Body, calibration and effective Body
files, the exact URDF, an `execution_config.json` selecting navigation-only,
and `navigation-body-workspace.json`. The manifest binds the source snapshot,
URDF, requirements, navigation review, fixture policy, compiled Body and output
file digests. Geometry and provenance are included in the immutable Body's
provider evidence. The Body is reopened through `BodyResolver` and checked with
the same `SimulationNavigationBinding` used by the daemon.

The result is `CONTRACT_COMPILED_NOT_ADMITTED`, with `binding_verified: false`,
`physical_acceptance_level: NOT_RUN`, no dispatched action and no REAL use.
Installing/admitting the source, verifying the independent observer and frame
alignment, and obtaining a guarded MCP/daemon SIM navigation receipt remain
separate gates. Synthetic renamed/size variants and CLI tests are compiler
contracts, not unseen-Body L1/L2. Freeze the complete generic adapter before
selecting a held-out asset. Existing known-model physics evidence from the
navigation executor does not promote this compiler to physical acceptance.
