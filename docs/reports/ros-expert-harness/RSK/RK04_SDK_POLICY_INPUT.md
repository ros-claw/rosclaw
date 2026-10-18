# SDK source-bound navigation policy input

The isolated SDK's existing fixture historically compiled every observed Body
with one known-fixture set of navigation limits. This cannot silently serve as
a new robot's policy. The Core compiler already accepts explicit source-bound
policies; this increment changes only the SDK operator adapter and its tests.
The frozen Core remains byte-identical to `7da96906` under `src/`.

An optional `settings.navigation_fixture_policy` object must contain exactly:

- `schema_version`: `rosclaw.sdk_navigation_policy_input.v1`.
- `scope`: `OPERATOR_OWNED_DISPOSABLE_SIM_NEW_BODY`.
- A bounded explicit `policy_id` and the current selected `body_id`.
- The actual `source_urdf_sha256` and current `navigation_requirements_hash`.
- `navigation_limits`: all eight existing explicit finite positive limits:
  `max_goal_distance_m`, `max_duration_sec`, `goal_position_tolerance_m`,
  `goal_yaw_tolerance_rad`, `max_speed_mps`, `max_yaw_rate_rps`,
  `stop_speed_mps`, `stop_yaw_rate_rps`.

The operator must derive and preregister these values from the new source and
its intended bounded SIM task. Old Body/workspace limits must not be copied as
new source evidence. A source/Body/requirements mismatch, omitted or extra key,
nonfinite/boolean/nonpositive value or stop threshold above its speed limit
fails before new Body compilation. Existing immutable Bodies are never edited.
The existing pure `SimulationNavigationBinding` validator and Body compiler
remain authoritative; the input cannot replace snapshot, witness, Source or
other policy fields. Policy selection is recorded in the SDK evidence with
`operator_input_present`, actual source pins and all selected limits.

If no input is provided, existing known-fixture defaults remain byte-semantic
unchanged for compatibility. New-Body acceptance must independently require the
explicit input flag and exact preregistered source/limit hashes, rather than
assuming that a missing settings key selected a new policy. The selection record
is not L1, a Permit, driver authentication or motion approval.

A generic `navigation_onboarding` source-only capture role now has the same
strict unadmitted, zero-action behavior as the legacy
`j100_navigation_onboarding` alias. It allows observing an anonymous new source
without assigning a vendor/model label. It cannot claim repair rollback or L1
admission; the existing role's isolation, freshness and sink checks are retained.

All ROS processes remain inside the owned fixed SDK image. Actual After actions
still require independent Source/Body binding, fresh full L1, canonical MCP →
rosclawd receipts, stop/zero-contact/source continuity and unchanged 500 ms source /
250 ms proof guards. This input does not install a driver, expose a new Agent
operation or establish unfamiliar-Body navigation acceptance.
