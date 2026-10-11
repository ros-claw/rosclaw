# Navigation-only SIM execution contract

The existing daemon-owned ROS simulation executor now has an explicit
navigation-only mode. It uses the same Runtime, action gateway and daemon
Session/Lease/Receipt boundary. It creates no REAL executor or MCP tool.
Agent actions still enter through MCP `request_action`; configuration and
executor construction belong to the isolated simulator/operator daemon.

The previous navigation path shared coverage configuration and required a
cleaning-disable acknowledgement even after successful navigation. The new
mode requires no coverage grid, repair centers, cleaning polygon, cleaning
state or cleaning service. The original coverage mode remains the default.

## Binding and configuration

`SimulationNavigationBinding.from_body()` checks an already compiled
`EffectiveBody` hash and copies only immutable scalar configuration. The Body
must explicitly prohibit REAL execution and declare navigation as enabled or
SIM-degraded, with no blocked/disabled navigation. SIM-degraded capability
state is preserved, not promoted. All existing gateway policy still applies.

The Body's `provider_interfaces.ros_capability_bindings` must explicitly contain:

- `navigation.navigate_to_pose`: fully qualified `name` and
  `action_type: nav2_msgs/action/NavigateToPose`.
- `safety.simulation_lease`: fully qualified `name` and
  `srv_type: std_srvs/srv/SetBool`.
- `safety.simulation_observation`: fully qualified `name`,
  `msg_type: std_msgs/msg/String`, `evidence_domain: GAZEBO_PHYSICS`, and a
  nonempty fixture `run_id`.

`frames.map` and all eight `safety.navigation_simulation` limits are required:
`max_goal_distance_m`, `max_duration_sec`, `goal_position_tolerance_m`,
`goal_yaw_tolerance_rad`, `max_speed_mps`, `max_yaw_rate_rps`,
`stop_speed_mps`, `stop_yaw_rate_rps`. Limits must be finite and positive;
stop thresholds cannot exceed speed limits. No distro, model or interface
name is inferred. These limits check the SIM execution and observed trace;
they do not configure or prove controller clamping or obstacle response.

The acceptance daemon's operator `execution_config.json` selects
`executor_kind: navigation_only`, `body_id` and the exact
`body_snapshot_hash`. It loads an existing compiled Body before Runtime
initialization or ROS connection, refuses mismatched hashes or coverage
configuration, and registers only navigation with the existing gateway.
It does not auto-compile a missing Body. The coverage memory executor is not
registered for this mode. This is fixture infrastructure, not an automatic
discovery-to-authority promotion.

## Execution and independent verification

Only a finite planar `pose` in the bound map frame is accepted. Its quaternion
must be normalized, and target distance from the independent starting pose
must fit the bound small-motion distance. Action arguments cannot select
endpoints. Coverage, initial-pose writes, REAL/SHADOW, stale/mismatched Body
hashes and a stopped executor are blocked before dispatch.

The independent observation stream retains the existing fresh wall timestamp,
finite x/y/yaw/SIM-time, completeness and collision checks. Navigation mode
also requires matching `body_id`, `body_snapshot_hash`, `run_id`, `frame_id`,
boolean `lease_enabled`, and finite `linear_speed_mps`/`angular_speed_rps`.
Wrong identity, collision, incomplete data and observed excessive speed remain
latched during an action even if a later good sample arrives. Absence of a
cleaning field is valid only in this mode; the default coverage witness keeps
its cleaning requirement.

A successful Nav2 result alone does not complete navigation. Position and
yaw must independently match their tolerances; all consumed samples must
satisfy the bound witness contract. The daemon then revokes the SIM lease,
requires a positive acknowledgement, and observes post-request stopped
samples for at least 0.1 advancing SIM seconds, within a bounded two-second
wait and the action's duration budget. `physical_stop_verified` refers to
this SIM witness only, never hardware acceptance.

Lease enable/renewal and goal dispatch share a lock with emergency cancellation
and revocation. An in-flight enable finishes before the final revoke, and a
stop flag prevents subsequent enable/dispatch. Cancellation transport failure
does not skip lease revocation. RPC acknowledgement, terminal action state
and independent physical-stop observation remain separate facts. Navigation
mode retains its stop flag across later requests; it does not automatically
re-arm itself after a stop or lease fault.

## Acceptance boundary

The existing isolated acceptance stack supports `stack.py --navigation-only
--controller-watchdog --profile waffle --seed <registered-seed>`. This operator
fixture compiles the pinned vendor URDF into a SIM-only navigation Body before
starting physics and binds its URDF digest and fresh run ID. It starts no
coverage server/lifecycle manager and exposes no cleaning service/state
publisher. Nav2 localization is initialized by the fixed-world operator setup;
the Agent does not submit an initial-pose write. The existing leased velocity
sink and bottom-level controller watchdog still apply.

`navigation_run.py --directory <fresh-evidence> --endpoint <loopback-rosbridge>`
submits a 0.35 m target through the real MCP wrapper and a separate daemon.
It requires a canonical COMPLETED receipt. The observer derives linear and
angular speeds from successive Gazebo ground-truth poses, independently of
Nav2 commands, localization and action results. These known Waffle/Burger
fixture templates are not a generic discovery adapter or held-out Body.
The fixed-world ground-truth frame convention must not be reused as proof of
arbitrary map alignment.

Unit and subprocess contracts use synthetic observations and no World, ROS
publisher, motion, or hardware. Their executor results do not prove Gazebo
physics despite exercising the production witness interface. Independent
daemon/MCP and actual Nav2/Gazebo execution must be accepted separately.
No unseen Body asset has been selected and no physical L1/L2 claim is made.
Discovery candidates do not grant this binding: operator-reviewed Body
generation, model/geometry/footprint consistency, source admission and
independent guarded navigation remain required before held-out acceptance.
