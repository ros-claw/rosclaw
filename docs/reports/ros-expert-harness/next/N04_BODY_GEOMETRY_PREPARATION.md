# N04 generic collision geometry preparation

Status: OFFLINE_PREPARATION. No held-out robot has been chosen or inspected.
No L0–L4 physical acceptance, Body binding or cleaning capability is claimed.

`derive_collision_envelope(expanded_urdf_bytes, base_frame=...)` reads named
collision primitives and fixed XYZ/RPY joint transforms. The requested base
frame must exist and have only fixed ancestors. Boxes, cylinders and spheres
use containing corner boxes; movable child subtrees use containing 3D spheres,
including full rotational sweeps and bounded prismatic travel. This can
overestimate clearance substantially. It never produces an optimistic mesh
approximation or infers geometry from visual links.

All collision links must belong to a single valid tree. An unsupported mesh,
unresolved frame/articulation, missing collision geometry, unexpanded expression,
XML entity declaration, nonfinite value or arithmetic overflow makes the
numeric envelope UNKNOWN. Partial successfully resolved primitives cannot
promote a whole-body envelope. Source bytes and the requested base frame are
recorded; output is candidate evidence, not verified Body truth.

The output always grants no capabilities and infers no cleaning attachment.
This parser uses no vendor whitelist, known-robot profile, topic-name guess,
room geometry, default radius, ROS transport or runtime action.

Validation: 25 anonymous synthetic-URDF cases pass, including frame inversion,
rotated/off-center collisions, sampled articulated sweeps, nested fixed links,
bounded prismatic travel, unsupported meshes mixed with valid collisions,
disconnected cycles, incomplete sources, UTF-16 entity inputs and overflow.
The ROS connector suite passes 368 tests (10 deselected); Ruff/format pass.

`discover_body_candidate()` now consumes a sealed fresh graph and actual typed
message frame metadata, fresh TF chains and uniquely matching live URDF digest.
No default topic/frame names are used. The optional read-only ROS2 host probe
captures original message receive times and robot_description parameter hashes;
it does not export URDF payload in the periodic snapshot. Stale or ambiguous
streams/descriptions, incomplete TF and mismatched provenance remain UNKNOWN.
A PROPOSED candidate never authorizes navigation, grants a capability or binds
a cleaner. There is no drive-topic publication or writable robot RPC.

Nineteen additional synthetic tests cover namespace/sensor/frame renames,
expired signal/header/TF/URDF evidence, ambiguous sensor/description sources,
source/hash tampering and actual probe metadata extraction. The combined ROS
suite passes 387 tests (10 deselected), both modules pass mypy and Ruff/format.
These are offline tests, not held-out L0–L4 robot evidence.

`validate_sim_attachment()` requires a named explicit SIMULATED_CLEANING
declaration and a finite simple polygon. It derives a conservative inscribed
radius and records its declaration hash; drive observations cannot supply it.
`derive_allowed_region_grid()` supports bounded arbitrary simple allowed task
polygons and observed spawn positions, including a shifted L-shaped region.
It checks the task/map/spawn frames and observed map orientation, applies
clearance/connectivity only inside the declared region and produces a frozen
initial denominator plus legal centers with provenance hashes. Temporary
occupancy cannot redefine that initial denominator. Rotated/unknown map
orientations and offset-only cleaning attachments explicitly remain unsupported.
The original known-Body run.py denominator is unchanged. The observer adds actual
origin quaternion metadata for later generic preflight.

Sixteen attachment/region tests and the complete current ROS suite pass:
416 passed /10 deselected, with three-module mypy and changed-file Ruff passing.
These remain offline preflight results and grant no physical capability.

Remaining before the N04 feature freeze: verified fixture-policy binding,
generic simulator workspace generation and live region/Body integration.
Only after that freeze may the actual held-out
asset be chosen. Physical acceptance must then retain separate L0–L4 results,
including two namespace/sensor/frame/spawn perturbations without core changes.

## Evidence-bound SIM workspace compilation

Fresh Discovery and explicit simulator-operator policy now feed `propose_sim_fixture_binding()` and `compile_sim_body_workspace()`. The policy binds the exact snapshot/URDF/attachment; the proposal requires active fresh named localization/planner/controller/navigator/monitor nodes, a fresh advancing SIM clock, actual typed map-frame pose estimates, measured monitor source parameters and graph connections, and explicitly declared cleaner service/state. Event-driven velocity channels require fresh graph/publisher ownership, without requiring motion before a Body can be compiled. Wrong-source lidar or initial-pose commands cannot stand in for readiness evidence. The returned proposal grants no authority.

The compiler creates a new exclusive workspace through the existing BodyResolver/e-URDF compiler and reopens its exact effective hash; an existing workspace is never overwritten. It preserves actual frames/interface names and conservative geometry, records source/attachment/policy evidence inside hash-covered provider fields, and declares only explicitly scoped SIM navigation/cleaner capabilities. It invents no wheel actuator or REAL cleaner. Generic observation source names, typed latched maps, explicit Nav2 lifecycle role bindings and selected sensor interfaces now survive probe/resolver paths rather than reverting to fixed scan/map/node names.

Latest ROS contract suite:462 passed/10 integration deselected/1 existing deprecation warning; changed-module mypy and Ruff/format pass. Actual compiler/reopen tests exercise two anonymous renamed fixtures, reject expired/rejected policy before creating files, preserve existing workspace contents and bind changed attachment declarations to different Body hashes. These are synthetic configuration tests, not actual third-Body L0–L4 acceptance.

Still missing before feature freeze: complete generic simulator/Nav2 workspace generation, guarded daemon endpoint/frame/region integration and live verification. No actual holdout has been selected or downloaded. Read-only discovery snapshots do not supply action authority; all future physical requests must retain rosclawd Session/Lease/Receipt and independent witness/contact/stop gates. Exact localization/model/namespace linkage and loaded-simulator asset identity must be verified by the actual holdout run, not inferred from these fixtures.

Repository-required checks: compileall PASS; CI-scope `ruff check src tests` PASS; all8 changed Python files Ruff/format PASS; required mypy set121 source files PASS; Practice183 passed/9 skipped/1 existing warning in29.50s. Full `ruff check .` remains FAIL with290 existing errors in report/example artifacts; an exact path/code/line/column/message comparison against merged main21838614 finds zero additions or removals. Full format check under local Ruff0.16.1 reports609 pre-existing files; it is not described as a full-repository formatting pass. Those unrelated files and immutable historical evidence were not rewritten.

Initial region admission now requires a timezone-aware fresh independent Gazebo spawn capture, fresh receive age and finite nonnegative SIM time. Historical/future/untrusted spawn inputs cannot select an initial connected denominator; the exact spawn payload hash is retained alongside the map and region hashes. Latest full ROS suite469 passed/10 integration deselected/1 existing warning;23 attachment/region cases and module mypy pass. Existing known-Body run.py denominator is unchanged.

ROS-host relative topic parameters now retain raw name/node/GetParameters source/capture timestamp and expansion/validation results using the installed rclpy API. Binding and resolver require matching complete fresh records and actual Graph connections; they do not concatenate an assumed namespace or claim to apply node remapping. Actual remapping mismatches remain UNKNOWN. The pure helper was exercised on the pinned image's installed Jazzy API (relative/private names pass; invalid full names reject; no ROS Node/RPC/publish). Full ROS480 passed/10 integration deselected/1 existing warning; targeted source/raw/time/node/remapping/connection fault tests and three-module mypy/Ruff/format pass. This does not count as held-out ROS/Body physical evidence. API reference: https://github.com/ros2/rclpy/blob/jazzy/rclpy/rclpy/expand_topic_name.py .

## Configured namespace, task frame, spawn and whole-region executor contracts

The SIM daemon executor now accepts a complete explicit endpoint set and copies it into an immutable mapping: NavigateToPose/NavigateThroughPoses/complete-coverage actions and initial-pose/lease/cleaner services. Observer topic is explicitly configured on its separate read-only transport. Relative, malformed, duplicated, incomplete or unknown endpoint roles refuse; defaults preserve existing Waffle/Burger interfaces. Action arguments cannot select another endpoint. This accepts operator-owned configuration and does not prove Graph evidence or grant a capability. Actual generic workspace/runtime generation and source admission are still required.

Goal headers, recovery accounting, initial localization and canonical artifacts use the configured actual grid frame. Older callers omitting a frame retain the existing map default. An optional bounded frozen startup pose supplies x/y/yaw to the existing daemon-owned initial-localization service; it is used only while the independent body remains within the original20mm/rad spawn tolerances, brush OFF, lease inactive and stationary. The Agent cannot supply an action-time GT correction or another startup pose.

An optional explicit simple mission polygon supports nonrectangular regions without deriving a bounding-box goal. The fixed grid denominator must already lie inside that region; out-of-region denominator/repair centers refuse instead of being trimmed. Exact cyclic/reversed vertex order is accepted and canonicalized to the configured whole polygon; smaller/larger/self-crossing/foreign-frame goals refuse. The old rectangle path is unchanged. The rectangular boundary-pass helper is unavailable for arbitrary polygons until its own generalization is validated.

Validation at the complete-region checkpoint:522 ROS passed/10 integration deselected/1 existing warning in9.67s, required mypy121 files and scoped source/tests Ruff pass. Forty-one focused endpoint/spawn/region cases exercise actual executor methods with fake transports, synthetic shifted poses and L-shaped grids; no Gazebo process/action or held-out model is invoked. Initial regression exposed five legacy fixtures omitting frame_id; the executor now retains the original map default and the corrected complete suite passes. Actual namespace/shifted-spawn L4 and held-out L0–L3 remain NOT_RUN, generic_live_integration=false.

Final scoped verification after repair-center screening:524 ROS passed/10 integration deselected/1 existing warning in9.84s. Invalid out-of-region and oversized repair centers fail before any client is used. Source/tests Ruff, changed-file format and executor/endpoint mypy pass.

## Observed coverage interfaces and generic execution configuration assembly

Optional execution_interfaces fixture declaration now requires exact fresh typed NavigateToPose/NavigateThroughPoses/opennav complete-coverage actions, typed initial-pose/lease/cleaner services, matching compiled navigation/cleaner endpoints, a unique active coverage lifecycle observation and a fresh unique String observer topic. Unknown/missing/stale/duplicated/mismatched interfaces refuse compilation, rather than falling back to guessed defaults. If complete, the compiler seals the proposal into the same generic URDF-derived Body and retains exact coverage/initial-localization bindings. Default compilation without this explicit declaration remains navigation/cleaner only. Interface presence does not prove transport authority or loaded physics; independent source admission remains required.

prepare_sim_execution_config reopens the actual compiled Body and hashed manifest, verifies exact URDF/snapshot/policy/interface identities, validates the fresh actual bound map and independently captured Body/run spawn, and derives the permitted arbitrary task-region denominator and legal repair centers. It returns the generic executor's exact frame/endpoints/spawn/whole polygon in READY_FOR_SIM_DAEMON_SOURCE_ADMISSION, with source-map/spawn/region/attachment hashes. It writes no state, registers no executor and dispatches no action. No model-specific profile or wheel provider is invented. Actual Gazebo/Nav2 generation/launch, independent source admission and held-out physical gates are still incomplete.

Validation:555 ROS passed/10 integration deselected/1 existing warning in9.41s;17 interface-source fault cases and12 complete configuration/reopen/shifted-region cases, three module mypy, source/tests Ruff and changed-file format pass. The complete fixture exercises actual Body compiler/resolver and actual executor goal validation with synthetic sources only. Two initial configuration tests exposed a wrong EffectiveBody metadata attribute assumption; the implementation now reads the compiler-retained provider evidence and coverage-verifier binding, and the corrected cases/full suite pass. No actual third robot selected; L0–L4 NOT_RUN.

Repository-required checks at this interface/config checkpoint: compileall PASS, required mypy121 files PASS, scoped new-module mypy3 files PASS, source/tests Ruff and five changed Python files format PASS; Practice183 passed/9 skipped/1 existing warning in29.16s. Generic live launch and physical held-out gates remain NOT_RUN.

## Generic Native fixture declarations from the reopened compiled Body

prepare_sim_native_fixture reopens the actual compiled generic Body/manifest, captured and compiled URDF bytes, observed interface proposal and fixed allowed-region/grid/brush hashes. It returns the actual Body identity/base/map frames, complete arbitrary mission polygon and exact configured interfaces; it reads no known-model profile and grants no source admission, action or REAL capability. Generic Native configuration is explicit via --generic-proposal, with bounded mission timeout, source SIM-only declarations and fresh exclusive body/execution outputs. Existing known-fixture configuration remains compatible.

The catalog follows the actual map frame and fresh generic/dynamic mission identity, refuses contradictory identities, and Practice uses that same mission. Before creating Runtime, the daemon refuses generic proposals without matching independent source-admission identities/initial packet/geometry evidence or without a temporal-source-capable executor. It forwards temporal/brush source options only when supported by the installed executor. Physical source admission and actual plugin/world integration remain incomplete; these declarations cannot satisfy held-out acceptance by themselves.

Twenty-two new cases use the actual generic compiler/resolver on synthetic namespaced/shifted/L-shaped sources, including source/geometry/endpoint/frame/brush/region substitutions, REAL/executed flags, exclusive output preservation, profile-fallback prohibition and pre-Runtime refusal. Initial verification caught a test assumption that a polygon was already explicitly closed; the actual six-vertex region is preserved and the executor closes its goal. Corrected tests pass. Full ROS578 passed,10 integration deselected,one existing warning in11.52s;module mypy PASS, repository-required mypy121 files PASS,compileall PASS,Practice183 passed/9 skipped/1 warning in36.34s;scoped Ruff/format PASS. No actual held-out asset selected or inspected;L0–L4 NOT_RUN.

## Combined temporal and generic endpoint checkpoint

The current integration preserves the frozen repair source and combines the temporal source contracts with generic Body declarations. Dynamic waiting now dispatches to the explicitly observed, typed and unique hold service from the immutable endpoint configuration. Generic compilation refuses missing, untyped, ambiguous or undeclared hold services. Legacy six-role static configurations retain the established default; dynamic configured execution refuses that fallback. Receiver fault cases run against both default and namespaced observation topics.

Validation:116 focused contracts passed;975 ROS tests passed,10 integration cases deselected,one existing warning in15.78s;changed-module mypy4 files and repository-required mypy121 files passed. These tests exercise real compiler/executor paths with synthetic sources and transports. They do not prove a live namespaced robot, actual temporal mission or held-out acceptance. Generic live launch/source admission remains incomplete and L0–L4 remain NOT_RUN.
