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
process split, runtime route/wait integration, bounded
DEFERRED wait/revisit and actual Native D2/D4/D5/D6 episodes. Existing native
historical runs cannot substitute. The accounting adapter is not currently
wired into daemon dispatch. `v1_done=false`.

## Sealed collision projection preparation

The pure OccupancyProjector now reads exact bounded SDF bytes, binds their SHA256 and declared obstacle model set, and encloses all supported box/sphere/cylinder collision primitives (including link/collision offsets) in conservative model-centered discs. Visual-only, mesh, articulated/nested/included or ambiguous geometry fails closed. This can overestimate occupancy; it does not claim exact surface rasterization. One whole ground-truth packet must contain every declared model at the brush packet SIM timestamp in the frozen map frame. Missing/stale/duplicate/nonfinite observations and projection work over100000 cells return no partial snapshot. Both projector and accounting bind the full original grid/frame/brush geometry as well as the unchanged accessible denominator.

Latest validation:56 focused occupancy/projection tests;393 ROS contracts pass/10 integration deselected/1 existing pkg_resources deprecation warning. Two-module mypy and Ruff/format pass. These synthetic tests are not physical episodes. Producer transport, actual Gazebo source binding/scene identity, geometry capture from the actual loaded world, monotonic/capture-time contract, passive observer/actuator separation, daemon integration and fresh Native D2/D4/D5/D6 remain NOT_RUN. At this historical checkpoint the adapter was not wired; see the later daemon integration section below.

## Time-paired bounded recovery preparation

TimePairedRecovery separates ready/deferred/exhausted pending cells instead of deferring every cell in a mixed blocked component. It requires the latest successfully accounted occupancy sequence/time and explicit same-packet legal-route reachability; missing/unpaired evidence yields UNKNOWN with no dispatch proposal. Withdrawal restores candidates without cleaning credit, waiting consumes no attempt, canonical-attempt delivery remains idempotent, and the immutable bounded SIM deadline removes all motion proposals. The inherited unpaired propose entry is disabled. Existing MissedRegionRecovery and runtime defaults remain unchanged.

Latest full ROS contract result:407 passed/10 integration deselected/1 existing deprecation warning in8.43s;13 focused temporal recovery cases and mypy/Ruff/format pass. This is a state/proposal layer only: the actual legal-center route producer, bounded runtime wait with brush off and lease-safe stop, accounting integration, canonical blocked receipts and new Native D2/D4/D5 episodes remain unimplemented/unrun. Ordered P0 merge prerequisite is still pending.

## Causal brush-state pairing before process separation

BrushStateTimeline binds ordered simulator-actuator transition/watermark events to run/Body/attachment/producer and capture/hash/receive freshness. A pose waits until an exclusive watermark passes its SIM timestamp, allowing service transitions at that same tick to arrive first. Latest brush-on state is never applied to a not-yet-confirmed past pose; disable arrival and exact transition boundaries cannot produce false on-credit. Gaps, reorder, stale/future/naive captures, wrong sources/digests, changed watermark state without a transition, reused poses and unconsumed bounded history latch failure. State/watermark hashes and a pose-pair chain are retained.

23 focused synthetic brush timeline tests and the full ROS431 passed/10 integration deselected/1 existing warning; module mypy/Ruff/format pass. This is an accounting prerequisite only, with no actual actuator event publisher, passive observer process split, daemon wiring or new Native dynamic episode. It grants no action authority and claims no Gazebo brush sensor. The cleaner remains explicitly SIMULATED_CLEANING.

## Daemon accounting and versioned artifact replay

The daemon-configured optional frozen run/geometry binding now selects OccupancyAccounting before each pose projection in the actual repair loop. Action arguments cannot enable it or replace its identity. A received sample must contain the complete independent same-SIM-time occupancy payload and its digest; bad/missing data latches failure before credit. The final artifact retains one occupancy record per trajectory pose under the explicit rosclaw.time_paired_mission_evidence.v1 schema. Executor final verification and public mission verification share replay_coverage, which applies each historical mask and disables dynamic interpolation. Missing records, count/time/sequence/hash mismatch, record keys replacing pose fields, or an unversioned dynamic payload are rejected. Canonical receipt exact-byte binding remains required; a complete synthetic replay has NOT_VERIFIED status without a receipt. Existing static evidence schema/result/interpolation behavior remains compatible.

This is opt-in runtime plumbing, not completed dynamic motion handling: current recovery still uses the original bounded recovery and has no new wait/route integration. The acceptance daemon accepts an operator-created occupancy_binding configuration, but existing static drivers do not generate or enable it. Actual geometry producer/loaded-scene identity, causal brush publisher/observer split, temporal legal-route production and bounded brush-off wait, canonical blocked receipts and fresh Native D2/D4/D5/D6 remain NOT_RUN. P0 source used by the physical matrix is unchanged.

Latest validation:445 ROS passed/10 integration deselected/1 existing pkg_resources warning in8.43s;14 new transport/runtime/replay cases,94 focused regression cases before adding the last two strict-type/time cases. Changed-file Ruff/format and three-module mypy pass. Synthetic coordinates use the original verifier speed/gap limits, with no production tolerance changes. No fresh physical dynamic acceptance is claimed.

## Opt-in process separation and causal brush transport

The isolated SIM stack now accepts a prepared frozen brush binding and starts separate simulator-owned actuator and passive observer processes. Default static launch is unchanged. In split mode the observer creates only its evidence publisher, no drive/cleaner-state publisher and no cleaner/lease service; guarded methods reject actuator use. It subscribes to ordered actuator events, waits for an exclusive watermark before pairing a physics pose, retains source/pair hashes in observations and latches invalid/stale source failures. Startup poses predating the first complete OFF watermark never establish readiness or credit.

The actuator alone forwards Nav2 velocity under the existing1.5s daemon lease and existing bottom-controller watchdog. Cleaner enable requires a live lease, expiry/release publishes stop and OFF transition, event/publication failure stops drive and disables cleaning. Ordered transition/watermark events bind prepared run/Body/attachment/producer, with a separate append-only audit. Daemon read-only receiver optionally checks that exact frozen brush source plus strictly later watermark and matching ON/OFF before accepting a sample; a single bad frame during an action latches failure despite later valid frames. Acceptance daemon rejects a prepared brush Body hash differing from its actual immutable Body.

Validation:455 ROS passed/10 integration deselected/1 existing warning in8.70s, including10 split/receiver cases. Endpoint recorders execute actual fixture node constructors and callbacks without loading ROS: no actuator endpoints in passive mode, live-lease admission, expiry OFF/stop, failed transition stop, delayed closure/exact disable pairing and mismatched source rejection. This is process/transport contract evidence only. The launcher requires an operator-prepared binding; existing static paired driver does not enable it. No actual two-process Gazebo launch or dynamic Native episode has run, no distinct-UID/DDS access-control claim is made, and producer/loaded-scene occupancy capture plus temporal route/wait handling remain incomplete. At this historical checkpoint canonical brush-proof replay was still missing; the following section completes that contract. P0 frozen physical source remains3c0f17d2.

## Retained brush pair proof in canonical coverage artifacts

BrushStateTimeline now returns the preceding pair chain and exact paired pose SIM time. A shared validate_brush_pair guard checks frozen source identity, matching observed cleaning flag, strictly later exclusive watermark, pair digest and consecutive pair-chain continuity. The daemon receiver tracks the chain during live reception; a dropped/reordered pose cannot disappear under a later valid pair. Canonical coverage artifacts persist one full pair record and its frozen binding per trajectory pose. Public mission replay checks length, identities, time/digest/chain and rejects any record replacing trajectory fields before granting credit. Source event hashes are retained opaque references to the separately kept actuator/observer ordered audits; this consistency calculation does not authenticate a DDS sender or replace canonical receipt/source trust.

Latest validation:462 ROS passed/10 integration deselected/1 existing warning in9.03s;7 additional retained-artifact tests cover exact replay and missing/hash/chain/Body/time/field-override failures. Three-module mypy and changed-file Ruff/format pass. Complete synthetic proof without a canonical receipt remains NOT_VERIFIED. Actual scene geometry producer, runtime temporal route/wait and fresh dynamic Native physical gates remain NOT_RUN.

## Actual-component packet parser and passive callback integration

The Gazebo8 PostUpdate plugin has now been built against the unchanged image SDK and tested on isolated actual SDK component fixtures. The core parser verifies whole-packet source/run/Body/attachment/world identities, exact closed model set/entity uniqueness, physical clock/sequence/capture freshness and normalized poses. It independently recomputes conservative envelopes from actual box/sphere/cylinder dimensions and model-relative offsets; a claimed radius, mesh, duplicate entity or incomplete source is never accepted as free space. Geometry binds actual model identities/primitive components; moving an obstacle changes occupancy without changing geometry identity.

Opt-in dynamic passive witness subscribes to this single physics packet instead of asynchronously caching robot/obstacle poses. It freezes the first complete geometry identity, verifies the actual map/frame/grid against the prepared fixed denominator, waits on ordered brush watermarks using a bounded six-packet queue and applies the corresponding occupancy before emitting each CleaningPose. Producer gaps/reorder, geometry/identity changes, stale/pause/overflow latch incomplete evidence. Startup data before the first OFF watermark never produces readiness/credit; incomplete bootstrap remains recorded. The supported frame transform is explicitly approved known-fixture world/map identity, not an inferred generic third-Body transform. A source admission file is not a mission-success receipt.

Validation:492 ROS passed/10 integration deselected/1 existing warning in9.06s, packet parser mypy and changed Ruff/format pass;60 focused packet/projection cases plus14 actual fixture-node constructor/callback recorder cases include exact same-time occupancy/brush pairing, zero occupied credit and only actual revisit credit after withdrawal, and gap/changed geometry/stale failure. The preserved nine C++ SDK-generated JSON records are synthetic component fixtures, not physics episodes. Actual plugin loading/stack bindings/world object preparation and runtime temporal route/brush-off wait remain incomplete; no new dynamic Native D2/D4/D5/D6 physical acceptance or third Body is claimed. P0 source/matrix unchanged.

## Fresh bootstrap, exact ROS wire resolution and preloaded obstacle movement

`dynamic_bootstrap.py` compiles the supplied known-fixture vendor URDF in a new exclusive home before Gazebo starts, generating new run/brush/closed-scene bindings and explicitly labelled fixture-policy grid. It writes neither measured_map nor physics_ready. After launch, run.py still recompiles the actual Body and strictly matches the measured map and independent component-source admission. Repeated bootstrap uses a new run; an existing output directory refuses reuse. This fixture compiler is deliberately restricted to Waffle/Burger and is not N04 generic adaptation.

ROS OccupancyGrid resolution is serialized as float32: the recorded physical static maps contain0.05000000074505806. Preparation previously accepted only exact0.05, making source preparation and strict measured-map admission incompatible. Preparation now accepts the exact policy value or exact wire value, recomputes the original fixed denominator with that selected resolution and retains it unchanged. Live observer and daemon still require exact equality; no floating tolerance or rounding is introduced. A near but different resolution remains rejected.

`prepared_obstacle.py` operates only inside a disposable fixture and calls its existing Gazebo world set_pose service for an already declared obstacle. It keeps model/collision identity, height/orientation and closed scene intact; it never publishes robot commands. Request screening requires the compiled Body binding, a complete contact-free independent body observation under0.3s and conservative clearance of Body radius + full3D box envelope +0.1m. Malformed/unknown names, stale/future observations, nonfinite/oversized geometry and unsafe placement refuse before service dispatch. The command reserves an exclusive append log before mutation. ACK explicitly requires independent PostUpdate pose confirmation and grants no cleaning credit or physical PASS.

Validation:579 ROS passed/10 integration deselected/1 existing warning in13.26s at this checkpoint, plus scoped Ruff/format. Bootstrap contract tests use explicitly synthetic URDF/ELF fixtures. A separate offline bootstrap using the actually recorded Waffle vendor URDF and compiled SDK plugin reproduced the actual static fixture Body hash510c836df556cde9d2e8b2fb25675aebec086234ec7177014724b0938d6aff3f. No Gazebo server or model movement was run for these checks. Actual dynamic Native D2/D4/D5/D6 remain NOT_RUN; P0 repair efficiency gate remains unmet.

## Bounded live audit cursor and retained temporal-credit diagnostics

The fixture audit cursor consumes an append-only audit from its genesis, validates run/sequence/hash continuity, waits on incomplete trailing rows, and latches completed malformed rows, truncation, replacement and bounded-backlog faults. It reports LIVE_PREFIX_NOT_FINAL_AUDIT; a live prefix never proves final closure or mission success.

`analyze_dynamic_credit` first uses public exact temporal replay, then independently recomputes new credit, occupied enabled-brush exposure and actual enabled free-cell revisits after obstacle withdrawal. Previously valid credit survives later occupation. No exposure cannot pass the D6 calculation. Synthetic tests caught a counterfactual footprint deduplication bug: each independent sampled footprint now clears visits, previous pose and last footprint. This diagnosis grants no canonical receipt, source authentication, Native task success or physical acceptance.

Validation: 595 ROS tests passed, 10 integration deselected, one existing warning in 12.32s. Fifteen focused cursor/credit cases passed; changed-file Ruff and format passed. Actual dynamic Native D2/D4/D5/D6 remain NOT_RUN. Failed pre-fix diagnostic regression logs are retained locally; they are not counted as passing runs.

## Fresh Native mission identity in catalog and Practice

Dynamic Native preparation now takes its tool-schema mission default and observed task-area mission identity from the admitted independent fixture source. Missing/malformed dynamic admission refuses catalog/daemon startup; legacy static fixtures retain their original mission name. Practice episode identity uses the same admitted fresh mission. The physical MCP tool body still refuses execution and requires Agentd/rosclawd.

Ten new cases include the actual installed FastMCP schema default, unchanged strict action parameters and refusal of direct invocation. Full ROS validation:605 passed,10 integration deselected,one existing warning in12.26s;changed-file Ruff/format passed. This corrects prelaunch wiring only; no new Native physical episode has run.

## Bound D2/D4 fixture scene controller and exact retained packet bytes

The passive physics audit retains both original packet UTF-8 bytes and their SHA256 alongside decoded fields, allowing later inspection of the actual source hash rather than a reserialized approximation. `dynamic_scenario.py` starts before the observer audit and consumes its genesis with the bounded cursor. Original packet hashes, decoded equality, source binding, packet sequence/time and immutable collision geometry are validated. Historical packet freshness uses its recorded receipt; movement separately requires a matched live sample and original PostUpdate packet under0.3s, identical body position, no contacts and the unchanged conservative placement screen.

Only preregistered D2/D4 policies and already declared obstacles are accepted. D2 introduction is relative to actual first enabled cleaning; dwell is10–30SIM seconds after independently confirmed placement. Withdrawal restores the frozen parked position. Both service ACKs require a subsequent actual PostUpdate position under a five-wall-second confirmation bound. D4 retains occupation until the original bounded scenario deadline or explicit task-runner stop. Neither controller completion nor ACK grants cleaning credit, task success or physical acceptance. The Native runner, canonical receipts and full credit/stop verification remain separately required; complete orchestration is not yet run.

Validation:621 ROS passed,10 integration deselected,one existing warning in12.41s. Sixteen new source/policy/controller cases include a synthetic successful two-move sequence and ACK-without-actual-motion rejection;all report NOT_VERIFIED. Sixteen actual node callback recorder cases retain original bytes and hash. Scoped Ruff/format passed. D1/D3/D5/D6 scene orchestration and fresh Native physical acceptance remain incomplete/NOT_RUN. No P0 source or active/queued trial was modified.

## Practice acceptance follows the fresh mission identity

The supplemental Native/static acceptance had a remaining hardcoded old Practice path. It now locates exactly one local bounded Practice record whose practice_id matches the canonical evidence mission, requires SUCCESS and an associated manifest with the same mission/session identity, and refuses foreign symlinks, oversized/non-object records, duplicate/missing or failed episodes. A prior successful static episode cannot satisfy a new dynamic mission.

Nine identity/manifest fault cases passed. Full ROS validation:630 passed,10 integration deselected,one existing warning in13.47s;scoped Ruff/format passed (the subsequent import-only sorting change was rechecked with the nine focused cases). The actual Practice episode/manifest records from all42 arms of the21 retained complete pairs passed read-only compatibility validation. That report and exact tested helper bytes are retained in local evidence under n03-practice-identity-compatibility-1118, and count as no new physical episodes. Actual new dynamic Native outcomes remain NOT_RUN.

## Terminal model/task counters are retained on Native failures

Native cleanup now captures public Core model usage counters and actual persisted TaskKernel states after stopping its owned Native process, for both successful and failed episodes. It opens the database read-only, closes the connection, bounds usage/task rows, exports no private authorization tables and never rewrites RUNNING/FAILED/CANCELLED into success. Missing/broken/oversized counter evidence remains incomplete and cannot bypass subsequent operator/daemon cleanup. SDK usage and failed canonical receipts remain separately retained.

Six tests use real isolated SQLite databases to check failed/cancelled/success/unclosed state preservation, unchanged database hashes, absent database refusal, bounded counters and no private authorization export. Full ROS validation:636 passed,10 integration deselected,one existing warning in14.10s;scoped Ruff/format passed. These are collector contracts, not fresh Native D4/D5 failure episodes.

## Closed raw-component replay against every canonical historical mask

`dynamic_source_replay.py` streams the closed observer audit with bounded complete rows, validates run/sequence/hash continuity and final writer count/hash/no-loss closure, and parses the exact retained PostUpdate bytes using their original receipt. The admitted fixed grid/frame/brush/denominator and collision geometry must match the canonical temporal artifact. For each canonical occupancy sequence it requires the same actual Body XY/yaw and SIM time, recomputes occupied cells from actual obstacle components, and requires exact snapshot payload/hash equality. Missing correspondence, substituted raw bytes, changed fixed grid, source gaps, pose/mask mismatch and incomplete closure refuse the calculation.

Thirteen synthetic SDK/source/artifact cases passed;full ROS649 passed,10 integration deselected,one existing warning in13.55s;scoped Ruff/format passed. Recorded observer processing age is explicitly reused as historical canonical evidence, not presented as a fresh live sample. Output remains NOT_VERIFIED and requires actual canonical/Native/brush/contact/stop gates separately. No new dynamic Native physics or authentication guarantee is claimed.

## Required dynamic scenario gates Native completion

The Native runner accepts an optional frozen required-scenario file. Before starting and throughout the task, its bytes must remain unchanged and its run/mission identity must match the actual source-admitted execution configuration and controller progress. Failed/stopped controllers abort Native acceptance. D2 positive completion additionally requires independently confirmed withdrawal and canonical temporal credit diagnostics showing actual enabled revisits of previously occupied unclean cells. A D4 perturbation cannot satisfy positive completion. Controller progress always remains NOT_VERIFIED; closed raw-component replay and actual canonical/Native/contact/stop gates remain separate.

Nineteen new cases check malformed/missing admission identities, bounded scenario input, foreign progress hashes, false PASS markers, failed/stopped controllers and partial trailing writes that must not conceal an already completed failure. Full ROS:668 passed,10 integration deselected,one existing warning in14.67s. Scoped Ruff and format passed. Fresh Native D1–D6 physical episodes remain NOT_RUN; these refusal contracts do not establish dynamic physical acceptance.

## Independent stop geometry for negative Native episodes

The read-only independent stop collector now requires a three-wall-second live collection with at least20 distinct actual Gazebo pose samples, advancing timezone-aware receipt and SIM times, at least2.5seconds of each span, finite geometry and the existing10mm/0.03rad standstill limits. Repeated/frozen/paused observations and Nav2 odometry cannot satisfy this calculation. Yaw differences use angular wrapping. Output remains NOT_VERIFIED and cannot replace brush-off/lease/contact/closed-source/canonical/task gates. The separate passive pose subscription supports an explicitly bounded60–1920second duration for full dynamic tasks; its historical90second default is preserved.

Eleven new calculation fault cases passed;full ROS679 passed,10 integration deselected,one existing warning in17.14s. Scoped Ruff/format passed. Actual D4/D5 negative task and stop evidence remains NOT_RUN.

## Opt-in actual Body collision source for future generic source admission

The passive SDK source now retains versioned v2 actual Body collision components when explicitly configured, preserving v1 defaults and obstacle accounting. The parser recomputes conservative actual world-XY bounds from primitive dimensions/local/world transforms, validates claimed envelopes and optional frozen maximum radius, and refuses missing/unsupported Body geometry or downgraded packets. Model/base identity and actual compiled-Body limit binding remain generic source-admission requirements;this calculation is no execution authority. Live observation and historical source replay consume the same optional constraint. Body geometry never becomes an obstacle mask or cleaning credit.

SDK compile/link/CTest/ldd-r PASS after retaining and correcting build-prefix/private-test-member failures. Thirteen SDK rows include the nine prior v1 cases plus four new v2 Body cases. Fifty-five focused parser cases and full ROS700 passed/10 integration deselected/one existing warning in15.34s;module mypy and scoped Ruff/format PASS. Actual v2 plugin load, generic Body acceptance and new Native D1–D6 remain NOT_RUN.

## Bounded actual Native negative-terminal collection

The Native fixture runner now supports explicit --expected-safe-failure D4/D5. D4 requires its frozen matching obstruction scenario; ordinary positive behavior remains unchanged. After a retained failed artifact, the runner stops answering further authorization cards and allows at most60wallseconds for the genuine failed response to reach TaskKernel. This does not extend the physical mission deadline. A negative mission-verification artifact or SUCCEEDED root refuses acceptance. Unclosed RUNNING/WAITING_INPUT/RECOVERING/CANCELLED roots remain pending and time out honestly. No task/transaction state is rewritten.

negative_native_progress reads the actual TaskKernel and ActionTxn tables in SQLite read-only mode. It requires exactly one Chinese-input SIM root matching the Body, a real FAILED/BLOCKED root, a matching failed coverage transaction, the public daemon canonical negative receipt and the exact retained failed artifact SHA256 inside the owned actions directory. Foreign missions/Body/action identities, REAL receipts, unfinished transactions, substituted bytes and escaping paths refuse or remain pending. The terminal record explicitly remains NOT_VERIFIED and requires separate actual perturbation, advancing independent stop, brush/lease/contact checks and closed component/source replay. D5 fault-controller evidence is a separate required outer gate; the CLI flag cannot supply it.

Twenty-two new cases use the repository migrations and actual ExecutionReceipt serialization with isolated SQLite data;47 focused Native/scenario/counter cases passed in3.69s. Full ROS747 passed/10 integration deselected/one existing warning in16.49s;compileall,scoped Ruff and changed-file format pass. Actual D4/D5 Native terminal propagation and physics remain NOT_RUN.

## Fresh D2 Native host entry and shutdown ordering

`dynamic_native_episode.py` prepares a new owned known-body D2 fixture and runs the existing Native→MCP→daemon path with one Chinese root task. Its closed protocol binds exact clean source, actual merged P0 PR #624, image ID, plugin/vendor bytes, seed, scene timings, mission deadline and actual SDK model identity. Preparation, scene control, independent passive pose observation, canonical/Practice/Memory verification and closed raw-component replay remain separate. No direct robot transport is opened by the launcher.

Preflight refuses an unmerged P0, changed source/image/plugin and another owned live physical episode. The private model profile is copied into a fresh home with mode0600 and is excluded from public evidence. Bootstrap failures retain a result. Cleanup always attempts the uniquely owned simulator stop, even after a lost docker-run response or an observer cleanup failure. Actual Native success is retained separately when a later source/shutdown gate fails. Closed source replay runs only after verified simulator shutdown; missing SDK identity or any incomplete gate refuses physical PASS.

Thirty-six host-entry contracts use no Docker, model or robot dispatch. They cover preflight refusal, startup failure, full ordering and retained bootstrap/run-response/SDK/source/child/world failures. The prior corrected full ROS checkpoint passed782/10 integration deselected/one existing warning in18.03s; final added lost-run-response case is rechecked separately. Fresh Native D2 and other D1–D6 physical results remain NOT_RUN until P0 formal evaluation and merge. These orchestration tests are not physical acceptance.

## Executor-to-closed-source Body identity regression

A full handoff review found that synthetic replay fixtures supplied a Body hash
which the actual temporal executor artifact did not yet retain. Temporal
artifacts now include the executor's frozen body_snapshot_hash; static artifact
format and coverage accounting are unchanged. A new regression runs the actual
executor writer, reads its retained artifact and replays all three original
component/mask correspondences through the closed-source auditor. No hash is
patched into the generated artifact and no acceptance rule is weakened.

Fourteen closed-source contracts passed, including the full writer handoff.
Full ROS784 passed/10 integration deselected/one existing warning in18.82s;
repository-required mypy121 files, scoped Ruff/format and compileall passed.
Transport/action feedback in this test is synthetic and no physics is dispatched.
Actual Native D1–D6 remain NOT_RUN pending P0 formal evaluation and merge.

## Canonical BLOCKED partial temporal artifact is a negative terminal

The negative Native collector now supports the real executor's safe BLOCKED
result with a retained partial temporal evidence artifact, as well as FAILED
with a diagnostic failed artifact. The BLOCKED branch is D4-only and requires
exact source-admitted run/mission/Body/action identity, canonical artifact SHA,
complete public temporal replay, coverage below98% and exact receipt/accounting
agreement. A COMPLETED receipt, foreign identity, changed bytes or an unsupported
case refuses. TaskKernel still must genuinely reach FAILED/BLOCKED. No state is
changed. A new owned terminal artifact stops further authorization answers while
the bounded existing60second propagation grace waits for the real task response.

Thirty-one negative-terminal contracts passed, including nine BLOCKED/artifact
substitution cases. Full ROS793 passed/10 integration deselected/one existing
warning in17.47s. The independent scene, brush/contact, closed actual source and
advancing stop checks remain required; no actual D4/D5 Native physics claim is
made by these isolated database/receipt tests.

### Whole known-SIM stack source preparation and fault retention

`backend_stack.py` joins the generated robot, instrument and world bundle,
original service worker, independent observer, required actor constraint,
Nav2 and independent witness as an owned operator fixture. Robot task commands
still require Native MCP and rosclawd. This launcher does not admit a held-out
robot and does not replace the frozen P0 stack.

The root World/binding used by witness and Native admission is exactly the
final source bundle; original World, binding and experiment files are preserved.
Runtime faults retain the original observation bytes and withdraw the observer
so the actor's source-age watchdog closes its gate. The World, actor and witness
remain alive until the immutable fixture deadline so a guarded stop and actual
independent pose measurement remain possible. A dead World is explicitly
`MISSING_STOP_PROOF`. Cleanup flushes owned sources before stopping the World.
The fixed duration includes source preparation and is never reset on dependency
startup. Lifecycle tests use plain Python processes, not World/robot evidence.

`backend_stack_source_contract.py` snapshots executed Python source and real ELF
bytes before validation, invokes the actual known vendor URDF/Body compiler,
checks final World model inventory and all root/bundle constraint bindings,
and uses the installed SDF parser. Both known bodies passed this source-only
check. No ROS Node, Gazebo World, scene service, Native task or robot actuation
was executed; physical acceptance remains `NOT_RUN`. Full N03 ROS regression:
1,239 passed, 10 integration tests deselected. The first whole-stack attempt
failed at import because the original P0 image lacks Python Body dependencies;
its failure log is retained. The frozen P1 dependency image passed.

The host's genuine Native/canonical acceptance still needs to be joined to
this qualified source path, including closed original service-wire replay and
independent stop proof. All six dynamic physical cases remain unaccepted.

### Qualified Native v2 host entry (D2/D4 code, physical tests pending)

`dynamic_native_episode.py` accepts the closed
`rosclaw.dynamic_native_episode.v2` protocol, including exact native contact
plugin and instrument-service ELF hashes and mandatory all-step/spatial/original
service-wire mode. Legacy v1 cannot silently acquire these inputs. Existing
actual PR #624 merge, clean commit, immutable image, model, Body and deadline
checks run before the operator fixture starts.

The v2 source path uses the owned backend stack. Both witness and the scene
controller read the exact final inventory, including the instrument. During the
genuine Chinese Native invocation, fresh qualified observation and mapped World
correspondence are mandatory. A fault requests `emergency_stop` through the
canonical stdio MCP server and rosclawd, and retains the original MCP response.
A successful RPC request is never itself stop proof. Independent actual pose
measurement is attempted while the World remains alive, including failed runs.
The no-daemon stdio transport check returned `DAEMON_UNAVAILABLE`; no robot,
World, ROS Node or SDK transport was present.

At successful source closure, an in-flight instrument RPC must finish its
measured cycle before the observer exits. World and witness remain alive. Closed
replay reopens original SDK service wire, all-step robot contacts and exact scene
joins, and requires the original observation interval to contain the entire
Native invocation. This source correspondence remains separate from canonical
TaskKernel, coverage/temporal credit, Practice/Memory and independent stop gates.
The complete N03 ROS source regression passed 1,259 tests (10 integration tests
deselected). D1/D3/D5/D6 full Native orchestration and all actual dynamic physical
runs remain pending; this checkpoint is not N03 acceptance.

## Qualified D3 two-blocker Native host source

The closed episode v3 protocol adds exactly a distinct second target, a second10–30SIM-second dwell and a2–30SIM-second gap to the qualified original-wire backend. Legacy v1 and v2 cannot silently select D3. Two separate blocker models are preloaded before World preparation under the same initial denominator; all geometry and identities stay frozen.

The owned scene controller introduces the first blocker only with independently measured Body clearance, records the original service response, and waits for an actual later PostUpdate packet at the target. After its dwell it requests withdrawal and independently confirms the original parked pose before waiting the frozen gap and introducing the second blocker. ACK alone cannot advance either stage. Native waits for both perturbations; all original observation/SDK/receipt/stop gates remain required, including the actual prior PR624 merge before any World or process launch.

After genuine task/source closure, `dynamic_two_blocker_replay.py` reuses the exact canonical component-to-occupancy replay and checks all four confirmations against their original observer packet bytes and original receipt ages. Per-model projections require a nonempty occupied grid at each introduction, an empty mask at each withdrawal, and an empty other-model mask throughout both confirmation windows. Full canonical replay rejects a stale mask carried into later trajectory samples. Dwell and gap are checked against actual recorded SIM times. This is historical source correspondence, not a new physics run or source authentication.

Validation: focused host/scenario/source contracts137 passed before the additional v3 early-merge refusal case; full ROS1293 passed/10 integration deselected/one existing warning in25.41s, source replay mypy and source/tests Ruff passed. Real source SDK preparation on both actual installed Waffle/Burger URDFs with two preloaded task blockers and the separate cache probe passed; generated Worlds passed installed SDF parsing and final/root physics inventories agreed. Executed source and real ELF bytes were captured before the SDK check. No Node, World, service or model call was started by it.

D2/D3/D4 host orchestration is implemented but actual new Native physics tasks remain NOT_RUN pending the frozen N02 formal review/merge. D1/D5/D6 complete actual task entry and acceptance remain incomplete. v1_done=false.

## D5 registered collection pause source foundation

`OwnedCollectionPause` accepts a closed operator SIM fixture policy pinned to the
same run, Body and constraint hashes. A matching request can pause only the exact
owned independent observer child; no PID is accepted from input. Its duration is
frozen to 1–10 wall seconds. The original policy/request hashes and observer PID
are retained before SIGSTOP. The World, actor and independent witness are never
signalled by this fixture. Resume runs even after a policy-change fault and during
cleanup before child shutdown. Resuming never clears the source fault latch or
asserts healthy data or physical standstill. Input files use bounded reads.

Validation uses real owned Linux Python sleeper processes, not a simulated robot.
The full ROS regression passed 1302 tests, with 10 integration tests deselected
and one existing warning, in 25.98s; scoped Ruff passed. One earlier collection
was interrupted before tests ran because its explicit marker overrode repository
exclusions; its original log is retained. The successful run uses repository
exclusions. No ROS Node, World, SDK transport or robot action was started by these
checks. D5 actual Native failure receipt and independently measured physical stop
are still NOT_RUN; this foundation is not D5 acceptance.

## D5 collector trigger and actual Linux signal confirmation

The source-only v4 protocol preregisters exactly the owned collector pause and
its bounded duration. It preserves the original backend hashes and mission
budget; the legacy complete episode entry still refuses v4 until the D5 host
failure/receipt/stop chain is wired. A preregistered operator declaration binds
to the actually prepared run/Body/constraint policy before process launch, with
its original bytes and hashes retained. No arbitrary process identity or new
robot authority can be supplied through that declaration.

`collection_pause_scenario.py` opens no transport. It requires fresh measured
Brush ON, a live lease, original Brush binding, complete same-time physics and a
healthy qualified backend before writing the one-shot owned fixture request.
Unknown sources, old timestamps, clock mismatch, contact, missing Brush source
or an existing fault reject the trigger. It retains the pre-fault observation
projection. That projection is derived evidence; complete original SDK audit
replay and source authentication remain separate. Request completion explicitly
does not claim the pause was applied or the robot stopped.

The stack separately records actual Linux process state T after SIGSTOP before
claiming collector pause application. World and witness remain untouched. Full
ROS regression:1321 passed/10 integration deselected/one existing warning in
26.65s; focused trigger/policy/process checks59 passed. These are source/process
checks, not an actual Native World run. Full D5 Native orchestration, negative
canonical receipts and independent actual stop measurement remain IN_PROGRESS.
Other D5 fault subclasses and D1/D6 physical entry remain pending.

## D5 Native collection-loss host and negative verifier

The qualified v4 complete episode entry now recognizes D5's registered collector
pause. It binds the operator declaration to prepared sources before launching
the World. Actual prior PR624 merge, frozen source/image/ELFs and vendor input
checks remain mandatory before any process. The scene controller observes fresh
Brush ON and writes the owned request. Linux T-state confirmation is separate.

On actual qualified-source loss after that registered pause, the host retains
original policy/request/application/pre-fault/current-source bytes and requests
`emergency_stop` through canonical stdio MCP. It leaves Native alive within its
unchanged budget so the genuine failed response can reach TaskKernel and produce
its negative canonical receipt. No task row is changed and no failed receipt is
created by the host. Unexpected failures before the pause retain the original
fail-closed path. A successful stop RPC still does not prove standstill.

The original closed audit/SDK replay now has an explicit healthy-prefix mode.
It checks the entire original writer chain and independently replays all original
robot/probe/scene/SDK wire through the exact retained pre-loss projection. The
suffix is UNKNOWN and cannot be consumed as full healthy-source acceptance. The
default positive replay still requires the full healthy closed sequence. D5
requires that prefix to occur during the actual Native invocation, at least one
SDK original-wire replay/cache cycle/scene join, and unchanged source hashes.

The negative verifier rereads actual TaskKernel SQLite and original canonical
failure artifact, checks unchanged pause originals and original stale interval,
rechecks SDK-prefix audit/summary hashes, and requires independent advancing
pose standstill plus complete closed actuator OFF/lease-release evidence. Post-
loss robot contacts remain UNKNOWN. This verdict is only the collector-pause
subcase; it cannot close the other clock/geometry/sequence/frame/contact cases.

Validation: full ROS1332 passed/10 integration deselected/one existing warning
in27.43s; prefix replay20 passed; complete host-source focused66 passed; original
loss/actuator/trigger focused29 passed. Tests use explicitly synthetic source
records, real SQLite migrations and actual owned Linux sleeper signals, with no
World, ROS transport or model call. New Native physics remains NOT_RUN until the
frozen N02 formal review and actual PR624 merge. D1/D6 complete entry and remaining
D5 fault subclasses remain pending. v1_done=false.

## D6 prospective nonvacuous credit gate (2026-10-09)

Qualified episode v5 explicitly binds D6 to the original all-step backend wire
and a pending-cell exposure policy. It uses the existing preloaded blocker,
unchanged placement clearance, bounded introduction/dwell/withdrawal and actual
post-update packet confirmations. It does not command the robot to contact an
obstacle or change the Body, brush, denominator, speed, leases or deadline.

After the genuine Native root and live canonical acceptance, the host still
requires closed original backend/source correspondence and independent stop.
Only then does exact temporal replay check that at least one previously unclean
cell was inside an enabled sampled brush footprint while occupied, received no
new credit while occupied, and was later actually revisited with the brush on
and the cell free. No overlap, brush-off exposure, prior valid credit alone,
missing revisit and a revisit to another cell all refuse D6. Synthetic positive
calculation remains NOT_VERIFIED and cannot confer a physical receipt.

The actual reviewed PR624 merge is still checked before any simulator process.
D6 physics is NOT_RUN; D1 and the remaining D5 subclasses are incomplete. The
v5 source is preparation, not closure of N03 or v1.

Validation: 113 focused tests; full ROS1393 passed/10 integration deselected
in32.25s; Practice183 passed/9 skipped in34.37s; required mypy paths plus the
changed diagnostics module122 files passed. All D6 data in these tests are
explicitly synthetic. The earlier focused run retained one expected regression
failure because D6 had previously been an unsupported-case refusal test.

## D1 prospective moving-blocker main-swath crossing (2026-10-09)

Qualified episode v6 binds an explicit straight crossing to the original backend
wire, source/image/Body/model and actual prior PR624 merge. The existing owned
scene controller moves only the preloaded blocker. It confirms every small
position update from a later original component packet; no robot action is
sent by the scenario. This is a discrete SIM moving-blocker surrogate, not
human walking dynamics or a hardware pedestrian safety claim.

The closed policy limits updates to1..5cm, separates requests by at least0.25SIM
seconds after the previous actual confirmation, bounds nominal traversal to
10..30SIM seconds and fixes a maximum30SIM-second traversal deadline. The
common dwell field must equal the nominal traversal duration in v6. Each step
retains the original placement guard and additionally screens the entire short
segment against the fresh body pose using the same original clearance margin.
Unavailable clearance waits within existing deadlines; no robot/brush geometry,
speed, denominator, lease or deadline is relaxed.

The controller follows the existing daemon diagnostic audit from genesis and
requires one admitted coverage action with an active MAIN_COVERAGE goal. It
requires a transverse interior intersection with an actual passive map-frame
LINE_LIST swath marker. Every crossing position confirmation must occur while
that main goal remains active and cleaning remains enabled. A main goal ending
before introduction fails immediately. The blocker is independently confirmed
back at its original parked position before scenario completion.

After Native/canonical acceptance and independent shutdown, closed replay
matches every step and withdrawal to original component bytes, the canonical
action ID, unchanged Body/geometry, the unique closed main goal interval, the
original swath event and a complete enabled canonical trajectory interval. A
service ACK, endpoints alone, a parallel or endpoint-touching path, wrong frame,
missing step, stale receipt, changed geometry, missing brush interval or motion
outside the main goal all refuse. Replay remains a source correspondence check;
it cannot replace backend qualification, canonical receipts or physical stop.

150 focused synthetic tests passed, including running the actual scenario loop
against original-format component callbacks. D1 physics is NOT_RUN; v1 is not
complete. All actual new dynamic tasks still require P0 review and merge.

Full ROS1436 passed/10 integration deselected in32.34s; Practice183 passed/9
skipped in32.97s; required mypy plus changed Core diagnostics122 files passed.

## Future fixture DDS preflight and precise startup failure records

The frozen formal Burger100828 baseline failed before action dispatch. Original
logs show coverage_server configuration began, its change_state response timed
out, and coverage bringup never completed. Original map, Nav2 startup and3374 of
3406 independent samples were complete. The underlying DDS cause is UNKNOWN.
No seed was rerun/replaced and the frozen source/image/driver remain unchanged.

The host ephemeral range is32768–60999. Existing default domains201/202 overlap
that range under the official DDS port formula. This is a confirmed configuration
risk, not proof of this timeout's cause. Future fixture source now defaults to81/
82 and checks a120-participant port reserve against the actual host proc range
before any transport. The result explicitly does not verify the future container
namespace or transport readiness. Startup timeout diagnostics preserve original
observations/log hashes and identify the absent coverage bringup marker; a log
marker is explicitly not a measurement of current lifecycle state. No readiness
or stop condition was weakened, no retry was added and no deadline was extended.

Validation: preflight/diagnostic17 passed; full ROS1349 passed/10 integration
deselected/one existing warning in28.97s. The live frozen v2 formal sequence is
untouched. Actual future DDS transport and startup recovery remain NOT_RUN.
Primary reference: [official Jazzy ROS_DOMAIN_ID documentation](https://raw.githubusercontent.com/ros2/ros2_documentation/jazzy/source/Concepts/Intermediate/About-Domain-ID.rst).
