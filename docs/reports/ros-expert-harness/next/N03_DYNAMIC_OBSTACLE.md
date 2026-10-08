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
