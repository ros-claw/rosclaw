# Parameter Atomic12 v1 — source-only delivery

## Status and scope

The scenario, pure oracle and tests are authored source, **not executed or verified
runtime evidence**. This authoring purpose permits only source reads/edits. SDK
imports, test execution, process dispatch, container access and observation are
not performed here. Root must separately review the sources, run the pure tests,
review its independent inert observer, and authorize any actual runtime launch.
A future runtime requires its own exact process dispatch authority.

Historical `NO_GO` proposal labels in the public contract describe earlier
coordination, not this new source-author purpose. The original AddTwoInts scope
remains **BLOCKED** because the fixed image lacks its package/generated binding.
This is a new built-in parameter-services scenario, not an AddTwoInts replacement
or proof of concurrent numeric service capability. No install, image pull or host
fallback. No physical, CAP80, NAV/VLA, video, or full-capability/equivalence credit.

## Entry and ownership

`run_parameter_atomic_scenario(case_id, nonce, domain_id, evidence_path)` has exactly
four required positional arguments and returns the same data object as FINAL,
only after writing, flushing, fsync and closing the evidence file. Importing the
module uses only the standard library; it does not create a Context. SDK imports
are confined to `_runtime`. The CLI is under the `__main__` guard with exactly:

```
--case CASE --nonce NONCE --domain-id INTEGER --evidence PATH
```

This is an interface description, not an instruction to run now. Case and nonce
are ASCII `[A-Za-z0-9_-]{1,128}` identifiers; domain is a non-bool integer 0..232.
Evidence is exclusively created (existing files are not overwritten). Exit 0
requires six verified PASS cases, truthful cleanup and a durable FINAL. Failed
imports, startup, futures, oracle checks and cleanup cannot become PASS. Invalid
arguments or an evidence I/O failure may propagate without a complete FINAL and
must be treated as failure by the external supervisor.

Runtime owns exactly one explicit Context, two Nodes and one bounded
SingleThreadedExecutor. Both nodes use `context=`, `cli_args=[]`,
`use_global_arguments=False`, `enable_rosout=False`, `enable_logger_service=False`,
`automatically_declare_parameters_from_overrides=False`, and explicit BOOL False
overrides for both `start_type_description_service` and `use_sim_time`. The client
has `start_parameter_services=False`, the server True. No global context is used.

The unchanged built-in server creates six ParameterServices. Three clients serve
healthy Get, healthy atomic Set, and a fresh never-created Get name. Two automatic
ParameterEvent publishers exist even with rosout disabled. Inventory is **11
logical ROS entities: 2 publishers, 0 subscriptions, 6 services, 3 clients**; it is
not a DDS endpoint/discovery count. Names are read from actual resolved entity
getters and retain duplicates (both publishers may be `/parameter_events`). No
foreign entities are filtered from these owned-node lists. This is not a claim
that an entire ROS domain contains no foreign participants. TypeDescriptionService
has its own handle outside Node.services: six services alone does not prove its
disablement; the explicit override is also subject to source review.

Server declares INTEGER target=10, limit=20, robot_id=42 individually, static typing
for all and read_only only for robot_id, before registering the custom callback.
The callback reads a snapshot from SDK parameters, computes the whole proposed
candidate, checks `0 <= target <= limit <= 100`, emits its decision and returns
SetParametersResult without mutating state or synchronously waiting. TimeSource's
internal callback is separate; these logs count only the newly added callback.
Descriptor read-only/type gates run before the custom callback. P4/P5 therefore
must not fabricate custom-callback events. SDK service behavior is not patched.

## Six cases, twelve real requests

| Case | Actual request sequence and required result |
|---|---|
| P1 | Submit both Get requests before waiting: `[target,limit,robot_id]` and `[robot_id,limit,target]`; two distinct real futures, ordered INTEGER replies `[10,20,42]` and `[42,20,10]`. Either reply observation order is accepted. |
| P2 | Atomic INTEGER target=12/limit=30 succeeds; Get shows 12/30/42; custom callback sees paired updates. |
| P3 | Atomic target=50/limit=30 fails semantic validation with nonempty reason; Get shows full rollback to 12/30/42. |
| P4 | Atomic read-only robot_id=43 plus otherwise valid target=13 fails before the custom callback; Get shows 12/30/42. |
| P5 | Atomic STRING target="12" plus INTEGER limit=31 fails descriptor typing before the custom callback; Get shows all INTEGER 12/30/42, no coercion or partial commit. |
| P6 | Fresh missing client's bounded two-second wait returns False with zero requests; healthy atomic target=13/limit=31 then succeeds and Get shows 13/31/42. This does not prove perpetual absence. |

Exactly seven Get and five atomic Set calls, maximum two outstanding futures.
All runtime state/rollback assertions use actual Get replies, never a cached
local dictionary. Expected oracle values are not substituted for SDK responses.
Future identifiers are `str(id(actual_future))`, retained locally to prevent ID
reuse, never represented as wire sequence numbers/GIDs. Polling records observed
resolution order; it does not claim wire arrival chronology when multiple futures
are already done. There is no ordering requirement between P1 completions.

## Evidence schema

UTF-8 finite unique-key JSONL, at most 256 rows and 8 MiB. Common exact keys:
`schema, case, nonce, domain, seq, event, data`. Schema is
`ROSClaw.ParameterAtomic12.Event.v1`; seq is a non-bool contiguous integer starting
at zero. Case/nonce/domain are constant throughout. Exact event data keys:

| Event | Data keys |
|---|---|
| BOOT | scope (exactly `PARAMETER_ATOMIC12_V1`) |
| ENTITY_INVENTORY | publishers, subscriptions, services, clients, names |
| REQUEST_SUBMITTED | request_id, case_id, service, future_id, request |
| RESPONSE_RESOLVED | request_id, case_id, service, future_id, response |
| PARAMETER_VALIDATION | case_id, proposed, current_before, candidate, successful, reason |
| ABSENCE_PROBE | case_id, service, timeout_s, available, requests_sent |
| CASE_RESULT | case_id, status, reason |
| FINAL | status, condition_statuses, counters, cleanup, errors |

Service enum: `get_parameters`, `set_parameters_atomically`. Request IDs:
`P1_GET_FORWARD`, `P1_GET_REVERSE`, then `P2_SET/P2_GET` through `P6_SET/P6_GET`.
Case IDs are P1..P6; status enum PASS/FAIL/UNKNOWN.

Get request: `{names: list[str]}`; Get reply: `{values: list[{type:int,value:int_or_str}]}`
in request order. Atomic request: `{parameters: list[{name:str,type:int,value:int_or_str}]}`
in original order; atomic reply: `{successful:bool,reason:str}`. INTEGER code=2,
STRING code=4; bool is never an integer. Rejection reason must be nonempty; no
fixed SDK reason text is required, maximum 2048 characters. Callback proposed is
the named list; current_before/candidate are mappings from each of the three
parameter names to `{type,value}`. These are snapshots, not a commit operation.
Inventory names maps each category to one resolved name per actual entity.

FINAL condition_statuses has exactly P1..P6. Counters have exactly:
`service_requests, get_requests, atomic_requests, missing_service_requests,
parameter_event_publish_calls, parameter_event_publish_observation`.
Complete counts are 12/7/5/0. Partial failures retain actual submitted counts.
Publish calls are **null**, observation **UNKNOWN_NOT_INSTRUMENTED**. Expected
nine automatic publish calls is NOT a measurement. No passive forwarding observer,
additional publisher/subscriber, or monkeypatch is authorized or implemented here.

Cleanup has exactly `executor_shutdown:bool, nodes_destroyed:int,
context_shutdown:bool`; successful values are True/2/True, marked only after actual
calls complete. Errors is a list of bounded `{class:str,message:str}` objects.
Completed case rows are retained; failed active case is FAIL and unattempted cases
remain UNKNOWN. FINAL is last and emitted only after cleanup attempts. A full
oracle failure downgrades FINAL, never overwrites old evidence as success.

## Bounds, testing and limitations

Each known service startup wait is at most five seconds. Each future has a
five-second observation deadline; spin_once slices are at most 50 ms. Missing
wait is exactly a bounded two-second API timeout, zero RPC. Absolute monotonic
work budget is 120 seconds, whole budget 180, with at most 60 seconds for executor
shutdown and remaining cleanup. Every cleanup action is attempted independently.
Python cannot forcibly interrupt blocking SDK construction/destruction, imports
or filesystem I/O: Root's external hard deadline and normal-exit/container-removal
checks remain mandatory. Durable in-process FINAL does not itself prove host
normal exit/removal. SDK/type-support availability is not execution proof.

Source-owned `candidate_validation`, `validate_case`, `validate_transcript`, and
`read_jsonl` are standard-library-only. Full transcript validation requires all
case evidence, counts, identities, types and cleanup; partial validation returns
False and never certifies PASS. Tests independently build explicitly synthetic
fixtures, exercise both P1 completion orders, paired semantic decisions and
nonmutation, rollback/type/bool/order/future/duplicate/missing/extra/callback/
cleanup/partial-record adversaries. No runtime function is called in those tests.
Pure oracle success would validate the oracle against synthetic data, not prove
ROS service execution. **No tests have been run during authoring.**

## Provenance

Protocol source: `inputs/PUBLIC_PARAMETER_ATOMIC12_CONTRACT.json`, SHA256
`9297066e62348908dab0b503d617d2a657fe17911d1b64dca56b8dcfcf5bebef`.
API excerpt: `inputs/ACTUAL_IMAGE_API_EXCERPTS.txt`, SHA256
`4ef82a73ee751f58e971763a574d6d765054b67ce6afbeea0520c9420760409b`.
It preserves per-primary-file hashes/line numbers, notably node.py 122–135,
212–246, 357–363, 648–665, 1093–1096; executor 270/308/898/922; Context
62/76–80/124; Client 112/172/214–217; generated Get/atomic request and response
fields, ParameterValue, ParameterDescriptor, and SetParametersResult fields.
The historical API book remains Root-only provenance and was not read here.
Target is fixed image `sha256:81b41b619cc44a3c4915cd7bf42a7faf5219ddf7d84e2506e890a2457adaaaf5`
with `/usr/bin/python3.12`, no interpreter substitution. Root independently binds
its future launcher/profile/identity and corrected observer qualification; old
import receipts are not current runtime evidence.

## Plain-Enum projection repair (source authoring only)

The supplied actual-image excerpt declares `Parameter.Type(Enum)`, not
`IntEnum`. Its official `get_parameter_value()` uses `self.type_.value`.
Integer wire constants do not make plain Enum members convertible with `int()`.
The supplied saved facts report an independently reproduced inert TypeError at
both `sdk_parameter` and `validate_update.before_snapshot`; this is not a new
observation or an SDK run during this authoring session.

`parameter_type_code` now reads the official `.value` and requires an exact
non-bool integer without coercion. Missing `.value` raises AttributeError;
noninteger or bool values raise ValueError. It projects type codes, not parameter
payload values, and leaves supported-code/value policy to the existing checks.
Only the two `int(p.type_)` expressions use this helper. No other existing module
AST, runtime behavior, request plan, oracle, callback policy, cleanup, imports,
contract, schema, version, counts or parameter mathematics is intentionally
changed. The separate ParameterValue wire-field conversion is unchanged.

One new pure unittest class adds three regression methods with genuine plain
Enum INTEGER/STRING members: named and snapshot projections, the old conversion's
TypeError, and malformed/noninteger/bool rejection. The original sibling loader,
17 test methods and their assertions are preserved. These are authored tests,
not newly executed results.

Keep the historical CLOSED1 workflow failure, the saved original 17-method pure
PASS (not SDK/software acceptance), and the independent two-site inert defect
reproduction as distinct evidence. This repair does not supersede any of them.
No commands, tests, SDK, ROS, subprocesses or physics execution were performed
for this source-only repair; regression verification and software acceptance
remain pending independent review and authorized testing. No CAP80 or physical
credit is claimed. JSON reader exponent overflow remains a separate nonblocking
backlog item and is not repaired here.
