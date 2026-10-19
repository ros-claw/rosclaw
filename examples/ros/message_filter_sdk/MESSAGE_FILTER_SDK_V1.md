# Local message-filter SDK scenario V1

## Status and scope

Source authored only. No import, test, command, SDK execution, simulation or runtime verification was performed in this author purpose. The eight specified inputs were read; only the three designated root files were edited. Runtime needs separate review and fresh authorization. This document is not runtime permission.

The experiment uses finite generated PointStamped records, not sensor measurements. It does not demonstrate DDS, hardware, physical fusion accuracy, physics, perception, TF, navigation or broader system capability. There is no ROS initialization, Node, Context, executor, transport, subprocess or SDK fallback in the authored runtime. Module loading and metadata tests use only the standard library. SDK imports are lazy inside the four-argument public entry:

`run_local_message_filter_scenario(case_id, nonce, input_path, evidence_path)`

All four arguments are mandatory. The CLI has required `--case-id`, `--nonce`, `--input-path`, `--evidence-path` flags. A future separately approved command would invoke this source with those four flags using the independently approved ROS Python environment. No command was executed here. Exit 0 means the local runtime's checks passed; exit 1 means failure. Neither exit nor a self-authored PASS alone establishes independent provenance or full workflow success.

## Actual SDK route and lifecycle

The entry lazily imports installed message_filters, its simple_filter module, the generated geometry_msgs PointStamped module, builtin_interfaces Time module and rclpy.serialization. It checks the documented ATS constructor signature and SimpleFilter identity. Missing imports and unknown APIs fail with typed errors, not synthetic runtime success. Two actual SimpleFilters feed one actual ApproximateTimeSynchronizer with queue_size=2, slop=0.01 seconds, allow_headerless=False. It registers an actual callback and invokes SimpleFilter.signalMessage in input order. No independent matching calculator is implemented. EXPECTED and TRIGGERS are explicit acceptance assertions from the public matrix, not producers of runtime callbacks.

Each case gets a new graph and empty queues. The epoch case explicitly unregisters both old input connections and the output callback, snapshots and clears old queues, discards that graph, then creates a new graph. Final and failure cleanup use the same owned-resource path. Queue snapshots contain actual SDK timestamp keys and message frame IDs, not inferred eviction counters. Input records absent from callbacks remain retained in feed evidence even if evicted, overwritten, reset or unmatched. Their individual disposition is not invented. Queue introspection uses the supplied installed implementation's queues/callbacks/input_connections attributes; lack of these attributes is an API failure.

Application epoch reset is caller-owned isolation, **not automatic SDK clock reset detection**. The old E1_L cannot feed the new graph. Before E2_L arrives, E2_R's feed_return must still show zero callbacks. Runtime ownership is confined to in-memory connections/queues and one exclusive evidence file; it creates no child process of its own.

PointStamped.header.frame_id equals the exact source ID. Integer nanoseconds split with divmod into sec/nanosec, without conversion to floating timestamp. Point payload is x=event index within case (reset consumes an event index), y=epoch, z=0 for left or 1 for right. This distinguishes duplicate timestamps. Each feed captures actual serialization bytes, SHA256 and actual deserialization fields. The callback serializes/deserializes the actual delivered messages again and compares those snapshots with retained source snapshots. No SDK call is replaced by JSON computation.

## Input and raw evidence schema

Input is the fixed `PROPOSED_LOCAL_MESSAGE_FILTER_INPUTS_V1` JSON contract in inputs/INPUT_CASES.json: generated origin, queue=2, slop_ns=10000000, allow_headerless=false, six ordered cases with left/right integer stamps, unique IDs, and the single explicit epoch RESET. Input validation is bounded and does not claim installed SDK compatibility. Public expected callback identities, in order:

| Case | Callbacks |
|---|---|
| ordinary | O_L / O_R |
| strict_boundary | none (difference exactly 10 ms) |
| queue_eviction | Q_L3 / Q_R3 |
| out_of_order | A_L1 / A_R1, then A_L2 / A_R2 |
| duplicate_stamp | D_L_NEW / D_R |
| epoch_isolation | E2_L / E2_R |

Strict comparison is difference < slop; equal stamp replaces the old SDK timestamp-keyed value; overflow evicts the smallest timestamp. These are installed-source semantics under test, not an alternative algorithm.

Raw output is a single UTF-8 JSON object, schema `LOCAL_MESSAGE_FILTER_EVIDENCE_V1`, newline terminated. Top-level keys include case_id, nonce, status (PASS/FAIL), rows, counts, modules, hashes_before, hashes_after, errors, work_seconds and whole_seconds. Failure may omit fields unavailable before the failing stage. Errors carry a code, bounded detail and usually exception type. Examples include SDK_IMPORT_FAILED, UNSUPPORTED_API, ROUNDTRIP_FAILED, SDK_OUTCOME_FAILED, INVALID_METADATA, RESOURCE_LIMIT, TIMEOUT, WORK_TIMEOUT and CLEANUP_FAILED. Any exceptional or incomplete outcome is FAIL, never an empty completed PASS. Failure cleanup is retained when applicable. Failure to write evidence itself returns CLI failure; it cannot guarantee a partial file when the path/storage is unavailable.

Rows have consecutive seq, kind and case. Their ordered grammar per case is:

* start: epoch=1.
* feed: phase=case, epoch, id, channel, stamp_ns, cdr_b64, cdr_sha256, decoded.
* callback (only if actually emitted during that feed): epoch, trigger feed ID, ordered left/right ids, two snapshots.
* feed_return: id, cumulative callback_count and actual post-feed queues.
* reset, only between epochs: from_epoch, to_epoch, cleanup evidence before new graph use.
* end: final callback_count and cleanup evidence.

A snapshot contains cdr_b64 (standard base64 of actual CDR bytes), cdr_sha256 (lowercase SHA256 of decoded bytes), and decoded={sec,nanosec,frame_id,point:[x,y,z]}. Cleanup includes verified, queues_before and connections_remaining. Feed-return rows establish absence as well as presence of callbacks. The pure validator enforces this event grammar, expected source identity/payload/epoch, callback source snapshot equality, counts, strict trigger timing and resource bounds. It does not decode CDR itself or authenticate who produced the file.

modules maps exact imported module names to resolved file paths, module names and spec origins. hashes_before/after cover this source, pure tests, this document, supplied input, the imported SDK Python sources and message_filters package.xml. These are observed hashes, not a claim that installed bytes match the readonly pin appendix: independent Root review must compare the recorded files/hashes to that appendix and qualify the interpreter and extension binaries. Hash equality protects the recorded interval, not an arbitrary prior approval. SDK transitive libraries and external tool provenance require independent verification. No credentials or private permits are stored.

## Bounds and failures

One attempt, no retry. Input is bounded to 1 MiB and 16 events per case. Evidence is <=256 rows and <=1 MiB; rows are additionally capped below that limit while collecting. CLI prints only a bounded summary (under 64 KiB), never raw CDR. Evidence path must be fresh and distinct from source/input; exclusive creation prevents overwriting an existing file. No parent directories are created.

Runtime requires POSIX alarms, main thread, and no preexisting real-time alarm. A real-time alarm covers the absolute 20-second work budget from entry, including imports. Cooperative checks occur between events. After SDK work, the timer covers the remaining 30-second overall entry budget for verification, cleanup and writing; prior alarm handler is restored. The separately admitted process must still enforce an external whole-child deadline: Python signal delivery cannot preempt every blocking C extension, startup precedes entry, and filesystem writes can fail or be interrupted. The code is not a substitute for that independent child lifecycle bound. Cleanup failure, deadline or source/input/SDK mutation detected by the runtime invalidates success; independent review must inspect before/after equality even for failure diagnostics. No exit receipt or actual child ownership is asserted by source authorship.

## Pure tests and independent acceptance

The unittest file never calls the runtime entry or imports an SDK. It constructs explicitly labelled SYNTHETIC_NOT_SDK evidence using deliberately NON-CDR bytes. Positive tests validate metadata structure only; they cannot claim valid ROS serialization or live SDK callbacks. Negative tests cover wrong pairs, payloads, epoch, extra/missing callbacks, malformed base64, byte/hash substitution, changed timestamp/hash, missing feed, premature epoch callback, cleanup failure, incomplete status, deadlines, row/byte caps and malformed input. A temporary-directory fixture is allowed only when that suite is later authorized to run. These tests have been authored, not executed.

A base64 string with a matching hash is not proof of valid CDR. Independent runtime acceptance must actually deserialize retained CDR using an approved decoder, compare all decoded source/callback fields, verify installed registration paths, six callbacks, exact per-feed event order, pins, source/input immutability, no-init/no-DDS behavior and all owned-resource cleanup. Synthetic fixtures are intentionally accepted by the structural validator to make this proof boundary explicit; self-reported SDK names are not provenance.

## Future workflow, not performed here

Before starting, state that the finite generated local SDK synchronizer test will run without DDS or hardware. Use exactly one freshly approved start command. If pending, wait for the actual watcher/continuation and retain the same operation/task/revision. If already finished, say so and claim no automatic-continuation proof. Obtain both typed terminal status and output for that same run. Deliver the unchanged runtime result as the first genuine Ref, then one bounded report linked to actual result hash/operation/outcomes as the second genuine Ref. Only then stop the future runtime workflow, reporting failures as failures. Component success cannot repair failed workflow, provenance, prior episodes or source review. None of these future orchestration actions, tests or Refs occurred in this read/edit-only source purpose.

## V2 retained-CDR semantic source binding — authored, not executed

This appendix supersedes V1 raw-byte snapshot equality for the repaired runtime, not the historical record. The preceding document bytes and the complete original 22-test module remain prefixes. `SCHEMA` and explicit V1 validation remain legacy metadata compatibility for those unchanged tests; no V1 result qualifies as repaired-runtime evidence. The runtime now emits only `LOCAL_MESSAGE_FILTER_EVIDENCE_V2`. Independent acceptance must require that exact schema. This is a visible version boundary, not a negative-test exemption.

OMG DDS-XTypes 1.3 section 7.4.3.2.2 Table 37 ALIGN(N), https://www.omg.org/spec/DDS-XTypes/1.3/PDF#page=148, states: “The value of the padded bytes is left unspecified.” Root's supplied layout inference identifies absolute offsets 24..27 in the retained 52-byte D_L_NEW representations as alignment padding (PLAIN_CDR_LE origin 4, stamp/string ending at payload offset 20, next double aligned to 8). The supplied feed121 and callback126 have different bytes and hashes there, with equal recorded projections. This is not permission to rewrite padding, nor a claim of an independently completed decoder check.

V2 retains each feed's actual serialization and each actual callback message's independent serialization, own base64, own SHA256, and own actual SDK deserialization projection. There is still precisely one snapshot serialization per feed and per callback argument, no third serialization, retry, normalization, canonical serializer or feed-byte substitution. Equality of decoded fields is checked instead of equality of representations. Both callback captures are appended before source-identity enforcement, preserving diagnostics for a wrong-object/field failure after successful capture. An earlier serialization/deserialization exception remains a scenario failure, not an invented capture.

Each snapshot has exactly `cdr_b64`, `cdr_sha256`, `decoded`, `binding`, `capture`. Binding is an exact typed mapping: case, phase, epoch, id, channel, stamp_ns, event_index, feed_seq. Integer coordinates reject bool substitutions. Feed capture is {kind:feed,row_seq,side}; callback capture is {kind:callback,row_seq,side,same_object:true}. Callback row sequence and ordered side distinguish captures from copied feed snapshots or reused/swapped callback evidence. The pure checker validates each capture's own base64/hash and compares its decoded fields with the retained source's expected fields. It does not require different bytes: legitimate serializations may also be identical.

Runtime retains strong references to every fed PointStamped in its case, including overwritten, evicted, unmatched and old-epoch sources. Before accepting a callback it requires Python object identity (`is`) with that exact retained feed object, unchanged fields/stamp, and matching current case/phase/epoch/channel. Equal fields or a copied frame_id on a foreign object cannot satisfy that runtime check. The SDK graph, callback registration, matching, scheduling, arguments/returns, six input cases and public four-mandatory-argument entry are unchanged. No substitute matcher, ROS initialization, Node, Context, executor, DDS or hardware is introduced. Existing 20-second work/30-second whole child limits, cleanup, final source/input/SDK hashes and metadata validation remain mandatory.

Binding/capture fields are checkable coordinates and observed identity assertions, not cryptographic attestations. A writer forging bytes plus all metadata is not detectable from JSON alone. In particular, copied raw bytes with recomputed hashes and fabricated capture claims cannot universally be distinguished from genuinely identical independent serializations. Independent reviewed SDK registration/child provenance and independent decoding of every retained actual buffer remain required. Appended pure tests deliberately use NON-CDR fixtures, exercise different own-hashed captures, wrong payload/stamp/source/phase/epoch, swapped/copied captures, foreign-object observations, stale hashes and malformed metadata; they prove neither padding validity nor SDK execution. They are after the original __main__ guard and intended for later pytest import/collection; direct standalone testfile main does not cover these additions. No tests were run here.

Original episode81882 remains FAILED/SUPERVISION_CLOSED1, parent exit 1. Its normal STOP/quit and two genuine failure-report Refs do not erase the failed experiment, its original usage/costs or lack of automatic-continuation credit. The supplied failure projection remains untouched. The separate Root saved-forensic whole also FAILED1 at the loaded-ROS-module whitelist after its two deserialize/equal-field/difference assertions; there is no complete independent decoder or SDK PASS. The baseline omitted dynamic packages. Root's OBSERVER_TAMPERING transport/unwind defect and SDK closure/decoder work are outside these three source files; no scenario error is swallowed to evade that defect.

Only these three source copies were edited in this purpose. SDK/tests/imports/processes were not executed, no Refs delivered and no runtime authorization inferred. Current actor/science/environment and Root workers are untouched. Future authorization must still preserve initial public progress before the exact approved process start, actual pending watcher versus truthful fast-completion distinction, same-operation typed terminal status AND output, independent final source/hash/metadata checks and partial diagnostics, two genuine Refs and honest STOP. None occurred here. STOP: authored, not executed; independent qualification remains outstanding.
