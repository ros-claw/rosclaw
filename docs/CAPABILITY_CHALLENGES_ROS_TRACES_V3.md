# ROS trace challenges v3.1: H11–H20

**Status: BENCHMARK_AUTHORED_NOT_EVALUATED.** Source-only repair, not accepted
benchmark evidence. No Python, tests, checkers, providers, ROS, containers,
SDKs, physics or hardware have been run in this author session. Independent
Root source review, freezing and evaluation are required. This is separate
from v2.2; no legacy baseline, tasks or grader is modified or re-graded.

## Isolation, version and API

Public construction: `benchmarks/harnessbench/capability_challenges_ros_traces_v3.py`.
Private author candidate: sibling `capability_challenge_judge_ros_traces_v3.py`.
Tests: `tests/benchmarks/test_capability_challenges_ros_traces_v3.py`.
Never stage the private candidate or tests to subjects. Public construction
imports no judge, provider, old benchmark, execution engine or external SDK.
The private candidate imports the new public module and standard library only.
Neither module dispatches execution or performs file I/O on import.

`VERSION` is now `rosclaw.capability_challenges.ros_traces.v3.1`. Old v3.0 input
versions are not accepted as v3.1. Do not reuse old grades under this version.
The new source must be independently frozen and graded. `_base`,
`make_instance` and `make_plan` are left untouched: every original family's
subject/invalid/companion decoded `data` remains unchanged at each seed. Input
version bytes, exact byte hashes, manifest version/contract hashes, and prompts
intentionally change. This is not a claim that an AST/data checker was run.

Exports include `VERSION`, `COMMON`, `CONTRACTS`, public resource constants,
`INPUT_DATA_SCHEMAS`, `RESULT_VALUE_SCHEMAS`, `make_instance`, `make_plan`.
Each plan has 30 instances, family-major H11–H20, then subject/invalid/companion.
Seeds are exact Python integers. The original SHA256-derived small integer
construction remains deterministic. Instance exposes `staged()`, `manifest()`
and `prompt()`. Only `input.json` is staged, with exact version/family/data keys.
Encoding is sorted compact JSON plus newline except intentional H15/H19
companion declaration-order permutations. Hashes always cover actual bytes.

The prompt includes COMMON, family prose, S/I/M relation, supported-input and
answer Draft202012 schemas. Schemas and prose jointly define acceptance.
H17/H18 schemas expose structural domain bounds; generic schemas for other
families defer their data shapes to prose. Aggregate node/string/event counts,
UTF8-byte lengths, dynamic live-edge capacity and error precedence are
procedural constraints, not claimed to be enforced by JSON Schema alone.
Public source includes no expected answers or private reference imports.

## Shared admission: exact precedence and error codes

All ceilings below are inclusive. Resource rejection is public, not a hidden
judge exception. Invalid inputs return INVALID with null values. Bad answers
fail the verdict; they are not successful INVALID results.

1. Raw bytes above **1048576 (1 MiB)**: `SIZE_LIMIT`. Strict UTF8 decoding
   follows, with invalid UTF8 producing `INVALID_JSON`.
2. Lexical preflight, left-to-right, **before recursive JSON decoding or numeric
   conversion**: at most **32** simultaneously open arrays/objects, root depth
   one; at most **20000** tokens. Each opening container, quoted string
   (including keys), number or literal counts once. Whitespace, commas, colons
   and closers do not count. Escaped quotes/brackets inside strings do not
   affect nesting. A numeric lexeme is the maximal non-whitespace run bounded
   by `[]{}:,"`. Check number length **64 characters**, integer digit count
   **20** (minus excluded), JSON number grammar, explicit float exponent
   absolute value **100**, then finite float magnitude **1e100**, in that
   order. Exceeding any ceiling gives `RESOURCE_LIMIT`; bad number grammar or
   non-JSON literals such as NaN/Infinity give `INVALID_JSON`. Resource failure
   encountered before later malformed syntax wins. Token count is checked
   before processing that token's depth/number limits. Bounds include
   zero-padded exponents. Thus `1e9999` is RESOURCE_LIMIT, not an infinity
   passed into recursive shape checks; a 5000-digit integer is never converted.
3. Complete JSON decoding rejects syntax and duplicate keys as `INVALID_JSON`.
4. **Input only**, walk the entire decoded envelope depth-first in insertion
   order, keys before their values. At most **2048 nodes**, counting every
   value and key once, including root; at most **128 entries per container**;
   at most **256 UTF8 bytes per string/key**, **16384 total string/key bytes**.
   For these counts only, UTF8 surrogatepass counts a lone surrogate as three
   bytes; H11's label encoding still requires strict UTF8. Integer magnitude
   at most **10^15**, float magnitude at most **1e50**; bool is not numeric.
   Any exceeded ceiling gives `RESOURCE_LIMIT` before envelope/family shape.
   Per-item order is node count, then the applicable string/container/numeric
   checks. This headroom permits revision increments and TF arithmetic without
   making their answers exceed shared JSON number limits.
5. Validate envelope and family shapes/semantics in their public order.
   Wrong generic shapes/types give `INVALID_INPUT`; family-specific errors
   remain as documented in CONTRACTS, including H16 MANIFEST.

Steps 1–3 also apply to answers. The input-only ceilings do **not** constrain
answer arrays or reason strings: legitimate H18 answers can have 4096 events,
and a well-formed answer with a reason padded to exactly 1 MiB is admitted.
Private `judge_bytes` reports `ANSWER_ENVELOPE` for bad answer parsing/shapes
and exposes the underlying parse code as detail. A valid answer then must
match the independently specified result (the supplied candidate is only an
author implementation, not that independent specification).

Answer keys: exactly scope,result,input_sha256,reason; scope FIXTURE_ONLY,
lowercase SHA256 of exact input bytes, nonblank reason. Result keys exactly
status,code,values. OK/OK uses family values; INVALID uses a public code and
null values; no BLOCKED. Integer tokens, strings, booleans and ordering are
exact. Only H14 rotation/translation and H15 sample slots use abs/relative
1e-9 tolerance. Booleans never satisfy numeric slots.

## H17 bounded ownership/wait graph

After exact data keys and groups/callbacks object and events array checks,
require **8 groups, 8 callbacks, 128 events** maximum, otherwise RESOURCE_LIMIT,
then validate declarations (DECLARATION). Events are checked sequentially for
shape/legality (EVENT). At most **16 distinct live wait edges** are allowed at
every prefix. A legal new edge exceeding capacity gives RESOURCE_LIMIT before
insertion; an already-present wait is idempotent even at capacity. Invalid
endpoints, self-waits and malformed events still give EVENT before the live
capacity check. Finish removes incident edges; unwait releases its edge.
A later release cannot legalize an earlier over-capacity prefix.

Only after event processing may cycle traversal run. At most eight vertices
bounds recursion depth to eight, with fewer than 8! partial simple paths per
start (at most eight starts); edge count is also bounded. The candidate retains
the original cycle rule: lexicographically least simple directed cycle, rotated
to its least vertex and closed by appending start, not shortest. Dense
20-callback graphs are outside the declared supported domain regardless of
encoded byte size. No real scheduler, starvation or fairness claim is made.

## H18 bounded event expansion

After exact keys and publications array type, check **128 publications** maximum
(RESOURCE_LIMIT), then integer/sign/positive-period checks (TIME). Next require
all publication times, horizon, deadline and lease <= **10^12** (RESOURCE_LIMIT),
then nondecreasing publications and horizon >= last publication (ORDER).

**Before allocating or emitting any expiration arrays**, count using integer
arithmetic for each publication t: end=nextpub-1, or horizon on the final
interval; D=max(0,(end-t)//deadline); L=1 iff t+lease<=end, else 0;
R=L for nonfinal intervals, else 0. Require **sum(D+L+R) <=4096** across the
whole trace, otherwise RESOURCE_LIMIT. This combines deadline_misses,
liveliness_lost and recovered, not 4096 per array. Count work is O(publications)
independent of horizon/period; expansion occurs only after complete admission.

Publication at exact expiration precedes and suppresses it; final horizon is
inclusive. Equal-time publications create no artificial expiration/recovery.
No publication means all arrays empty and alive=false. Every emitted event time
is <= horizon <=10^12, so at most 13 decimal digits: the three compact arrays
together require <=57350 bytes, excluding their keys and envelope. The supported
result always fits the 1 MiB answer ceiling using a short reason. In particular,
a tiny `[0]` trace with unit deadline and enormous horizon returns RESOURCE_LIMIT
without constructing a million-element output list.

## Other family coverage and remaining output budgets

| Family / CAP80 | Construct and planned invalid | Metamorphic obligation |
|---|---|---|
| H11 / 001 | Nested little-endian wire format; TRUNCATED | Documentary alias rename; canonical roundtrip |
| H12 / 002 | Ancestor parameter lookup; TYPE_MISMATCH | Move subtree/namespace under /m |
| H13 / 003 | Rewind epochs and sample tie order; LENGTH | Shift second epoch stamps, keep bindings |
| H14 / 004 | Unique-parent TF composition; MULTIPLE_PARENTS | Rename and conjugate by diag(-1,-1,1) |
| H15 / 005 | ROS1/2 header/time loss declaration; LOSSY | Declaration permutation; seq=0 roundtrip |
| H16 / 006 | Manifest-only ABI comparison; MANIFEST | Symbol permutation |
| H17 / 007 | Bounded wait graph; EVENT | Order-preserving callback rename |
| H18 / 010 | Bounded symbolic timers; ORDER | Shift times/horizon within supported domain |
| H19 / 011 | Atomic revision transaction; PREDICATE | Update-key permutation |
| H20 / 012 | Cancel/result race; EVENT | Goal rename and indexed idempotent duplicates |

All metamorphic obligations are conditional on both inputs remaining within the
supported domain. Family semantic rules are otherwise unchanged: H11 requires
strict UTF8 labels and lowercase even hex, rejects trailing bytes; H12 remaps
once after cycle validation, missing is OK; H13 ties use gap/stamp/index and
samples are reusable; H14 identity queries work for unknown equal frames and
rotation checks precede parent/cycle checks. H15 default seq=0 may drop without
loss; negative duration normalizes, negative stamp rejects. H16 incompatible
manifests are OK compatible=false, with no library loading. H19 declarations
precede stale revision, stale does not evaluate operand types, current revision
validates all operand types before predicates and commits atomically. H20 ack
is nonterminal, first legal terminal wins; duplicate submit always invalid and
unknown late events/goals still reject. See each complete public contract.

The input-wide limits also bound non-H17/H18 computation: H13's largest match
matrix is 128x128 (16384 cells), fewer than 20000 JSON tokens including its
other outputs/envelope with a short reason, and well below 1 MiB. H12 ancestor
visits are bounded by a 256-byte path; H14 tree/path size by 128 input edges
(and the tighter node budget). Wire expansion is bounded by 2048 input nodes
or the 256-byte decode hex string; H19 merge uses at most two 128-entry maps;
H20 has at most 128 events. Total source strings are capped at 16384 bytes;
even JSON escaping and repeated bounded paths leave ample byte headroom.
Integers in arithmetic outputs stay within 20 digits, while TF real outputs
remain far below 1e100. These are design bounds, not measured performance.

## Authored verification, not run

The original 14 test functions, imports, helpers and assertions remain an
unchanged prefix. Appended tests cover exact and one-over byte/depth/token/
number/node/container/string limits, large integer/exponent/depth rejection,
malformed UTF8/duplicates/number grammar, public error codes and precedence,
H17 callback/group/event/live-edge limits, duplicate waits and capacity release,
H18 publication/time/aggregate expiration limits including losses/recoveries,
wrong types/shapes, representable output headroom and larger bounded outputs.
Explicit H17/H18 data assertions cover all three roles at four seeds. Original
per-family tests still cover all 30 role outcomes and metamorphic relations.
No self-skip/xfail, subprocess, provider, SDK or test execution is introduced.
Largest single byte boundary fixture is 1 MiB+1; graph/array fixtures are bounded.

No test pass count, accepted score, baseline AST comparison result, simulation,
live ROS evidence or physical execution is claimed. Source registration and
hashes establish provenance only; author candidate agreement, even if later
observed, cannot substitute for Root's independent references and acceptance.
