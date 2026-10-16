# Stronger10: source authored, not evaluated

Actor: ActorQ256 PI1.1 ChatGPT. Version `rosclaw.capability_challenges.v2.2`.
No tests, syntax checks, evaluation, provider tests, ROS, engine, physics, neural
inference or hardware were run. Root must independently review the candidate
reference calculations, qualify tests/mutations and separately authorize evaluation.
Author-owned test agreement is not independent correctness evidence.

## Isolation and staging

The public module exposes `build_registry`, `make_instance`, `Instance.prompt`,
`manifest` and optional lazy `as_bench_task`. No tasks.py registration is changed.
The judge, tests and this document stay Root-side. Stage ONLY an instance's files
and public prompt. Never give subjects repository access, reference code, tests,
expected values or private Root algorithms/data. No original cases or judges are
imported. Old fixed inputs, hashes, four-key answers and grading remain frozen;
this judge never submits generated inputs to the old judge.

The new answer retains exactly scope,result,input_sha256,reason. `result` is a
versioned status/code/values object. The public contract defines output fields.
Invalid inputs return INVALID with null values, never repaired numeric success.
The finite registry pins all auxiliary bytes as well as input.json. Alteration
fails immutable-input grading, including whitespace and stale auxiliary hashes.
H10 damaged checkpoint bytes are intentionally pinned, not silently fixed.

## First30 provenance frozen before expected values

Each row creates S (success/boundary), I (invalid), M (metamorphic companion).
Seed expansion is SHA256(decimal seed), first two bytes big-endian modulo5+1.
No Python PRNG/version dependency. JSON is sorted compact UTF-8 plus newline.
Manifests expose exact file and contract SHA256, version, seed and relation.

|Family|Seed|I trigger|M relation|
|---|---:|---|---|
|H01|2026101601|missing type|id/node/topic renaming, reverse records|
|H02|2026101602|zero quaternion|q sign and frame rename|
|H03|2026101603|string timestamp|uniform +100 ticks|
|H04|2026101604|truncated buffer|endian/channel/padding reencoding|
|H05|2026101605|frame mismatch|point permutation|
|H06|2026101606|negative weight|order-preserving node rename|
|H07|2026101607|zero integration dt|m to mm|
|H08|2026101608|zero R|offset measurements/state/center +7|
|H09|2026101609|duplicate joint name|named channel permutation|
|H10|2026101610|damaged checkpoint JSON|landmark id rename|

Numeric tolerance: absolute/relative 1e-9; quaternion unit check 1e-12.
Types, shapes, keys, names, booleans, hashes and status codes remain strict.
The ORIGINALS mapping preserves coverage lineage to all50 directions, NOT proof
that three samples exercise every subdirection. New tie/integration/recovery rules
are explicitly versioned; unavailable old private semantics are not inferred.

## Qualification limits

Hand-calculated assertions, metamorphic checks, stale/altered-input rejection,
typed-answer mutations, ambiguity, budget exhaustion and unreachable paths are
software tests authored only. Root should add independent references and mutations
for every subdirection: lifecycle/QoS combinations, arbitrary rotations/composition,
out-of-order streams, invalid dimensions/nonfinite values, obstacle monotonicity,
diagonal corner edges, asymmetric limits/addressing and idempotent recovery.
The first30 graph has a diagonal prohibition but no corresponding edge; it does
not alone test removal of an actual corner-cutting edge. Seeds are finite examples,
not distribution coverage, holdout secrecy, generalization or capability evidence.

H10 batch file names are symbolic workload entries, not actual file/tool execution.
Its history/budget result is a plan only. Cost, interruption, current revision,
actual reads, registration uniqueness and STOP require independently captured
transport evidence. The judge explicitly marks operational recovery unverified.
Filesystem adapter is bounded read-only convenience, not an adversarial filesystem
sandbox (TOCTOU/parent symlinks need trusted staging). Stdlib imports are declared,
not installation or runtime verification claims. Later Root evaluation must enforce
subject/reference isolation, resource caps, authority and separate approval.

## v2.2 public output precision repair (source only)

Status: `BENCHMARK_AUTHORED_NOT_EVALUATED`. Each original family contract is
retained verbatim as a prefix, followed by field/type/order clarifications.
Only generator VERSION, CONTRACTS, literal schemas and Instance.prompt changed;
_payload, make_instance, manifest and the remaining generator AST are preserved.
The judge is read-only and untouched, importing the new VERSION naturally.
The original five tests/helpers/assertions are retained; seven tests are appended.
They are authored, not executed. No syntax check, scope-guard execution, benchmark,
model/provider call or evaluation was performed in this phase. jsonschema is a
qualification-test dependency, not a new generator/runtime dependency; its
availability has not been checked.

### Envelope, result and values are different levels

The answer envelope has exactly `scope`, `result`, `input_sha256`, `reason`.
Scope is `FIXTURE_ONLY`; the hash is lowercase SHA256 of the exact input bytes.
Reason is a non-whitespace string explaining fixture-only limits. Result has
exactly `status`, `code`, `values`. OK requires code OK and the family values
object below. INVALID requires the applicable documented input-error code and
values null. The emitted schema enumerates family input-error codes; the planned
invalid variant still uses ERRORS. Only H10 permits BLOCKED/BUDGET_EXHAUSTED,
whose values has exactly history and required_calls, preserving history unchanged.

All object keys are required and no extra/duplicate keys or aliases are allowed,
at every depth. JSON object member order is irrelevant. All numeric positions
require finite JSON numbers, excluding booleans and numeric strings. Integer
positions require integer tokens, excluding decimal/exponent tokens and booleans.
Boolean positions require true/false, not 0/1. Arrays are ordered, never scalars.
Null is legal only in explicitly nullable positions. Numeric comparison tolerance
remains absolute/relative 1e-9; it does not excuse shape or key mismatches.

In this table N means finite number, I genuine integer, B boolean, S string.
`T[k]` means an ordered array of exactly k elements; `T[]` has variable length.
Each row lists the complete OK values key set, not an example answer.

|Family|Exact OK values fields and ordering|
|---|---|
|H01|links: {id:S, compatible:B}[], one per input edge ascending id; topic:S; state:S (unconfigured/inactive/active), never lifecycle|
|H02|point:N[3], roundtrip:N[3], gravity_free:N[3], torque:N[3], all x/y/z; distance:N, never plane_distance; frame:S; fresh:B|
|H03|matches:I[anchor_count][stream_count], original anchor then stream order, -1 or original stream index; rewinds:I[] ascending clock encounter indices; lost:nonnegative I; fresh:B|
|H04|rgb:I[pixel_count][3] in R/G/B order with channels 0..65535; valid_depth:B[pixel_count]; xyz:(N[3] or null)[pixel_count], x/y/z iff valid_depth true, otherwise null; all arrays row-major|
|H05|voxels:I[][2] and blocked:I[][2], unique [x,y] pairs sorted lexicographically by x then y; voxels can be negative, blocked inside grid; empty arrays allowed|
|H06|reachable:B; cost:N or null; path:S[] in start-to-goal sequence; braking_safe:B; time_valid:B. Unreachable means null cost and empty path; reachable means nonnegative numeric cost and nonempty path|
|H07|reachable:B; graspable:B; odometry:N[3] in x/y/theta order, meters/meters/radians|
|H08|ma:N[sample_count], ema:N[sample_count], posterior:N[sample_count], variance:N[sample_count], ALL aligned to original samples including gated-out samples, without initial-state prefix; posterior and variance are per-sample arrays, never final scalars. accepted:I[] ascending original sample indices|
|H09|joints (never rows): {name:S,address:nonnegative I,delta:N,command:N,force:N,action:N}[], one per channel ascending name; action is mapped output, not normalized input|
|H10|decision:S in ACT/WAIT/AMBIGUOUS/NOT_FOUND; target:S for ACT/WAIT, null otherwise; history:S[] duplicate-free preserved/append order, not sorted; required_calls:nonnegative I|

RESULT_VALUE_SCHEMAS provides literal per-family OK values schemas.
RESULT_SCHEMAS provides literal per-family INVALID result branch schemas.
Instance.prompt exposes a complete Draft 2020-12 answer schema combining the
family OK and INVALID branches, plus H10 BLOCKED. It specializes public array
cardinalities from the staged input dimensions/counts, without computing outputs.
Fixed vector lengths, nested keys, nullability and status branches are structural;
ordering, hash equality, input-derived semantics and validity alignment remain
additional textual obligations. Standard JSON Schema treats some decimal numbers
as integers; the stricter token rule is stated explicitly and the appended type
test uses a strict checker. Schema validation alone is not the private judge and
must not be treated as proof of numerical correctness or judge equivalence.

Changing VERSION intentionally changes input.json bytes and therefore input
hashes; extending CONTRACTS intentionally changes contract hashes. Existing
manifest logic still binds both. There are still exactly the same thirty task
IDs, roles and seeds; numerical payloads, transformations and auxiliary fixture
bytes are unchanged. Do not reuse v2.1 answers/grades as v2.2 results. Old 6/10 is
retained unchanged but contract-confounded: four naming/shape mismatches do not
establish a computation ranking. No prior grade or answer is edited here.
Root independent boundary review, dependency qualification, test execution and
fresh grading remain separate. No concrete expected outputs, new reference
algorithms or answer payloads are included in the public schema or prompt.
