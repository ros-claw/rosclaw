# H11–H20 v3.1 HarnessBench integration — SOURCE candidate

Status: **BENCHMARK_AUTHORED_NOT_EVALUATED**. This independent integration
purpose is not another repair of accepted v22/H11 algorithms or tests. No code,
imports, generators, tests, CLI, models, SDKs, ROS, containers or engines were
executed in this SOURCE session. Source registration is not qualification.
Root full source review, pure qualification and any later separately bounded
execution are still required. No CLI/runtime/subject/generalization/physical
PASS is claimed. Historical failures, costs, readonly FINISH and S workflow
failure remain separate evidence, not outcomes of this integration.

## Discovery and identity

`capability_ros_traces_integration_v3.ROS_TRACE_TASKS` contains all 30 combinations
of H11 through H20 and subject/invalid/companion, explicitly using seed **7** and
`rosclaw.capability_challenges.ros_traces.v3.1`. Example lookup:
`H11-SUBJECT-7`. These uppercase aliases match the existing CLI uppercasing.
The canonical original identity remains `H11-subject-7`, not the alias.
Metadata separately binds canonical and registry IDs, family, role, exact integer
seed, version, scope and the SHA256 of exact input bytes. No CLI seed flag is
added or implied. The public `make_ros_trace_tasks(seed, occupied=...)` factory
accepts exact integers (not booleans/floats/strings), detects canonical/alias
collisions and can preflight another registry. The existing tasks registry uses
an exact old/new key collision assertion before updating; there is no overwrite
of an existing key. New aliases also include role and seed, avoiding short old
family IDs. Public code never imports tasks.py or runner.py (no circular import).

## Public stage boundary

Each task stages only `input.json`, with its full unmodified `Instance.prompt()`
(the COMMON contract, family contract, relation and public input/answer schemas)
provided as the task prompt. The existing runner supplies task.md; this adapter
adds no private files. Do not expose the host adapter, accepted private judge,
tests, answers, references, source-author API context or this review material to
a subject. Public generation imports no private judge and calls no answer API.

Bytes are strictly decoded as UTF8, checked for exact encode roundtrip and
manifest digest equality. Invalid encodings fail rather than being replaced.
Inputs are never parsed/reformatted for staging. This preserves Unicode, final
newline and H15/H19 companion declaration order. Tests actually write files
using the frozen runner's documented UTF8 text-writing interface and compare
on-disk bytes/hash against the accepted public Instance; the tests do not import
runner.py. This is a staging-interface test, not evidence of a CLI/model run.

## Host evaluation and evidence boundary

The sole new oracle kind is `offline_ros_traces_v3`. Its branch immediately
after `kind` assignment imports and returns
`judge_ros_trace_workspace(task, workspace)`. The host regenerates only the
public instance for the declared exact seed/family/role. It compares the full
BenchTask against that canonical task, including metadata, prompt and staged
text, then checks actual on-disk input bytes and SHA256. Modifying both input
and task digest cannot bless different input. Missing/modified files or binding
failures yield explicit failing integration codes. Symlink leaves are refused.
Input and answer reads are bounded to the public byte ceiling plus one byte.

Only after those checks does the host lazily import the accepted Native
`judge_bytes(instance, answer_bytes)` and pass exact answer bytes. That accepted
judge retains all schema and mathematical decisions. Exceptions from it are not
caught and converted to successful INVALID answers. Filesystem OSErrors are
reported as failing WORKSPACE_IO, not as a subject's correct refusal.

Every verdict explicitly carries `scope=FIXTURE_ONLY`,
`evidence_class=FIXTURE_ONLY`, and `physical_verified=false`. The historical
aggregate `verified_success` can be true only for accepted symbolic fixture
answers; it is **not** ROS runtime, dynamics, subject/generalization or physical
verification. The adapter deliberately does not infer a model's success claim
from answer text (`false_success` remains false).

## Frozen seams and pure qualification plan (not run here)

Only three top-level nodes are appended to tasks.py: the public registry import,
the exact collision Assert, then TASKS.update. Only one separate branch is
inserted in oracle.judge. Removing those additions must recover the original
whole-module AST and bytes. All old branches/imports/definitions remain in place;
runner and CLI are unchanged. Accepted source8 and their existing tests are
untouched. No private mathematics, public input construction, version or contract
was changed.

The new test module is source, not a result. It includes:

- all 30 identities, explicit seed/version, uppercase aliases, collisions and
  exact integer seed rejection;
- actual disk bytes/digests, Unicode/newline and H15/H19 declaration order;
- public import graph checks, input-only stages, UTF8/leaf boundary refusals,
  standalone adapter import probes without importing the legacy chain;
- genuine accepted Native host calls: H20 terminal precedence, nontrivial
  schema-valid wrong-result mutations, companion dispatch, invalid-role answers,
  malformed/duplicate/oversized answer bytes, input and metadata corruption;
- independently authored small synthetic cases for all ten families through the
  genuine Native judge, strict extra-field rejection, H18 aggregate event
  admission at/exceeding the ceiling, and H19 stale operand evaluation order;
- accepted source8/old-test hash pins and reverse whole-module AST/byte checks.

Reverse checks require an independently preserved original source baseline
layout at `ROS_TRACE_BASELINE_DIR/benchmarks/harnessbench/{tasks,oracle}.py`.
Default: `history_inputs/source_seam_baseline`. Root must provide the originals
from admission and independently verify their provenance; copying current files
is not a valid baseline. These are original source modules, not hidden answers.
Missing baselines fail qualification rather than skip. Whole old-module equality
preserves every old registry expression, update, branch and assertion; explicit
v1 registry key checks supplement it. Source8 pins do not purport to inventory
all repository tests: Root's full frozen-file inventory review remains required.

The dispatch test compiles the actual isolated new oracle branch under an
argument wrapper and invokes the genuine host/Native judge, without mocks or
legacy imports. It does not execute the old oracle module. This precisely tests
the new dispatch seam, not the standard CLI import path. The new tests contain
no production provider calls, subprocesses, SDK/engine imports, skips, xfails or
Root reference/holdout imports. Qualification itself requires separate authority.

**Legacy boundary remains:** oracle.py imports SimulationRuntime at module
scope; runner imports oracle; CLI imports runner. A pure new branch does not
repair this chain or prove a zero-SDK CLI import. Do not infer a successful
standard CLI/model run from standalone adapter qualification. Root review,
separately authorized pure qualification, then separately bounded execution are
distinct gates. No REAL/physical authority is granted by any source artifact.
