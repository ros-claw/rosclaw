# Owned Darwin camera factory — SOURCE only

Status: SOURCE_AUTHORED_NOT_RUNTIME_VALIDATED. Pure tests, syntax/lint/type
checks, actual CGL, original strong camera test and cross-platform execution:
**NOT_RUN**. This document is not camera, pixel, simulation or hardware evidence.
The session authored files through file tools; no shell, standalone SDK test,
provider test, ROS, container, physics or NN execution was performed. Host model
source authoring is not a claim of zero host provider/SDK activity.

Task binding: `history_inputs/TASK_PROMPT.txt`, supplied SHA256
`efd4f6a6b8a28d6c11fb86ae5a240250486bc41c93e246509be9cb9da41dad95`.
The first file write was a progress report binding that task before any artifact
registration. Current authority is source authoring only; historical NO_GO
labels remain provenance. No runtime approval is inferred from this source work.

## Design and exact scope

`create_renderer(*args, **kwargs)` preserves original positional/keyword calls.
Outside Darwin it returns `mujoco.Renderer(*args, **kwargs)` with no CGL probing.
On Darwin it constructs a private subclass of the pinned classic Renderer.
It does **not** call the upstream constructor: precreating a context then calling
that constructor would silently allocate its default context instead.

Only the constructor/lifecycle is adapted. Width/height framebuffer validation,
scene/options/rectangle creation, font-scale handling, offscreen buffer selection,
readDepthMap, and default flags follow the primary. Render, update_scene, depth,
segmentation, camera lookup, output shape/dtype, flip and context-manager methods
are inherited as the upstream function objects, not rewritten or wrapped.
No upstream module/global assignment, dependency modification, environment-driven
GL disabling, cached substitution, or monkeypatch is used in production source.

Exactly three worker sites change: the two backend embedded workers and the
independent vision oracle embedded worker. Each adds exactly
`from rosclaw.sim.backends.mujoco.owned_renderer import create_renderer` and
changes only the Renderer function target, preserving its original arguments.
All model/state restore, forward, output and oracle projection/metadata math
remain untouched. Oracle input still independently produces its own pixels from
original XML/state, not producer masks. The original strong camera test is frozen.

## Finite native selection and lifecycle

Exactly these ordered integer arrays are admitted, with no other formats/retries:

1. `(99,4096,8,24,11,8,12,24,13,8,59,55,1,56,4,73,0)`
2. `(99,4096,8,24,11,8,12,24,13,8,73,0)`
3. `(99,4096,8,24,11,8,12,24,13,8,0)`

These correspond to upstream legacy/MSAA/accelerated, legacy/no-MSAA/accelerated,
and legacy/no-MSAA/unrestricted acceleration. The prior observer reported that
only the third succeeded, as **Apple Software Renderer**, on that hosted Mac.
That diagnostic was not a camera PASS, not fresh execution here, and not proof
of hardware acceleration. Two earlier SIGSEGVs remain historical limitations.

Owned bindings are loaded lazily, with explicit native argument/result types.
Choose/create require zero integer RC, positive npix for choose, and non-NULL
pointers. Each partial pointer is released before moving to the next candidate;
cleanup failure aborts rather than continuing. Native release APIs are void:
release return codes remain UNKNOWN, never fabricated as zero. Error messages
use numeric RC, not CGLErrorString or raw native string-pointer inspection.

Thread identity is captured on creation. Repeated make_current rebinds the same
selected context but locks it only once. Render's inherited make_current thus
cannot accumulate unmatched locks. Explicit close frees Mjr resources with their
selected GL context current, then unlocks, restores the borrowed prior context,
and releases context/pixel format in that order. Successful close is idempotent.
A failed cleanup preserves remaining ownership for an explicit same-thread retry.
A failed set-current is treated as potentially having changed the binding, so
cleanup restores the prior binding. No fourth format or renderer-construction
retry is introduced after make-current/Mjr failure.

Caller contract: use, render and explicitly close on the creating thread. Keep
the borrowed prior CGL context alive; do not independently switch/destroy it
while this renderer owns the binding. Context nesting must be LIFO, not arbitrary
interleaving. Cross-thread use/close fails without native cleanup. A destructor
is only a best-effort same-thread fallback; abandoning an object on another
thread can leak resources rather than risk an unsafe native call. This design
does not promise general shared-context synchronization or concurrent rendering.

Optional query_strings uses public `OpenGL.GL.glGetString` (bytes/None), never
pointer indexing. Its 512-byte slice bounds Python processing/log size only;
PyOpenGL's native c_char_p read is **not** bounded by that slice. It is not called
as a constructor safety test or used to claim pixel success.

## Tests authored, not executed

New pure tests supply fake CGL and fake engine APIs. Upstream Renderer class AST
is extracted from the pinned read-only source and compiled with fake globals;
upstream module imports and its constructor are not executed. No live model,
GL context, production provider call, or physical action is needed by these tests.
Assertions cover exact finite ordering, stop-on-success, RC/npix/NULL rejection,
partial creation and renderer failure cleanup, balancing/repeated lifecycles,
thread ownership, cleanup-error retry, no upstream mutation, non-Darwin exact
argument delegation, Darwin constructor avoidance, RGB/depth/segmentation math,
output buffers, flag restoration, camera lookup, and bytes/None query handling.
They depend on the supplied static primary fixture and NumPy/pytest. They are
not a replacement for the original strong-camera test.

Manual source reasoning only was performed here; no parser/checker/test was run.
Root still needs full effects/alias review and inverse-AST scope verification,
including negative controls for argument, state, physics, oracle and retry
mutations, before pure test execution. Later actual work requires separate Root
authorization and the unchanged original strong camera test with independent
`pixel_count > 4` and `physics_steps_by_oracle == 0`, plus the original
cross-platform job. Source artifact registration grants no such authorization.

## License and exact installed-byte provenance

Adapted constructor and CGL references retain DeepMind's Apache-2.0 header in
the new module. Complete primary/license copies remain read-only in
`history_inputs/primary/`. These pins describe installed MuJoCo **3.13.0** bytes,
not a claimed upstream Git commit. Changed installed versions/bytes require
review before relying on inherited semantics; version ranges alone do not
establish byte identity.

- classic/renderer.py: `c193df6a8b8cc1659819abd0ded5c17dab0e3e64edb7af6a1e8d249bb4a4a548`
- classic/gl_context.py: `a3cac3b6d582c149a93e95e4742c7fd4ad2083d31a329ee5b675e59e5377ff75`
- cgl/__init__.py: `075193793fb0743385eb37d0ca46ac3949a6eb1db6ad5bf41091bc588d96d41b`
- cgl/cgl.py: `b836c7148846cd0d6ea1f461c3ad60eade74e2687b10d9c68a7de06db50467c0`
- MUJOCO_APACHE_2_LICENSE.txt: `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`
- MUJOCO_THIRD_PARTY_LICENSES.txt: `aec5167579b94d6926340175b4f764b5159f4933657556d89bbfb8238d3b3eb8`

The seven current artifact registrations comprise these five authored source
components, the evidence report, and the delivery manifest. Registration hashes
bind actual authored file bytes; they do not certify runtime correctness.

## Bounded primary/type/cleanup repair — 2026-10-15

This section supersedes the current-status and registration-count statements
above without changing the historical document prefix. Current task:
`OWNED_DARWIN_TYPED_PRIMARY_CLEANUP_REPAIR_SOURCE_20261015_V1`;
`history_inputs/TASK_PROMPT.txt`, supplied SHA256
`6948b9bdac8205340eff32a33b095c2fd34b4444109ce45cba244ac128438ef3`.
The first current-task write was a progress checkpoint in reports/evidence.json,
before any registration. This revision is SOURCE_AUTHORED_NOT_RUNTIME_VALIDATED.

The previous formal result remains LOCAL_MODEL_REQUEST_ALLOWANCE_EXHAUSTED
(367.0028 seconds, 846135 inclusive input, 14859 output). Root's subsequent
pure21 PASS applies only to the old source hashes, not this revision, and does
not erase that formal failure. The original six references remain historical;
the preceding seven-registration statement is historical and not a claim about
this task. This task registers exactly five current snapshots: the selected
renderer, tests and document as diagnostic_source; evidence as diagnostic_report;
delivery as progress_report. Producer/backend and oracle are frozen and are not
registered again.

### Authored repairs and retained contracts

Before an owned renderer type can be built, the supplied installed MuJoCo
module must expose exact __version__ 3.13.0. The actual inherited Renderer module
must expose that same class and a readable __file__; SHA256 of its entire file
must match the installed-byte primary pin above. Missing/unreadable source or
changed bytes fail closed before owned CGL allocation, with no cached approval.
This is a compatibility check, not a hostile Python-process attestation or a
TOCTOU guarantee; it does not verify a signed wheel or compiled extension.
Non-Darwin delegation does not enter the gate and remains unchanged.

_checked now accepts only exact built-in int zero; False, True, float zero,
ctypes integer wrappers and None cannot report native success. Eight binding
prototypes and three literal format arrays/order are unchanged. No SDK prototype
confirmation, additional format, upstream method rewrite, global patch, Linux
relaxation, backend change or oracle change is introduced.

Initialization catches BaseException only to attempt safe same-thread cleanup
and then preserves cancellation. Ordinary format errors retain the old finite
selection behavior only when cleanup succeeds. Cleanup failure aborts selection
and re-raises the identical original error object, not the cleanup error.
Cancellation never advances to another format, even if cleanup succeeds.
The original exception carries owned_cleanup_error, owned_cleanup_owner and a
snapshot owned_cleanup_status when cleanup fails, plus a note about remaining
ownership. The retained owner exposes a fresh ownership_status for inspection
and explicit same-thread cleanup. Handles are cleared only after their release
call returns; a raising release means ownership/release is unconfirmed, not a
claim that native memory is definitely still allocated. Bound/locked flags on
failure similarly reflect unresolved obligations, not measured driver state.
Cleanup stops at a failed prerequisite rather than freeing dependent resources:
in particular an unfreed Mjr keeps its GL context. Initialization failure does
not trigger a hidden destructor retry. Successful normal close remains
idempotent; normal explicit-close failure still permits an explicit retry.
These retained exception/owner references require caller attention; dropping
both without recovery may leak resources. No physical safety claim is made.

The original 14 test functions/21 cases and assertions remain, with import
ordering, reference-equivalent SimpleNamespace naming, fake snake-case methods
with exact instance API aliases, and a same-expression local callback def.
The boundary fixture adds narrowly scoped fake version/module/file metadata
for the pinned AST class. Its module registration is test-only and restored by
monkeypatch; production never patches upstream globals. The Darwin-positive
factory fake likewise points at those exact pinned bytes. Four new test
functions author 17 parameterized cases (38 total intended): version/full-byte/
missing/unreadable gates before allocation; exact RC types including False;
partial-context cleanup failure identity and cancellation; renderer cleanup
failure identity, retained Mjr/GL ownership and explicit recovery. They compile
only the pinned class AST under fake globals; no installed upstream constructor
or production provider is exercised.

### Validation boundary

All current-revision pure tests, Ruff, syntax/type checks, actual CGL and camera
checks are NOT_RUN here. Repairs were authored and reviewed as source text only;
there is no new PASS receipt. Root must perform inverse-AST/frozen-scope review,
pure execution (including old21) and Ruff later. Separate actual-camera approval
is required for any live provider/CGL work and the unchanged original strong
camera test. Historical SIGSEGVs and Apple Software Renderer observations remain
limitations, not current evidence. No Bash, standalone SDK test, ROS, Docker,
physics, NN, hardware or actual-camera execution occurred in this source task.

## Final cleanup diagnostic isolation — 2026-10-15

Current task: OWNED_DARWIN_FINAL_CLEANUP_AUTHOR_REPAIR_SOURCE_20261015_V1,
TASK_PROMPT.txt supplied SHA256
bce97a77d732cda5b473ea7aae4bc1b1ee21aaebf92fc0bcfd5f09e52921a3d3.
Only the cleanup helper body changed; two test functions (six parameterized
cases) and this appendix were appended. Existing callers, ownership/release
rules, primary gates, old 18 test methods/38 cases and document prefix remain
untouched by these edits. No AST checker was run to certify that scope.

Cleanup failure still returns False so callers re-raise the exact original
BaseException. Owner/error/status attachment, status access/representation and
add_note are independently best-effort, including against BaseException hooks.
Status failures are labeled UNKNOWN, not release success; accepted notes report
unconfirmed attachments. If both attributes and notes reject diagnostics, no
exception-carried owner or diagnostic is guaranteed. The traceback may retain
the caller/owner until discarded, but is not a recovery API or durable registry.
Dropping the remaining references can leak unresolved resources. No global
retention store, implicit retry or weakened cleanup obligation was introduced.

The appended fake-only tests assert exact exception identity through both real
initialization callpaths, rejection of attributes/notes, hostile status access
and representation, retained handles, no candidate/destructor retry, and explicit
same-thread recovery. Current tests, lint, format and runtime checks: NOT_RUN.
Historical Root 38-case/114-phase passes, Ruff check PASS/format FAIL, original
formal allowance failure and prior fees remain historical, not current results.
Five current registrations cover three source snapshots, evidence and delivery;
all are diagnostic/progress only. Root independently reviews/tests afterward.

## Cleanup lint and registration handoff — 2026-10-15

Current task: OWNED_DARWIN_CLEANUP_LINT_REGISTRATION_SOURCE_20261015_V1;
`history_inputs/TASK_PROMPT.txt`, supplied SHA256
`df1920042cd1599a397487291042cd5a53107b372200446f585af6a809ed2031`.
This appendix supersedes earlier current-task registration claims, not their
historical provenance. The prior final-cleanup attempt ended in allowance
failure (163.02548 seconds, 216334 inclusive input, 6249 output), with zero
registered Refs and delivery missing. Its three source files are historical
source provenance, not a prior five-Ref acceptance.

Only the terminal add_note try/except BaseException/pass in
_cleanup_after_failure is replaced by contextlib.suppress(BaseException), using
the existing import. The add_note body, all other helper nodes and callers are
unchanged. The same original exception object remains the caller's failure;
notes remain best-effort, never a release or retry guarantee. All 20 existing
test methods / 44 cases and imports remain byte-for-byte untouched in this task;
no new tests were added. The preceding document prefix is preserved.

Root independently reported the current 44 ordinary cases / 132 phases /
zero effects PASS before this lint edit. That independent result is retained
as historical evidence, not represented as execution of these revised bytes.
New tests, lint, formatting, AST verification and runtime checks are NOT_RUN
here. Only source text review was performed. Root may later format an
independent copy with AST-equivalence proof and verify the revised source.
No Bash, checker, standalone SDK test, production provider call, actual CGL,
ROS, Docker, physics, NN, hardware or camera execution was performed here;
actual work remains NOT_GO and requires separate authorization.

The first current write was reports/evidence.json before any registration.
This handoff selects exactly three current source snapshots (renderer, frozen
tests, document) as diagnostic_source, plus schema-bound evidence as
diagnostic_report and delivery as progress_report: five current registrations
are required, with actual returned IDs/hashes recorded, not inferred from paths.
Registration is source provenance only, not runtime acceptance or a new PASS.
