# Finite native macOS CGL diagnostic (source only)

## Current evidence boundary

Status: **SOURCE_AUTHORED_NOT_RUNTIME_VALIDATED**. Pure tests **NOT_RUN**;
actual CGL **NOT_RUN**. No shell, checker, standalone SDK test, model, STEP,
physics, NN, rendering, ROS, container, or production-provider calls were run.
Native host authoring did occur; this is not a claim of zero authoring-provider
activity. All original production, test and CI files remain outside this change.
The new fakes do not replace or weaken the original camera oracle.

The retained remote result is **1 failed, 666 passed, 10 skipped, 4 deselected**
on `macos-26-arm64`, MuJoCo 3.13.0. The failure was
`SIM_RENDER_UNAVAILABLE: cgl: mujoco.cgl.cgl.CGLError: invalid pixel format`
in the strong public snapshot camera-binding test. The old message does not
identify Choose versus Create, a numeric RC, NULL pixel format, or npix.
Neither that failure nor the historical hosted-VM discussion proves global
absence of CGL support. Prior durability results are not changed or rerun here.

## Source basis and independent experiment choice

The copied official installed MuJoCo 3.13.0 `cgl/__init__.py` and `cgl/cgl.py`
are the authority for the default array, constants and existing ctypes
prototypes. Their supplied SHA256 values are pinned in the diagnostic. The
copied generated PyOpenGL raw GL1.0 source is the authority for `glGetString`:
`@_p.types(arrays.GLubyteArray, _cs.GLenum)`, and GL_VENDOR, GL_RENDERER,
GL_VERSION constants. The implementation uses that installed function, not a
new local C declaration. It does not enumerate renderers using unestablished
APIs. Source equality on the future host is checked before loading CGL; a
version/hash mismatch yields BOUNDARY_ERROR, not a guessed replacement ABI.

Primary rationale:
[Apple, Choosing Renderer and Buffer Attributes](https://developer.apple.com/library/archive/documentation/GraphicsImaging/Conceptual/OpenGL-MacProgGuide/opengl_pixelformats/opengl_pixelformats.html)
explains that restrictive attribute requirements can exclude renderers and
illustrates relaxing sampling. This is archived guidance, not a measurement of
the current runner. [MuJoCo issue 959](https://github.com/google-deepmind/mujoco/issues/959)
is historical context only. No Linux OSMesa workaround is silently applied.

All three arrays are immutable module declarations, before any attempt:

1. **mujoco_3_13_default** — exact official default, first unconditionally:
   `[99,4096,8,24,11,8,12,24,13,8,59,55,1,56,4,73,0]`.
2. **legacy_no_sampling_accelerated** — remove the multisampling flag and
   sample-buffer/sample-count pairs, retaining all sizes and acceleration:
   `[99,4096,8,24,11,8,12,24,13,8,73,0]`.
3. **legacy_no_sampling_unrestricted_acceleration** — additionally remove the
   acceleration requirement, keeping the same legacy profile and buffer sizes:
   `[99,4096,8,24,11,8,12,24,13,8,0]`.

These choices isolate sampling restrictions, then the acceleration restriction.
The third does NOT assert that a software renderer exists or will be selected.
There is no dynamic candidate expansion, implicit fallback, alternate profile,
renderer-ID probing, or production configuration change. All candidates are
independent diagnostics, including after an earlier success. The cap is three
(maximum allowed four), not a comprehensive search of CGL configurations.

## Interface, ownership and context policy

`probe(system=None, boundary_factory=None)` returns a JSON-compatible object.
The injected factory exists solely for pure tests. Production defaults to lazy
`_NativeBoundary`; module import and non-Darwin paths never import graphics
bindings. Non-Darwin returns typed `status=NOT_APPLICABLE`, zero attempts, not a
platform assertion or an emulated graphics result.

Production looks up distribution metadata without importing MuJoCo, hashes the
installed sources, then loads the pinned `cgl.py` directly under a private module
name. This avoids importing the MuJoCo parent package/model machinery. It uses
upstream `_CGLChoosePixelFormat`, `_CGLCreateContext`, `_CGLSetCurrentContext`,
`_CGLReleaseContext`, `_CGLReleasePixelFormat` with their unchanged prototypes.
Raw functions retain numeric RCs instead of checked wrappers discarding them.
The upstream unusual Create prototype is preserved: the output address is cast
to its declared void-pointer parameter; it is not replaced by a guessed header
signature. Release functions have upstream `restype=None`: their RC is UNKNOWN,
not invented zero. No global dependency monkeypatch or installed-file edit.

CGL current context is thread-local. Each invocation starts a **fresh**, non-pool
worker thread with initially NULL current context, and performs all native work
there. This deliberately avoids introducing an unestablished GetCurrentContext
prototype. The invoking thread's prior context is never read, overwritten or
released. The worker restores its prior NULL context before releasing its owned
context; it attempts that restore even after MakeCurrent error/exception. No
CGLLockContext is needed for an unshared context used by only its creating
worker. A future runtime review must validate this isolation design; pure fake
thread-local checks are not native proof of it. Do not repurpose `_attempts`
on a pre-existing graphics thread.

Each attempt owns only non-NULL handles returned in its own output slots. Slots
are captured even if an injected boundary raises after partial allocation.
Created contexts and pixel formats are released once, including after partial
Create failure; no destructors retry releases. Restore precedes context release,
which precedes pixel-format release. Cleanup records distinguish never needed,
attempted, returned, and exception outcomes. Cleanup errors stop further attempts.
If restoration fails, releasing a potentially current context is unsafe: retain
that context, label `RETAINED_UNSAFE_TO_RELEASE`, release the pixel-format owned
reference once, stop, and report failure. This explicit quarantine is not a
successful cleanup; end the diagnostic process and investigate it. The probe
does not promise leak-free recovery from a native detach/release failure.

GL strings are queried separately only after successful Create and MakeCurrent.
The loaded GL1.0 source hash and its platform library identity must match the
reviewed source and system framework path; otherwise query fields stay UNKNOWN
with errors. NULL strings are UNAVAILABLE/UNKNOWN. Query exceptions do not erase
other fields. Reads are capped at 512 bytes per string. There are no pixel reads,
buffers, camera binding checks or rendering operations.

## Output and future remote use (not performed here)

Future, separately authorized review may execute the pure file with pytest and
the standalone diagnostic with the host's Python:

```text
python -m pytest tests/sim/test_macos_cgl_probe_native.py -q
python tools/diagnostics/macos_cgl_probe_native.py
```

These are review instructions, not execution evidence or permission for this
phase. No installation, root remote job submission, or SDK/AgentSession test is
part of the source deliverable. The remote supervisor should impose a process
timeout, retain stdout JSON and exit status, and preserve the original failing
camera test unchanged for a separately authorized investigation. Native calls
are finite in number but cannot safely be interrupted by a Python thread timeout;
there is no claim of a hard wall-clock bound on a hung system driver.

Output includes all declared symbolic/numeric arrays; actual attempt rows;
Choose RC, npix, pixel-format NULL; Creation RC/context NULL; MakeCurrent RC;
per-string query outcomes; owned-handle flags; and individual cleanup outcomes.
`attempted_count`, `created_count`, `current_count`, and `succeeded_count` are
separate. Succeeded means all three strings were obtained and cleanup returned
without error, not that pixels or camera assertions passed. `OBSERVED` means
only that the finite collection ran; it is not a universal success verdict.
Boundary exceptions are type-only, without stack traces, addresses or secrets.
The program exits zero for NOT_APPLICABLE or at least one fully successful
attempt, otherwise one; consumers must also inspect cleanup and per-stage data.

Metadata records system/release/architecture/macOS/Python, installed package
versions, SHA256 of this probe, upstream sources, and requested system OpenGL
library. Loaded GL query-source/library identities are added if querying starts.
File hashing streams with a 128 MiB cap; unavailable or shared-cache-only library
files are explicitly UNKNOWN, never a fabricated binary hash. A requested path
and package version are not proof of dyld-resolved binary byte identity. Where
the framework cannot be hashed, a future authorized reviewer needs separate
platform-supported image identity evidence before claiming full library
provenance. No hostname, environment dump, device identifier, or raw pointer is
emitted. JSON cardinality is bounded by three attempts and fixed fields.

## Pure test scope and limitations

Twelve test functions author checks for exact first array; predeclared finite
candidates/no fallback; numeric Choose failure versus NULL/npix; partial Create
failure; MakeCurrent failure; per-field query failure; partial allocation before
exception; once-only cleanup after release exception; non-Darwin no graphics
imports/construction; invoking-thread preservation/order; unavailable UNKNOWN
strings/JSON size; and unsafe-detach quarantine. All inject a boundary; none
constructs production bindings or contacts a provider. They test control flow,
not the C ABI, Apple driver behavior, hosted runner availability, or camera
fidelity. They remain **NOT_RUN**. Source has only been reviewed as text here;
syntax, formatting, pure tests, remote library identity, actual context creation,
and the original camera failure all require later independent verification.

## Typed-RC and cleanup-exit source repair (current revision)

This append supersedes the earlier exit description: any report with
`status=STOPPED_CLEANUP_FAILURE` now exits one, even if an earlier independent
candidate succeeded. The JSON retains that earlier success count; it is not
rewritten to conceal the cleanup failure. NOT_APPLICABLE still exits zero;
otherwise the ordinary success-count exit rule is unchanged.

`_invoke` accepts only an exact built-in `int` as a numeric RC. Only integer
zero succeeds. Boolean values, floating-point zero/nonzero, NaN and infinities
produce FAIL with UNKNOWN RC, not numeric success. Exceptions retain ERROR,
UNKNOWN RC and the exception type. No native declarations, candidates, boundary
methods or ownership algorithm were changed.

Three unparametrized pure test functions were appended, bringing the authored
file to fifteen while retaining the original twelve and their context. They
exercise `_invoke` with bool/float values plus integer and exception controls,
and `main` with a fake-boundary report containing a first success followed by
a second-candidate release failure. Main-output JSON and ordinary exit controls
are also checked in the authored test. None constructs production bindings.

The copied `history_inputs/inputs/Root_original_pure12_receipt.json` records an
independent historical PASS for the original twelve, with no platform credit;
its original formal source-oracle FAIL is preserved. That receipt does not
validate this repair or its new tests. The original source rejection concerned
the external literal-domain guard; no historical result is promoted here.

Current status remains **SOURCE_AUTHORED_NOT_RUNTIME_VALIDATED**. All fifteen
current tests, syntax/format/checkers, actual Darwin ABI/CGL/GL, camera fidelity,
model/STEP/NN/rendering, ROS and containers remain **NOT_RUN** in this phase.
Only source text authoring/review occurred, using the native host authoring
model; this does not claim zero total SDK/provider activity. No shell, checker,
standalone SDK test or production-provider test was executed.

## Public string crashfix source revision (2026-10-15)

This append supersedes the earlier raw-string implementation and its claimed
512-byte read bound. Status: **SOURCE_AUTHORED_NOT_RUNTIME_VALIDATED**. The
current prompt is `history_inputs/TASK_PROMPT.txt`; this revision is source
creation only, not a repeat of either historical native attempt.

Both historical SIGSEGVs and their incurred costs remain retained evidence;
none is reset, reclassified as a Choose failure, or replaced with a success.
The supplied confirmed trace records default and no-sampling-accelerated Choose
RC 10002 / npix 0. The third, unrestricted-acceleration array obtained Choose
RC 0 / npix 1, Create RC 0 and SetCurrent RC 0 before GL_VENDOR manual `ptr[i]`
segfaulted. Its process exited -11 after 1.0707981250000103 seconds, with empty
stdout and no native report. The actual pointer value and first bad index
remain **UNKNOWN**. No unsupported numeric cost for the other crash is inferred.
The original source PASS and fifteen-test pure PASS are historical source/fake
checks, **not Mac availability evidence**. The strong camera oracle remains
unchanged; neither these checks nor the trace establishes pixel fidelity.

### Public API and provenance

`_NativeBoundary.query` now calls **OpenGL.GL.glGetString**. The copied installed
PyOpenGL 3.1.10 public chain is the basis: public `OpenGL/GL/__init__.py` imports
`glget`; `OpenGL/GL/glget.py` selects `_simple.glGetString` from public
`VERSION/GL_1_1.py` and assigns `ctypes.c_char_p`. The diagnostic does **not**
set restype, redeclare a prototype, or call the raw GL1.0 string function.
Before the first query call, actual loaded public initializer, glget, public
GL1.1, arraydatatype and raw GL1.0 files are hashed and checked. Public callable
alias identities and the existing requested framework identity are checked as
well. `metadata.query_sources` retains those observed hashes even on a source
mismatch; `metadata.query_api` identifies the public c_char_p route on success.
The four new public source pins are from the copied confirmed crash receipt,
not inferred from the raw GL1.0 pin. These checks are source provenance, not a
claim of authenticated process memory or complete dyld binary identity.

Only exact Python `bytes` or `None` are accepted from this call. None is
unavailable; other objects raise `TypeError` without truth probing, indexing,
casting, address logging, or pointer dereference. Bytes are sliced to at most
512 **before** NUL splitting and UTF-8 replacement decoding. Embedded NUL and
empty strings are handled without a manual C-pointer loop. The public
`c_char_p` conversion relies on OpenGL's **NUL-terminated C-string API contract**:
512 is only a Python input-processing limit, **not a hard 512-byte C read
bound**, and not a guarantee against every driver/native crash. Logged stage
records contain no string contents. The frozen report code still truncates
returned Python text to 512 characters; JSON escaping and replacement decoding
can expand serialized byte size. No new GL requests or retries are added.

### Flushed progress, not completion evidence

One `_cgl_stage_event` helper emits allowlisted JSON lines to stderr and flushes
each write. Its process-lifetime budget is at most 128 events, 1024 ASCII bytes
per event and 64 KiB total reserved output, including newlines. It never echoes
arbitrary objects, strings, exceptions, paths, handles, addresses, credentials
or environment dumps. Sink exceptions are suppressed so logging cannot change
the existing ownership/cleanup algorithm. A broken sink cannot guarantee log
persistence. Budget exhaustion silently stops logging; it never retries native
work. No file or native operation is introduced by this helper.

Standalone helper calls surround Choose/Create/MakeCurrent, each GL field,
detach and actual resource releases. Constructor/binding entry and successful
return are also marked. The query method marks its own entry and ordinary
return/error. A fatal native signal may leave only the last flushed **pre**;
missing post is not a fabricated return code. Python exceptions give typed
ERROR progress; release still records no invented numeric RC. An unsafe detach
still quarantines its context and performs no context-release call/event.
Original CGL method bodies, three arrays, handshake/cleanup statements and exit
semantics are preserved. Events are progress observations, never proof of
native success, full GL availability, rendering, camera fidelity or physics.

### New pure tests and next evidence boundary

Three tests are appended, for **eighteen** authored tests total; the original
fifteen tests/assertions remain intact. One executes the actual query method
bound to a fake boundary using fake public OpenGL modules and independently
supplied pinned metadata. It covers bytes, None, NUL/length/decoding cases,
poison-pointer rejection, source/library/alias gates and flushed stage ordering
through ordinary cleanup. Two further tests cover adversarial serialization,
all three output budgets, broken stderr, query-error post events, and detach
failure quarantine/order. No production `_NativeBoundary` constructor, native
load, real OpenGL/provider call, or model is used by these new tests.

The fix and all eighteen current pure tests are **NOT_RUN here**. No Bash,
checker, standalone SDK test, actual CGL, ROS, Docker/container, STEP, NN,
model physics or rendering was executed. Authoring used the native host/model;
this is explicitly **not** a zero-total-provider-activity claim. Independent
Root review must perform Ruff, then pure tests, then any separately authorized
third actual macOS run. Actual native availability and the original camera
failure remain unresolved until fresh independent evidence. Source registration
alone supplies no runtime/platform credit.
