# Renderer CI capability contract (source-only change)

## Boundary and status

Only `tests/test_harnessbench_vision_oracle.py`,
`.github/workflows/data-flywheel-gate.yml`, and this document change. The product
MuJoCo backend is protected: no prefix, fallback, evidence validation or renderer
error behavior is weakened. This SOURCE phase runs static validation and pure
mock tests only: **model/Renderer constructors, mj_forward, STEP and NN = 0**.
Real backend capability remains unverified. The workflow and real tests below
are implemented for a subsequent, separately authorized renderer phase
(**RootGO**); writing them does not authorize or execute them here.

The source/software PASS statuses mean source integrity and mocked control-flow
checks, not real GL capability, image correctness, scientific success or physical
success. SOURCE must not execute the real matrix, required-constructor assertion,
or the observations fixture. `renderer_capability` selects the pure mock tests;
the real matrix and required backend test names intentionally do not match it.

## Decisions and exact evidence

`ROSCLAW_REQUIRED_RENDERER_BACKENDS` defaults to `osmesa`. CI declares it at job
scope. OSMesa must remain required; empty, unknown or OSMesa-removing declarations
fail. EGL may additionally be declared required, never silently substituted.

Each selected backend is probed in a fresh bounded subprocess with both platform
variables explicitly aligned. Linux CI uses `libEGL.so.1` and `libOSMesa.so.8`.
A native-loader check precedes the production imports and any model construction.
Only this **exact** `OSError` message is recognized as known native absence:

```
<selected SONAME>: cannot open shared object file: No such file or directory
```

The child emits `renderer_capability.v1`, selected backend, `UNAVAILABLE`, stage
`native_loader`, reason `SELECTED_NATIVE_LIBRARY_ABSENT`, SONAME, exception type
and exact diagnostic. The parent records the probe worker, production-prefix
SHA256, selected/inherited platforms, return code, stdout and stderr. The
classification is conservative and specific to this Linux loader contract:
missing dependencies, symbols, permissions, other platforms, malformed output,
unknown status and other failures do not establish optional absence.

| Fresh result | Required backend | Optional EGL |
| --- | --- | --- |
| Exact native absence record | FAIL: `REQUIRED_RENDERER_UNAVAILABLE` | typed `NOT_RUN` |
| Unknown/malformed/probe failure | FAIL | FAIL |
| Actual offscreen constructor closes successfully | Execute matrix | Execute matrix |
| Available matrix import/constructor/other worker defect | FAIL | FAIL |

The optional verdict is a structured record with type `renderer_capability.v1`,
verdict `NOT_RUN`, reason `OPTIONAL_SELECTED_NATIVE_LIBRARY_ABSENT`, and the full
capability evidence. Pytest reports it as a skip whose message is that JSON;
JUnit properties retain the same evidence. **NOT_RUN is not PASS**. Reporting
must show the actual executed/skipped matrix denominator; four optional EGL
NOT_RUN entries cannot be called eight passing combinations.

No package list, `find_library`, historical failure or successful import is
accepted as AVAILABLE. After the native library loads, the probe executes the
exact `_CAMERA_WORKER_CODE` prefix extracted up to `request = json.loads`, then
compiles a minimal scene, constructs a real 32x32 `mujoco.Renderer`, and closes
it. There is no catch converting import/constructor exceptions to absence.
Only successful offscreen construction establishes AVAILABLE for this invocation.
It proves construction/close, not rendered pixel quality. Probe caching is
backend-specific and limited to the current module's invocation, not a saved
external capability declaration.

## Renderer phase (not run in SOURCE)

The required-backend test independently checks fresh real offscreen construction;
CI cannot pass merely because `libosmesa6` installed. The preserved real matrix
covers **camera/render worker x selected EGL/OSMesa x inherited EGL/OSMesa = 8**.
For every available selection, each combination runs the **exact production
prefix**, asserts both environment variables equal the selected backend, compiles
the minimal model, constructs/closes a real Renderer, and requires the exact
`REAL_RENDERER_CLOSED_ZERO_STEP` marker. Opposite inherited platforms therefore
still exercise production alignment rather than a replacement/mock prefix.
These constructor tests have zero explicit engine stepping, mj_forward, NN and
image generation. They do not execute the production worker's request body.

The probe and matrix use separate child processes; neither changes parent
platform variables nor imports OpenGL into the parent. Timeout is 30 seconds
per child. `subprocess.run` kills and waits for its owned child on timeout.
Timeout, nonzero return, native signal/crash and missing success marker fail.
Return code is checked before parsing any purported absence record: a failure
cannot launder invalid scene/state or original-state evidence into NOT_RUN.

The existing
`test_renderer_worker_initial_state_mismatch_still_rejected_as_evidence_failure`
is preserved unchanged. Runtime evidence validation remains outside the optional
matrix skip policy. No broad `except Exception -> skip`, fallback to another
backend or pytest `continue-on-error` is introduced.

The workflow keeps existing dependency provisioning, explicitly requires OSMesa,
runs the required constructor assertion and real matrix, and always uploads the
JUnit evidence (legacy family includes record_property). Full regression is
retained without weakening its selector. No workflow, package installation or
real test was executed by this SOURCE task. Future CI runs are separate evidence.

## Pure verification implemented here

Pure tests supply mock subprocess responses, never native loads or constructors.
They exercise exact optional-absence evidence and nonexecution, required absent
or unknown OSMesa, required EGL, available import/constructor/render regressions,
nonzero evidence failure despite plausible absence output, original-state evidence
classification, nonmatching loader diagnostics, all eight prefix/environment
combinations with unchanged parent GL state, timeout/signal propagation, required
policy and rejection of package/native-presence-only availability. Passing these
fixtures proves the decision contract, not the mock's claimed backend capability.

## Historical evidence remains failed

Closed run 37757140799 has four EGL-selected errors before model/Renderer
construction: `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`.
That stack alone does not distinguish package absence, loader resolution or
vendor/platform support and does not prove Ubuntu lacks EGL. It is not used as
a current capability declaration. Such an import defect after a native library
loads remains a failure under this contract. Other historical CI failures are
not automatically attributed to this cause or retroactively changed to NOT_RUN.
PR630's cited run was cancelled, not a proved renderer failure. The saved local
constructor success does not establish hosted capability. No historical failed
run becomes passing as a consequence of this source-only change.
