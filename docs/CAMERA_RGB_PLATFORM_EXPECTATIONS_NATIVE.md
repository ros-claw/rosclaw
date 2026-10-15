# RGB camera platform expectation — source-only revision 1

## Scope

For PR694, change only the `Assert.test` of the original backend assertion
(original line 59) in `test_camera_rgb_returns_artifact`, plus one unaliased
stdlib `sys` import. The assertion reads the returned
`value["renderer_backend"]`, not capability metadata, `gl_backend`, or an
environment declaration. This field identifies the actual successful renderer
subprocess candidate according to the supplied source contract.

On Darwin (`sys.platform == "darwin"`), only `cgl` is accepted. On every other
platform, only `egl` or `osmesa` is accepted. In particular, `cgl` is not accepted
unconditionally on Linux or other non-Darwin platforms.

All other assertions, including artifact identity, dimensions, dtype, camera,
intrinsics, extrinsics, simulation time, byte type and PNG magic, remain
unchanged. Fixtures, model inputs, five original test functions, skip behavior,
scenarios and channels are unchanged. No helper or mirror test is added.
Production, owned renderer/backend, oracle, strong-camera tests, selectors and
markers are outside this change and remain untouched.

## Historical evidence, not a new execution

The supplied `Root_actual_camera_receipt.json` and
`ROOT_ACTUAL_FAILURE_EXCERPT.json` record original Mac run 38754562951,
job 116489446851. The independent original strong-camera target passed setup,
call and teardown, returned pixel count 954 and oracle physics steps 0, with
the original profile restored. That target result does not make the whole job
successful.

The original Mac suite remains **FAIL**: 1 failed, 737 passed, 1 skipped,
4 deselected, 4 warnings, 280.54 seconds, exit status 1. Its RGB assertion
rejected actual `cgl` because it expected only `egl`/`osmesa`. The byte-type
and PNG-magic assertions after that failing assertion were **NOT_REACHED**
in that RGB test; their presence must not be presented as executed evidence.
Historical failures, costs and logs are not superseded by this source edit.

## Validation boundary

Status: **SOURCE_AUTHORED_NOT_RUNTIME_VALIDATED**. No Bash, checker, standalone
SDK test, provider invocation, rendering, actual CGL, ROS, Docker/container,
physics, neural-network or hardware execution was performed in this session.
Source inspection and artifact registration are not test execution, camera
validation, dynamics simulation, a green cross-platform job, physical success
or qualified video evidence.

Independent review must restore only the selected assertion predicate and
remove only the added import, then require exact equality of the remaining
module AST against the pinned seed. This inverse-AST check was not run here.
Later authorized validation must use the original cross-platform jobs and
original strong-camera tests; no functional PASS is claimed by this document.
