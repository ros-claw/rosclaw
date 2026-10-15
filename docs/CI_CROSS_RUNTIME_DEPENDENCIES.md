# Full-regression cross-runtime prerequisites

## Exact job boundary

Only `.github/workflows/ci.yml` job `full-regression` receives three new prerequisite steps, immediately before the existing `Run full non-deployment suite` step. Existing job fields, timeout, original steps, pytest commands, selectors and assertions are unchanged; all other jobs remain untouched. No helper, test, runtime product, package version or lockfile is changed.

GitHub Actions jobs have separate workspaces/install state. The dependency installation in `node-agent-unit` does not supply `full-regression`. Python editable installation alone does not populate `packages/rosclaw-agent/node_modules` or build the agent CLI.

The added steps configure Node 22.19.0 (the agent's declared minimum and existing CI pin), use the agent lockfile as npm's cache key, execute `npm ci --include=dev` in `packages/rosclaw-agent`, then execute `npm run build` in that same directory. npm's cache is not a substitute for installation. Locked installation retains the package postinstall patch hook and includes TypeScript development dependencies; no ignore-scripts, unlocked install, version upgrade or fallback is introduced. The manifest and lock retain the pinned 1.1.0 dependency family; this is not default upgrade/adoption approval.

The existing build interface runs `tsc -p tsconfig.json`, asset copying and build stamping. NodeNext compilation with rootDir `.` and outDir `dist` supplies `dist/src/main.js`. Asset copying includes the prompt, digest-checked bundled skills and optional generated effects; existing fail-closed checks are not bypassed. Build stamping uses GITHUB_SHA in CI. These are future CI prerequisites, not commands executed during this authoring session.

## Genuine recorded continuation dependency

The unchanged `tests/agentd/test_recorded_continuation_target.py` imports the real SessionManager from `packages/rosclaw-agent/node_modules/@earendil-works/pi-coding-agent/dist/index.js` and invokes the actual built CLI entry at `packages/rosclaw-agent/dist/src/main.js`. Its first case creates two offline histories and a temporary SQLite mission fixture, then checks the selected identity/mode despite a newer corrupt file. The other cases cover empty history, outside-directory rejection and shared identity in CLI assembly. No skip, mock replacement or alternate selector was added.

The supplied PR682 public failure is before target/mode assertions: ERR_MODULE_NOT_FOUND for the SDK path (9248 passed, 1 failed, 120 skipped, 1299.20s). The initial Q51 snapshot had no terminal conclusion. The supplied completed baseline update subsequently records Q51 failure with the same nodeid and missing module (9220 passed, 1 failed, 120 skipped, 1287.10s). These are supplied historical observations, not local runs; unrelated ROS/macOS failures remain out of scope.

## Verification and authorization limits

Status: source authored; Root external scope audit and actual4/CI pending. Native host PI/model used for authoring; no additional standalone test/model/engine/ROS commands. This is not a claim of zero Native SDK/provider/AgentSession activity: the Native host PI/model performed authoring. No additional shell, install, build, YAML checker or offline test commands were run here. Registration schema validation is distinct from source-scope or CI validation.

Supplied prior authoring history is preserved separately: 265354 inclusive input, 2506 output, 79.77963390998775 seconds, 15REQ/15DONE, usage complete. These are historical costs, not refunds or final current-phase billing; Root measures current billing after closure. Original octet-only oracle CURRENT_CONTEXT FAIL remains retained; the MIME supplement is separate and does not rewrite that result. This independent fact-correction phase freezes the workflow and adds no CI functionality or default actor adoption. Additional model phase: purpose300/work270/cleanup30 LOW1M8192, request reserve100k/2k, retry0; old clocks, logs, source snapshots and fees remain preserved.

Root must independently check duplicate-key-rejecting YAML parsing, byte-exact outside-job boundaries, unchanged original job fields and ordered original steps, and prerequisite-only additions before the full suite. Root must separately authorize/run the four original offline cases and actual remote CI; this change is not proof that dependencies install, assets validate, builds succeed or the full suite passes. No test/SDK/remote-CI approval is implied by source-authoring authorization. The historical SOURCE_CONTRACT NO_GO remains preserved history, not a passing validation.

Allocation remains purpose300/work270/cleanup30, LOW, inclusive input 1M, output8192, automatic retry0. Historical failures and elapsed costs above remain recorded; no fresh execution cost or success is invented.
