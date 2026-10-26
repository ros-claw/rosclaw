# ROSClaw thin upstream patches — PI 1.4.0 candidate

Target: `@earendil-works/pi-coding-agent@1.4.0`; standalone TUI: `@earendil-works/pi-tui@1.4.0`.
Node requirement remains `>=22.19.0` in both ROSClaw manifests.
Official latest was observed as 1.4.0 in the supplied 2026-10-26 registry snapshot:
`https://registry.npmjs.org/@earendil-works/pi-coding-agent/1.4.0`.
Official npm artifact integrity is not a verified upstream source-commit association.

## Active mandatory patches

- `patch-01`: `formatResumeCommand` in `dist/modes/interactive/interactive-mode.js`.
  Implements the existing `AppIdentity` / `resumeCommandFormatter` contract named
  by `ROSCLAW-PATCH-01`: replace the exit resume hint with a saved-session notice
  (optionally its name) and product commands, not a new upstream API.
  Present `rosclaw continue` and `rosclaw sessions`, without internal session IDs or
  external `pi --session` / `--session-dir` hints. Resume stays through ROSClaw's
  kernel, SessionBinding, lease, ResourcePolicy and tool table.
- `patch-02`: guard `defaultEditor.onSubmit` before builtin command dispatch.
  `globalThis.__rosclawBuiltinPolicy.disabled` blocks policy-disabled commands,
  including the ROBOT profile's /trust, /share, /import and /reload semantics.
  A later input hook alone cannot enforce this pre-dispatch boundary.

## Retirement and continuation

The current patcher is unchanged: its mandatory anchors/replacements also occur
in the complete historical 1.4 reference. No application or idempotence checks
were executed by this author; Root's later results are recorded below.
Missing mandatory anchors still hard-fail.
Only the existing old-resume-hint cleanup and retired patches are optional.
Retired patch-03/04 cleanup preserves upstream reasoning storage/replay required
by provider continuation; do not reinstate global transcript stripping.
No historical forced-prompt/startRun patches or absent integration APIs are imported.

## Minimal current-main lock migration

Agent dependencies/overrides and both lock roots use exact 1.4.0. The seven PI
nodes (agent-core, ai, codemode, coding-agent, mcp, telemetry, tui) use 1.4.0
artifact integrities from the supplied official metadata/review reference and
exact internal PI edges. Both TUI nodes require `marked-cjk-friendly@0.1.2`;
both locks add its integrity and `marked >=15.0.0` peer (existing 18.0.11 satisfies it).
The official 1.4 coding-agent no longer declares chord: that edge is removed.
Pre-existing chord/esbuild lock records, nested resolved paths, AWS/Babel/dev
versions and all other unrelated records are retained; no broad donor-lock copy,
deduplication or historical product implementation is introduced.

## Author delivery is not qualification

At PI89377 delivery time, the original staged `pi-upstream.lock.json` recorded
version 1.4.0, an UNVERIFIED null source commit, and null `package_lock_sha256`
with `PENDING_ROOT_HASH_BINDING`: this minimal lock was not the whole hashed donor.
Read/edit-only author tools could not calculate it; Root had to compute and
separately review the final lock binding before installation/tests. That original
pinner remains byte-exact in this delivery's seven frozen seed files, not replaced
by Root's reviewed copy. The digest assertion must remain strict, never accept null.
No code, patcher, JSON parser, build, install or tests were executed by this author.

After that delivery, Root independently reviewed/materialized the eight files in
an isolated current-main review worktree (not the default runtime), calculated and
bound the exact final Agent lock SHA-256
`6d477a6c69bbb3b640e7f351aefab191888e23b9f6d8e1beb14c9135e4a020d4`
in its separate pinner, successfully ran npm ci for Agent/TUI, and actually applied
mandatory patch-01/02. The supplied book also records Root's isolated scratch
application/idempotence checks (exit 0 twice, complete target bytes unchanged on
repeat) and missing-mandatory01/02 refusals (exit 1 each). Source commit remains
UNVERIFIED; lock hashing does not establish it.
Root's complete Agent run reported **502 pass / 1 fail / 3 skip**; the real failure
was na-fix-red P0-3's README contract assertion, addressed by this documentation
repair. Root reported **27 pass** for TUI. These are independent Root results,
not this author's execution or full upgrade qualification. The prior failed Agent
log remains preserved in the supplied book; this correction has not been rerun.

Root still must rerun the corrected Agent suite and finish qualification, including
review of the entire installed lock dependency tree and retained no-longer-required
records, CLI73/UI51 focused regressions, package-entry/pin/runtime-boundary tests
and all AGENTS repository checks with WT/src mapping and offline isolated
ROSCLAW_HOME. Remaining checks are not established as passed here. Disclose owned
rg/fd fixtures and existing bwrap skips exactly; do not relax tests or unrelated
baseline Ruff findings.
Historical 790 Agent + 29 TUI + 19 Python and separate 504 Agent + 27 TUI + 73 Python
results are different old inventories, not this current-main qualification.
Commit, push, remote verification and controlled activation remain Root-owned/pending.

## Operational review

1.4 summary generation uses the model output limit and ignores summary reserveTokens;
reserveTokens still affects compaction timing. Do not infer historical opt-in model
admission features exist in this baseline. Current policy, allowed-tool, session
continuation, cancellation and local budget contracts must pass current tests.
Recovered tool-call/result cleanup, resumed tool activation, ChatGPT Business quota
retry changes and codemode queueing (potential waiting latency) need review.
No provider billing, output or latency bound is established. Isolated ordinary
ChatGPT 1.4 use proves that route only, not full current-main compatibility.
No live provider, SDK, credential access, hardware smoke or runtime replacement is
authorized here. Root alone may launch after review and a fresh exact slot check;
a prior model0/SDK0 observation is not that check and does not prohibit independent pureSDK0 work.
