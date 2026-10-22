# Optional UTF-8 admission for edit attempts

Status: proposed; isolated implementation, not adopted by existing author runtimes.

Two source-author episodes accepted edits larger than their operator contract,
then failed the operator's post-STOP check. A prompt and a post-hoc assertion do
not enforce a pre-effect restriction. The existing final `wrapTools().execute`
seam already owns immutable argument snapshots and synchronous call admission.
No provider transport or scientific source needs modification.

An opt-in policy field, `maxEditUtf8BytesExclusive`, requires a positive safe
integer and an allowed `edit` tool. For example, `24576` admits only
`Buffer.byteLength(oldText, "utf8") + Buffer.byteLength(newText, "utf8") < 24576`.
Both fields must be strings. The configured value is captured at construction,
reported in policy snapshots, and included in the existing policy digest. With
the field omitted, existing wrapping, notice shape and digest inputs are unchanged.

The wrapper snapshots final execution arguments, applies the existing scope and
call-budget checks, charges the admitted attempt exactly once, then checks bytes
before calling the tool body. An oversized attempt throws a tool error with a
stable code, measured byte counts and instructions to split the edit. It invokes
neither the underlying edit nor its file-content IO. The real offline PI runner
converts this into an error tool result and can continue with a smaller edit.
Rejections do not refund call allowances. Without final-execute wrapping, a
configured edit is rejected with `EDIT_EXECUTION_GUARD_REQUIRED`; an event hook
alone cannot protect against subsequent argument mutation.

Operator collectors must separately recognize this error as an acknowledged
no-write attempt, retain its call count and replay unchanged file bytes. The
existing post-hoc operator assertion must remain mandatory for successful edits;
it cannot also reject the newly permitted recoverable refusal. That collector
change and new runtime qualification are separate from this core patch. Existing
failed episodes remain failed. No active runtime, default policy, model budget,
workspace scope, robotics permission or physical acceptance changes here.

Validation: TypeScript compile; focused UTF-8 boundary, combined-field, mutation,
zero-effect, budget exhaustion, opt-in schema/digest and offline PI recovery
regressions, plus existing policy/path/CLI/literal-edit suites. The complete
package suite passed 790 tests with 3 existing skips (793 total), using the
original lifecycle fixture's supported private owned-home interface and exact
pinned rg/fd binaries. An earlier full run failed two bootstrap modules because
no rg exists in their system-only discovery paths; that result is retained.
Python Stage-B CLI policy tests passed 175 tests, including unchanged-byte
forwarding and the same positive safe-integer/edit-allowlist contract. The
commented external JSON-schema path is not a tracked schema at this baseline;
the actual Python and Node CLI validators are both updated. No live provider,
robotics SDK or model is called. Adoption awaits review.
