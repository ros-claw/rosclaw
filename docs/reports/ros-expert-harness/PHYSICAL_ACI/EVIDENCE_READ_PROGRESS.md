# Bounded evidence access progress

The opt-in typed runner now keeps a bounded record of successful original page
reads and serialized-record slices. Earlier independent option-interface pilot
traces show repeated reads from offset 0 after switching tools. The existing
workbook kept measured page records and one recent tool result, but did not retain
expansion progress or earlier slice text. The frozen experiments remain unchanged.

The workbook exposes each returned page's original observation hash, view,
requested cursor, next cursor and repeat count. Expanded records are keyed by
both evidence reference and serialized record SHA256. Coverage is the union of
actually returned character ranges; `next_unread_offset` points to the first
missing character, including holes. Partial JSON/URDF text stays an explicitly
labelled serialized slice, never a parsed complete fact. These records persist
across intervening tools. A returned full serialized record is still historical
input, not current robot state or physical admission.

Progress entries and original slices each have a 2,000-character projection cap,
with whole-entry omission counts. The existing 6,000-character measured-fact cap
and hard 48,000-byte prompt limit remain. The helper retains at most 32 progress
entries and 16 slices; the current protocol permits only 12 model/tool turns, so
its count metrics cover every interaction within an episode. This is current
state metadata rather than replaying the full transcript. Full original tool
responses and model events remain separate immutable artifacts.

All arms receive the same access mechanism. The instructions distinguish the
host's original-input/diff compilation from the model's conditional choice and
the SDK's fresh physical proof. They supply no fault-specific answer, mandate no
repair, and preserve abstention. Selection alone still starts no World. No model
is allowed to infer present L1 or motion authority from reading historical data.
No SDK, Runtime, controller, admission, lease, source or stop gate changes.

Unit and recorded-event replays validate progress, overlap and omission semantics.
Synthetic model clients are protocol tests only. This change has no measured model
efficiency result until a separate prospective study is registered and completed;
results of previous pilots must not be retroactively recomputed with this runner.
