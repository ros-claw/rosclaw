# Compact visible budget (opt-in)

`visibleBudgetMode` is an optional exact enum: `full` (default) or `compact`.
It affects result presentation only. `visibleBudget` defaults false; false plus
compact is legal and installs no prompt/result notice hooks. Both strict TS
parsers and Python reject null, numbers, booleans, arrays, objects, case changes
and surrounding whitespace before runtime/auth access.

When visible, every before_agent_start still appends the full effective policy
with `ROSCLAW_TOOL_POLICY_JSON:`. Default/explicit full retains the old snapshot
field ordering, bytes and reserved full-prefix line replacement semantics.
Compact results and denial reasons use `ROSCLAW_TOOL_BUDGET_COMPACT_JSON:` with
exactly policyDigest, usedCalls, usedTotal, remainingCalls, remainingTotal.

Digest canonicalization: take the effective full snapshot, remove the four
used/remaining fields, recursively serialize objects with keys sorted by JS
String sort (UTF-16 code units), JSON.stringify each key and scalar, preserve
array order, no spaces, then SHA256 of UTF-8 bytes, lowercase hex. Static fields
are allowedTools, exactCommands, maxCalls, maxTotalCalls (null if unbounded),
and, only when exactPaths is configured, canonical workspaceRoot and exactPaths.
Presentation flags are excluded. Paths are bound before hashing. The digest is
computed once from copied effective settings; mutable caller input and counts
cannot change it. This is a reference checksum, not authentication of tool text.

In compact mode all lines starting at column zero with either reserved marker
are removed and a genuine current notice is appended last. Ordinary text,
non-text content and structuredContent remain; SDK details and isError are
unchanged by the result patch. Forged reserved lines cannot consume budget,
change ACL or suppress the final canonical notice. Full mode only strips the
original full marker, exactly as before.

Admission remains synchronous and atomic, including parallel calls; denials
consume nothing. Exact path final-argument revalidation and symlink fail-closed
behavior are unchanged. This source-only change grants no robot, ROS, container,
physics or provider network authority. UTF-8 byte comparisons are fixture-local;
no token-cost, model quality or provider-success causal claim is made.
