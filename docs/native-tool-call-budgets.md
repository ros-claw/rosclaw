# Restrict native tool calls before execution

An operator can pass `toolCallBudget` to `createRosclawRuntime`, or in the PI
backend's create options. `allowedTools` names the admitted tools; optional
`maxCalls` and `maxTotalCalls` restrict admitted executions. Zero denies all
calls covered by that limit. Invalid budgets fail before runtime setup.

The installed public PI `tool_call` hook blocks excess and unlisted calls
before the tool body. Admitted calls consume the budget even when the body
fails. The frozen policy and counters survive ordinary new inputs and reloads
within that runtime. Parallel and nested calls pass through the same hook.
Existing tool, task, lease and daemon checks still apply.

## Exact command allowlist

`exactCommands` optionally maps an allowed tool name to a nonempty list of
unique nonempty command strings. A configured tool's `input.command` must
exactly equal one declared command — raw byte equality, with no parsing,
normalization, trimming, execution, path resolution, or shell sandbox claim.
Mismatches, nonstrings, and missing commands are rejected before the tool body
with `TOOL_CALL_BUDGET_EXACT_COMMAND_REJECTED` and consume no quota, the same
as `TOOL_CALL_BUDGET_TOOL_NOT_ALLOWED` and `TOOL_CALL_BUDGET_EXHAUSTED`.
Keys must be allowed tool names; invalid policies throw
`INVALID_TOOL_CALL_BUDGET*` before runtime setup. All admitted values are
deep-frozen copies, so caller mutation after admission cannot broaden policy.
`exactCommands` works independently of `visibleBudget` and changes neither
old counter semantics nor old hook return shapes when notices stay disabled.

## Model-visible budget snapshot (opt-in)

`visibleBudget: true` (boolean only, default `false`) appends one standalone
`ROSCLAW_TOOL_POLICY_JSON:<JSON>` marker line to the `before_agent_start`
system prompt, to `tool_result` text, and to blocked `tool_call` reasons. The
JSON snapshot is taken after current admissions and has exactly the keys
`allowedTools`, `exactCommands`, `maxCalls`, `maxTotalCalls`, `usedCalls`,
`usedTotal`, `remainingCalls`, `remainingTotal`. Unconfigured maps/totals are
`{}`/`null`; `used` counts are nonnegative integers covering every allowed
tool; `remaining` is `max(0, limit - used)` or `null` when unbounded. Original
system prompt and result text, `details`, and `isError` are preserved; a prior
own marker line is replaced rather than duplicated. Later prompts reflect
prior calls, but no cross-process counter persistence is claimed. Snapshot
maps keep every allowed tool name as an own key — including reserved names
such as `__proto__`, `constructor`, and `prototype` — and never mutate any
prototype. The provider stream receives the decorated policy as the
normalized leading `context.messages` system entry (there is no
`context.systemPrompt` in PI 1.0.4).

This is opt-in caller configuration, not a shell sandbox or a hardware permit.
It does not inspect command arguments, restrict user bash, limit model tokens,
or persist counts across a new process. The caller must preserve episode
identity and reject fresh-process resume when that would reset its budget.
No automatic CLI or backend resume policy is installed by this change.

The SDK integration test uses an explicit offline synthetic model stream and
harmless counter tools. It establishes pre-execution blocking and continued
counts across prompts, not Kimi quality or physical capability.
