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

## Exact file paths (opt-in)

CLI: `--tool-call-policy JSON_FILE`. The strict JSON object accepts only
`allowedTools`, `maxCalls`, `maxTotalCalls`, `exactCommands`, `visibleBudget`,
and `exactPaths`. Both the Python `agentd chat` entry and the TypeScript
entry validate this schema before home/store/auth/runtime or subprocess startup.
Schema validation is lexical only: relative and absolute file strings are valid,
without resolving them against the policy file's parent. Explicit `null`, arrays
or booleans instead of the map reject. Paths retain their bytes (no trimming);
blank means ECMAScript `String.trim` whitespace in both languages. Both slash
and backslash `..` components reject, as do NULs and trailing `/`. Filesystem
containment and canonical alias checks happen separately at runtime assembly
and again at the final execution seam, not in the Python schema parser.
Example:

```json
{
  "allowedTools": ["read", "write", "edit", "rosclaw_deliver"],
  "exactPaths": {
    "read": ["inputs/marker.txt", "reports/allowed.txt"],
    "write": ["reports/allowed.txt"],
    "edit": ["reports/allowed.txt"],
    "rosclaw_deliver": ["reports/allowed.txt"]
  },
  "maxTotalCalls": 5,
  "visibleBudget": true
}
```

`exactPaths` maps only those four supported tools (also in `allowedTools`) to
nonempty arrays of unique nonblank filepath strings. Empty maps are valid;
empty lists, nonstrings, unknown/disallowed keys, NULs, `..` components,
directory paths and duplicate canonical aliases reject. No glob, regex, prefix
or recursive directory authority is granted: strings name exact files only.
Unconfigured tools keep their old restrictions; absent `exactPaths` preserves
all legacy admission counts, reserved own keys and exact notice shapes.
Schema errors are `INVALID_TOOL_CALL_POLICY` at the early CLI parser, before
SDK/model/auth access; the programmatic factory reports `INVALID_TOOL_CALL_BUDGET*`.
Filesystem binding errors report `INVALID_TOOL_CALL_BUDGET_EXACT_PATH_BINDING`
at runtime assembly, before session/body setup.

Public factory API: `createToolCallBudgetExtension(policy, workspaceRoot)`.
An explicit absolute selected workspace root is required when paths are present;
legacy no-path policies can still omit it. Native supplies ActiveTaskContext's
workspace root, never the policy file's parent or an ambient/stale cwd. Relative
and matching absolute aliases bind the same canonical file. Outside roots,
sibling/prefix confusion, protected unlisted files and symlink ancestor/leaf
escapes reject with `TOOL_CALL_BUDGET_EXACT_PATH_REJECTED`, without consuming
quota. Validation walks existing ancestors, rejects dangling links, and resolves
nonexistent leaves against the nearest existing canonical parent without creating
anything. An allowed nonexistent read/edit/deliver still gets its honest tool IO
error and consumes admission; path policy does not promise file existence.

PI exposes `ExtensionContext.cwd`, but its `tool_call` input is mutable and
later handlers can patch it. The native configured tools therefore use
`wrapTools` on the same budget factory: the public hook prechecks, then the
actual execute seam snapshots final arguments, revalidates and replaces the
path with its bound canonical absolute target before task/file/artifact effects.
Only then is admission counted, synchronously and once. Configured SDK read
is explicitly wrapped too. Parallel/nested executions share these counters.
Hook-only embedding callers must use `wrapTools` for this final-input guarantee;
the ordinary public hook alone cannot secure later mutators. Other existing
workspace/task/ownership/daemon checks remain in force.

No path-policy dry-run flag is provided. Early parsing only validates schema;
model-selection validation is not a file-effect test. Bounded offline tests can
exercise the real compiled CLI against a synthetic provider and private HOME.
This is tool-layer admission, **not an OS sandbox**, Bash confinement or a
race-free concurrent-filesystem guarantee. Hostile simultaneous filesystem
replacement is outside this guarantee; no paid-provider/physics/hardware claim
is made.

## Model-visible budget snapshot (opt-in)

`visibleBudget: true` (boolean only, default `false`) appends one standalone
`ROSCLAW_TOOL_POLICY_JSON:<JSON>` marker line to the `before_agent_start`
system prompt, to `tool_result` text, and to blocked `tool_call` reasons. The
JSON snapshot is taken after current admissions. With paths configured it also
includes canonical `workspaceRoot` and `exactPaths` (only declared file authority,
no file contents or credentials). Without paths it has exactly the legacy keys
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
