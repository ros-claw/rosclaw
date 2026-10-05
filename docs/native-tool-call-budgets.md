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

This is opt-in caller configuration, not a shell sandbox or a hardware permit.
It does not inspect command arguments, restrict user bash, limit model tokens,
or persist counts across a new process. The caller must preserve episode
identity and reject fresh-process resume when that would reset its budget.
No automatic CLI or backend resume policy is installed by this change.

The SDK integration test uses an explicit offline synthetic model stream and
harmless counter tools. It establishes pre-execution blocking and continued
counts across prompts, not Kimi quality or physical capability.
