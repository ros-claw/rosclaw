# Output truncation in native chat

When PI finishes an assistant message with `stopReason=length`, ROSClaw reports
`MODEL_OUTPUT_LIMIT` and explains that the response is incomplete. This includes
responses containing only reasoning, partial text, or an incomplete tool call.
The harness adapter emits `turn.failed`, rather than `assistant.completed`.

The notice offers a smaller task or a changed output budget followed by another
message. It does not restart the task, send a continuation, retry a provider
request, mark a kernel task failed, or expose reasoning. Normal stop/toolUse,
provider errors and user cancellation retain their existing behavior.

This addresses a native Kimi experiment whose final response used 4000 output
tokens (3997 reasoning tokens), produced no source or delivery, and then became
idle. A closed, truncated response is different from a provider transport stall.
The notice does not assert that another budget guarantees delivery.
