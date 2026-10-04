# Preserving managed operations when agentd closes

`AgentService.close()` preserves managed process operations. It preflights
handoff, settles output observers and maintenance while SQLite is open, then
closes service resources. Concurrent closes join one cleanup task; cancelling
one caller does not cancel that cleanup. It does not cancel a task or authorize
signalling producers. Operation cancellation still uses persisted process birth
proof, owned-session membership, and confirmed-stop evidence.

New process operations write merged stdout/stderr to a private regular spool,
created with exclusive/no-follow flags and mode 0600. SQLite records its device
and inode. Observers validate ownership, permissions, identity, and the byte
cursor before reading bounded chunks. Each output event and its byte cursor /
partial UTF-8 decoder state share one SQLite transaction. Recovery drains saved
output before recording a terminal result; terminal cancellation remains
irreversible while unread output is recovered.

The launcher uses a process handle independent of asyncio transports, so event
loop shutdown does not close a PIPE or kill a producer. Its wrapper atomically
renames the exit-code file and itself exits with the child's actual code. An
inherited open-file-description lock delays final output flushing until all
inherited output descriptors close, including descriptors held by descendants
that outlive the wrapper. This preserves the inherited-output boundary, rather
than declaring EOF solely because the wrapper exits.

Legacy operations have empty spool metadata. An active local PIPE, unverified
legacy output, or active ROS Action cannot be transparently handed off.
`OperationHandoffUnresolvedError` identifies these operations before maintenance
or the store is closed. The caller must keep the service resources alive,
await their actual completion, or use an independently authorized cancellation
path; retrying close does not grant stop authority. Existing legacy PID recovery
remains limited to identity/exit status and does not invent lost output.

Output observation errors stay unresolved and cannot produce a success event.
For spool-backed operations, a later valid observer can resume from the last
committed checkpoint. Legacy PIPE failures have no such output replay contract. Spools
are retained for audit and consume disk space; no automatic rotation/deletion is
part of this contract. Disk exhaustion and power loss remain storage risks.
SQLite follows the store's durability settings, and the launcher does not fsync
each producer write. This is not a power-loss durability guarantee or an OS
sandbox. Producers deliberately unlocking their inherited descriptor or opening
independent writers violate the inherited-output completion contract.

A terminal operation with an inherited writer still open reports
`terminal_output_pending` during recovery and observes output in the background;
startup and a wait on the terminal status do not wait for log EOF. Closed writers
are drained synchronously. A confirmed cancellation proves the recorded owned
session stopped, not that a descendant in a different session stopped. No
additional signalling authority is inferred from its pending log descriptor.

Connected MCP stdio sessions use a dedicated lifecycle task for initialization
and cleanup of the SDK's task-bound AnyIO contexts. Shared service cleanup asks
that owner to exit and awaits it; it never exits a caller's cancel scope from a
different task. A cancelled close caller leaves this cleanup running. This is a
same-event-loop connection lifecycle contract, not persistent handoff of MCP
servers or confirmation that an interrupted tool effect stopped. Transport loss
after dispatch remains unconfirmed and is never automatically replayed.

MCP client shutdown waits for its same-loop call lock, then rejects new calls
with `MCP_CLIENT_CLOSED`; reconnect requires a new client. A session owned by
another loop causes `MCP_CLOSE_UNRESOLVED` before touching any owner event/task
or discarding its registry entry. Cross-loop/thread shutdown is unsupported.

Completed Action callbacks check the manager's close state before scheduling
and again when applying a queued update on the owning loop. After close they
cannot read or modify SQLite, terminal results or stop evidence. Callbacks whose
owning loop has closed are ignored; listener threads never use a synchronous
SQLite fallback. This protects mock/production callback scheduling boundaries,
not an implemented shutdown handshake for active DDS Actions: active Actions
still cause preserve-only service close to report unresolved.

If MCP initialization has not published a session, close supplies
`MCP_CLIENT_CLOSED` to its waiting callers and cancels only that same-loop owner's
initialization scope before acquiring the call lock. Level-triggered AnyIO scope
cancellation remains active through SDK context cleanup, so a non-responsive
stdio initializer cannot hold that lock waiting for a reply. An externally
cancelled caller still receives cancellation. Initialized/dispatched sessions
retain their call-lock and unconfirmed-outcome/no-replay rules. This owned SDK
connection cleanup does not signal independent durable process operations.
