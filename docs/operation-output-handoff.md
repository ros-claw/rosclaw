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
