# Local operation monitor health

The existing two-second observation poll now maintains separate `monitor:task:` health widgets. Unresolved operations use `monitor:pending:` widgets until task identity is known. Progress widgets remain separate and unchanged. Health uses only setWidget: no transcript, provider, notification or model turn.

A failed events read leaves the event cursor untouched and shows `monitor unavailable`, locally measured monotonic seconds since the last successful observation, and `worker state unknown`. Before any successful read it says `no successful observation yet`. Exception text, endpoints, paths, credentials and RPC payloads are not rendered. Health content uses static categories rather than raw identifiers or errors.

A successful empty event batch shows `monitor connected/no new output` and refreshes observation time. This is status-channel freshness, NOT a heartbeat, progress measurement or evidence of process liveness. Ages are snapshots updated by polling, not a separate high-frequency timer. There are no additional health RPCs and at most one health publication per entity per completed poll.

Concurrent timer/direct ticks coalesce into one flight. Direct tick before start remains a supported test seam. Explicit stop clears introduced health widgets and invalidates pending observation publication; late success or rejection cannot resurrect health. Explicit ticks after stop may drain preexisting pending terminals through the unchanged ownership, revision, proof and idle gates, but cannot resolve operations or poll events. Restart permits new polling only after the old flight settles. Task health retires after tracked task/operation and pending terminal ownership are gone; terminal reminders retain their original proof, revision and idle-gating rules.

An optional monotonic millisecond clock dependency defaults to performance.now. Tests use the real watcher with inert RPC, sink and clock; no daemon, physics, inference, STEP or provider execution.

Session ownership, cancellation and provider watchdog policy are unchanged. Manual session-lock recovery remains manual: inspect canonical session ownership and owner liveness before following existing safe recovery procedures. This monitor never removes locks, signals foreign processes, restarts workers or infers that a disconnected worker has died. A future read-only session diagnostic journey requires separate scope and authority.
