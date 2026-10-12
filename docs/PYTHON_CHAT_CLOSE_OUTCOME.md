# Python chat close outcome (SOURCE only)

The repair is confined to `_chat_pi`'s final outcome aggregation. Two fresh
locals retain the child result and explicit close failure. The terminal return
maps zero to 2 only when the existing close Exception handler ran. Clean
KeyboardInterrupt still yields 0. Every nonzero child result, including negative
signal results, is preserved exactly. No return is added to finally, so original
child exceptions propagate rather than being replaced with an outcome.

Existing shutdown statements, exception boundaries and diagnostics remain in
place: request server exit, wait for serve_done (10 seconds), submit close to
the original server loop and await its result (15 seconds), schedule loop stop,
join the thread (5 seconds), then terminate only the managed operator process.
A wait returning false alone is not an explicit close failure. Exceptions from
other cleanup boundaries retain their existing semantics.

## Bounded unit evidence

The new tests call the imported production `_chat_pi` function, not a copied
outcome function or reconstructed AST. Inert boundary doubles replace runtime
discovery, configuration writes, diagnostics routing, service/app construction,
uvicorn, loop creation, thread/event objects, filesystem probes and log opens,
subprocess call/Popen and runtime environment assembly. The thread target is
never run; no coroutine, real process, service, Node, provider or engine starts.
A future double raises TimeoutError immediately; no real timeout is waited out.
The short synthetic home and socket probe avoid the startup wait entirely.

The parametrized outcome family exercises child zero, positive 7, negative -9
and KeyboardInterrupt against clean close, RuntimeError and TimeoutError.
The propagation family exercises the identical original OSError and SystemExit
objects against those three close outcomes. Each invocation checks the complete
shutdown order and exact wait/result/join timeout arguments, including managed
termination. Diagnostics are checked for clean versus failing cleanup.
These are individually collected parametrized pytest IDs, not subtests; actual
collected IDs and results belong to the final operator RUN receipt and evidence
report, not a presumed test count. The four pre-existing refusal/metadata
selector families are unchanged.

CHECK proves only AST scope and syntax, not Ruff or functionality. RUN provides
bounded Python unit fault-injection evidence. The final CHECK and RUN must use
identical production, test and document snapshots; their hashes and genuine
receipt digests are recorded in reports/evidence.json.

## Explicit limits

Not run: full CLI/real PTY; actual AgentService.close handoff cancellation;
Node SDK lifecycle; real producer/process termination; ROS/DDS/engine/hardware.
Injected close errors demonstrate launcher aggregation, not actual producer
cancellation, real service handoff correctness or a full CLI lifecycle. No
simulation or physical-execution claim follows from this source-only proof.
