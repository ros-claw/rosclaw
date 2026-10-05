# Native episode private process cleanup

`await cleanup_owned_episode(actual_home, declared_home=sealed_spec_home)` runs
on the caller's current event loop. It imports the production `TaskKernel` from
`rosclaw.task_kernel.service`, joins manager close before closing its connection,
commits cancellation and checks terminal state on a fresh read-only connection.
Pair it with `episode_contract` metadata admission before running an episode.

The two HOME paths must resolve identically; the user's main `.rosclaw` and any
additional protected HOME are rejected before opening SQLite. The caller must
have created/authorized that fresh private HOME. Path equality is not an OS
ownership sandbox. No database or schema is created by cleanup.

Only process-provider operations may be canceled. Existing terminal rows remain
terminal; absent identity or only a gone leader yields
`DURABLE_TERMINAL_PHYSICAL_NOT_PROVEN`, not a whole-process-tree stop. Zero
operations yields `NO_OPERATIONS`. A legacy/changed process identity leaves pending `CANCELING` and
`STOP_UNCONFIRMED`; unknown PIDs are not killed. Active ROS action providers are
unconfirmed and not canceled. The ledger path must resolve inside the declared HOME, outside protected HOME;
symlinks into a main database are rejected before connection. This path check is
not a defense against a malicious concurrent host replacing paths.

Live captured birth members, commit failure, close
failure, or budget overrun prevent confirmed cleanup. Actual process stop and
persisted ledger state remain separate evidence.

The budget is cooperative: manager close is joined even when settlement exceeds
it, and elapsed overruns are reported. It does not guarantee an OS kill deadline,
all arbitrary detached descendants, in-flight ROS callbacks, or hardware stop.
Receipt publication and a parent's actual-exit budget adjudication remain caller
responsibilities. Caller cancellation still propagates after close settlement.

This helper prevents the hand-copied cleanup import/transaction mistakes exposed
by the four native Kimi software episodes. Their original operator-invalid
status and frozen evidence are unchanged; it does not rerun or regrade them.
