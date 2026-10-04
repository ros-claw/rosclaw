# Finite job completion versus service startup

Baseline: `d46bef8e75f3c65810e0381a4ed96a3d56257310`, PI 1.0.2.
This change adjusts the model-facing contract, not the automation protocol.

## Observed SENSOR01 sequence

The frozen native SENSOR01 session started its first managed operation at
04:04:46.009, then ended the assistant turn at 04:04:54.980, saying it would run
the selftest “after startup notification”. An operator neutral continuation
arrived only **0.934 seconds later**, at 04:04:55.914. The operation did not reach
its actual FAILED terminal state until 04:05:09.384. There was therefore no
terminal event to wake the model at that earlier boundary. This does not prove
an OperationWatcher failure or loss of an already completed result.

Evidence is preserved under
`experiments/harness_audit_20261003/native_kimi_sensor_cohort_20261004_v1/SENSOR01/`
in `persistent_sessions/2026-10-04T03-54-24-684Z_01a1050c-742c-7541-8809-d022f86d94a0.jsonl`
and `private_ledger_before_observer_v1.json`. No fixture or historical event was
modified for this review.

## Actual notification semantics

`process_start` registers its operation through the extension's finalized tool
event. The watcher polls task events every two seconds. Output/progress only
update TUI widgets; even stdout spelling `READY` does not start a model turn.
An unconsumed operation terminal result triggers a real new native agent turn
when idle, provided the owning task/revision remains active. Task-terminal and
old-revision results are archived without a new turn. The existing busy-pending
and finalized terminal-read ACK logic remains unchanged.

An offline fixture using the real public PI 1.0.2 `AgentSession`, private history,
an in-memory credential store and a synthetic stream verified this distinction:

- Initial prompt: one offline stream invocation, real agent start/end.
- RUNNING + readiness-looking output: widget updated; still one invocation, zero
  custom publications, session idle.
- SUCCEEDED: exactly one `rosclaw.operation.result` publication, a second real
  agent start/end, and the custom message persisted in session history.

No provider request, ROS/physical execution, NN, or live model/config mutation
was performed. This fixture proves notification mechanics, not that any model
will obey the new wording or that a service is actually ready.

## Narrow repair

The old English tool description unconditionally said `END YOUR TURN`; the
Python result summary repeated that instruction for steps depending on the
result. A long-lived service may require validation while still RUNNING, so
waiting for its final termination is the wrong dependency boundary.

Both surfaces now distinguish finite jobs from long-lived services. Finite jobs
still wait for termination via the existing notification. Long-lived services
have **no automatic startup/ready notification**: admission/RUNNING/stdout alone
do not prove readiness. Use existing bounded `process_status`/`process_output`
and actual service observations under the original task deadline, then perform
dependent startup/test steps in the current turn. Do not sleep or poll without a
bound. Independent steps must not mutate the running operation's inputs.

The native system prompt was inspected; it contains no duplicated unconditional
process-start/end-turn instruction to update. The public tool still accepts only
its existing `command` field. No ready flag, TTL field, stdout heuristic, new
event, task-finish rule or physical permission is invented. RUNNING means the
task is not complete; it does not mandate endless automatic model turns.

## Resume boundary

Operation notification registration, consumed proofs and pending terminal maps
are currently watcher-instance memory. The previous stop/start fixture retains
the **same** watcher; it is not a cross-process replay guarantee. A fresh watcher
without an explicit registration emits no notification in the private fixture.
Session resume restores binding, leases and kernel reconciliation, but does not
automatically reconstruct these notification subscriptions from all historical
operations or advance an unfinished physical phase. This limitation is recorded,
not patched by automatically taking over old revisions or replaying historical
terminal operations. Any future recovery design needs explicit session-scoped,
revision-bound subscriptions and consumption evidence.

## Validation

Private package build passed. The real public SDK fixture plus the existing
notification-consumption, terminal/revision, idle-gate and widget cohort passed:
**30 tests**. Python process/tool bridge regression: **11 passed**. Ruff and
diff checks pass. No main runtime reload or test of the current physical case
occurred. The effect on model behavior still needs a separately supervised native
trial; these checks do not claim the original SENSOR01 task was repaired.
