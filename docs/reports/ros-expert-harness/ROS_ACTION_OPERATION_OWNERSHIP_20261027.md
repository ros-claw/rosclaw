# ROS Action Operation ownership and cancellation

Source component change against main `6f569d21b58d36bb9d5fe139cb3c0602db87880d`.
This report records protocol fixtures and source checks. It provides no new
hardware, installed-distribution, or HomeClean mission acceptance.

## Problem and behavior

A cancel send can fail, or its native result can remain absent while the goal
continues running. The previous grace handler recorded `CANCELLED` in either
case. Shutdown cancelled owned OS processes but omitted owned ROS goals, and
ROS admission could still send a goal after the supervisor had closed.

The existing OperationManager now owns ROS admission through the same lifecycle
lock and shielded start tracking as its process provider. Shutdown requests
cancellation of its own native goals and joins their bounded reconciliation.
A real native CANCELED result retains `CANCELLED`. An unconfirmed cancellation
records `LOST` with the stable goal ID, the native terminal status if received,
`cancel_confirmed=false`, `physical_stop_verified=null`, and
`reconciliation_required=true`. The existing five-second grace is unchanged.

A lost send reply retains an owned `DEGRADED` operation. It does not resend the
goal. Late native results remain audit events without replacing an already
committed primary terminal. Closed providers cannot write late callbacks into
an already closed journal. A returned `close()` is not physical-stop evidence;
an unconfirmed goal still requires reconciliation through its authorized owner.

This adds no Runtime, Session, Permit, device access, or motion authority. The
ROS Action provider is not itself a canonical rosclawd admission adapter. A
future coverage binding still needs canonical authorization, actual public
observations, and independently observed cancellation/parent-loss behavior.

## Validation

- Four regression assertions fail on the original main: missing cancel result,
  cancel transport error, shutdown ownership, and admission after close.
- The initial candidate passed 48 focused Operation/AgentService regressions, including lost send reply,
  caller disconnect during admission, late success, closed-journal callbacks,
  real OS process reap, and the actual AgentService shutdown wiring with a
  protocol peer. Further review found that the actual Ros2ActionClient discarded
  goal tracking on a failed send. Two additional regressions failed on that
  candidate; the client now retains its stable goal and listener for native
  reconciliation. Client/Operation integration uses a queue transport fixture,
  with no ROS connection. The final selected Operation/client suite passes
  47 tests, including both newly failing real-client cases; its file selection
  differs from the initial 48-test suite, so the counts are not additive.
- Practice: 183 passed, 9 optional tests skipped in an isolated ROSCLAW_HOME
  with the declared MCAP dependency installed in a local development venv.
- Compile checks pass. The repository's configured mypy command passes for
  121 files; both changed product modules pass their explicit type check.
- Changed-file lint and formatting pass. Full-repository Ruff has the same
  291 pre-existing findings as clean main, with zero added findings; the
  578 full-repository formatting findings also predate this change.

The current HomeClean H1 development run remains pinned to its existing image
and source manifest. None of that run's results qualify this separate candidate.
