# Cancellation covers owned sessions across process groups

Baseline: `cbbc8a030d8f5df99d4395d40944b495d328a9c8`.
This change concerns generic managed process lifecycle, not a physical controller.

The private native SENSOR01 receipt showed a confirmed cancellation of wrapper
PID 744689 with SID/PGID 744689 while GNU timeout PID 744692 and its Python child
744700 remained live. Both still had SID 744689, but GNU timeout had created
PGID 744692. The original `group_members` required both original PGID and SID,
so it omitted those members and reported the original group empty.
Original receipts and their terminal history remain unchanged:

`experiments/harness_audit_20261003/native_kimi_sensor_cohort_20261004_v1/SENSOR01/`

- `private_ledger_before_observer_v1.json`
- `private_operation_cleanup_v1.json`
- `same_owned_session_different_PGID_birth_evidence_v1.json`
- `exact_private_fallback_cleanup_addendum_v1.json`

Private real GNU timeout fixtures reproduced two different failure modes. A
recovered manager returned CANCELLED with a same-SID/different-PGID child alive;
the current manager waited on its original subprocess transport because that
child retained inherited stdout. Restoring the old PGID predicate while running
the bounded fixtures produced **2 failures in 3.50s**. Test cleanup used only
captured private identities, then awaited fixture subprocesses/drivers.

`session_members` now enumerates all live PGIDs within the already proved owned
SID. Captured members must still match SID, UID, boot and a birth tick no earlier
than the original leader. The captured member PID/start tick/PGID/SID/UID/boot
is checked again against a fresh capture after opening the pidfd, before any
signal. Individual kernel handles receive TERM/KILL; no numeric SID/PGID broadcast
is introduced. Cancellation checks the entire owned session before reporting
`process_stopped`, which now explicitly reports `scope: owned_session` and SID.
The compatibility helper name `group_members` delegates to this complete session
snapshot; the operation manager itself uses the clearer name.

Leader proof is unchanged: new adoption requires the persisted original leader
identity to match the currently live leader. A gone leader, legacy missing proof,
uninspectable member, changed birth or an uncaptured member at escalation cannot
authorize new signals; cancellation remains explicitly unresolved. In particular,
a reused numeric SID does not authorize adopting a new arbitrary session after
the original leader disappears. Captured processes that change group/birth
identity after the snapshot also fail closed. Zombies count as exited.

Coverage means **new PGIDs inside the managed owned SID**, as created by ordinary
GNU timeout. It does not adopt children that call `setsid()` to leave that SID,
prove an arbitrary descendant closure, or provide hostile-host isolation.
`--foreground` remains an operator workaround; the framework fix does not require
that workaround. ROS action cancellation/ACK behavior is unchanged. Existing
terminal records are not rewritten or retrospectively upgraded to new evidence.

Six new real-process fixtures cover current and recovered managers, timeout's
new PGID, a TERM-ignoring child requiring escalation, unrelated different-SID
process survival, and SID/UID/boot/earlier-birth inspection failures which keep
both actual private processes live and the ledger CANCELING.

Validation uses the original lifecycle/ROS-action/storage cohort, adding those
six cases. Exact cohort and outcomes are in the private durable review receipt;
stdout logs are `/tmp/rosclaw_same_session_cancel_red_20261004.log` and
`/tmp/rosclaw_same_session_cancel_cohort_20261004.log`. No main runtime, user
processes, hardware, native SENSOR01 files/database or physical fixtures were
modified during this work.
