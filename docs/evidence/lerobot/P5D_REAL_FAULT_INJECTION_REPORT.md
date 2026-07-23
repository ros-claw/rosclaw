# P5-D REAL Fault Injection Report (Exp 4)

- Body: `rh56_left_01` (Inspire RH56 left, FTDI FT232R adapter, Modbus slave 1)
- Pipeline: ActionGateway + SingleStepExecutor + SerialModbusTransport (REAL)
- Date: 2026-07-17
- Evidence: `/tmp/exp4_fault_injection.json`, `/tmp/exp4_fault_injection_s8.json`

## Results: 8/8 scenario expectations verified

| # | Fault | Expected (DoD §5) | Observed | Pass |
|---|---|---|---|---|
| S1 | Stale observation (2 s old) | command_not_sent | receipt BLOCKED `stale_action`, 0 commands, permit kept | ✅ |
| S2 | Calibration hash tampered | permit 拒绝 | `permit_hash_mismatch` + revoked; later submit `permit_revoked` | ✅ |
| S3 | Sandbox BLOCK (index target 100 < safe_min 138) | command_not_sent | `is_safe: false` (`calibration_safe_range`), 0 commands | ✅ |
| S4 | Slave no response (slave id 2 silent) | COMMUNICATION_LOST | receipt FAILED `communication_lost`, arming COMMUNICATION_LOST, permit revoked | ✅ |
| S5 | Policy worker restart | session/permit 失效 | `on_worker_restart()` revoked permits; later submit `permit_revoked` | ✅ |
| S6 | STATUS protection bit (0x04 simulated) | estop + failure event | verifier `status_protection` → step fault, arming FAULT, permit revoked | ✅ |
| S7 | Ctrl+C mid-level | DISARMED | SIGINT → emergency stop, `interrupted: true`, process exits | ✅ |
| S8 | USB unplug (sysfs unbind, STATIC pose) | 停止后续动作 + permit revoked | unbind → receipt FAILED `communication_lost`, arming COMMUNICATION_LOST, permit revoked; bind → rediscovery (re-enumerated ttyUSB2) → re-arm → noop COMPLETED | ✅ |

S1-S7 ran as one suite (`--skip-usb`); S8 ran separately with the sysfs
electrical disconnect (`--only s8 --usb-sysfs 5-1.4`, sudo-assisted).

## Real defects found by fault injection (fixed, P5-D)

1. **termios.error escape** — a physically vanished adapter raises
   `termios.error` (EIO), which is NOT an OSError subclass, so it escaped the
   transport as a raw `EXECUTOR_ERROR` instead of escalating
   COMMUNICATION_LOST + permit revocation.  Fixed in
   `SerialModbusTransport._io_guard` (converts OSError + termios.error;
   EIO-class errors also mark the transport disconnected).  Regression tests
   in `tests/unit/body/test_rh56_serial_modbus.py`.
2. **USB re-enumeration changes the device minor** — after bind, the adapter
   can come back as `/dev/ttyUSB2` instead of `ttyUSB1`; recovery must
   rediscover the adapter by sysfs id, never assume the node name.
3. **Fresh-rebind first-contact flake** — the adapter can time out on first
   transactions right after a re-bind; harnesses settle + retry.

## Hardware caveat (honest)

Repeated electrical unbind/bind cycling (~6+ cycles across the S8 iterations)
wedged the FTDI adapter firmware: `can't set config #1, error -110` →
`device descriptor read/64, error -110` — the adapter no longer answers USB
enumeration and needs a **physical power cycle** (unplug/replug by hand).
usbreset + authorized toggles do not recover it.  This matches prior CH340
experience; it is an adapter-hardware limitation, not a bridge defect.
