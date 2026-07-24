# RH56 右手故障注入报告（v1.0.1 清单 §4 Exp 8）

- 日期： 2026-07-23
- 身体： `rh56_right_01`（/dev/ttyUSB0，FTDI 5-1.2）
- 执行器： `scripts/experiments/exp4_fault_injection.py`
  （`--transport-profile configs/rh56_right_rs485_v1.yaml --calibration configs/rh56_right_01_calibration.yaml --body-id rh56_right_01 --usb-sysfs 5-1.2`）
- 全程静止/开放位姿下进行；FTDI 当日 unbind/bind 共 4 个循环（远低于 ~6 循环 wedge 阈值）

## 场景结果（8/8 PASS）

| 场景 | 预期 | 实测 | 证据 |
|---|---|---|---|
| S1 过期 observation | stale_action，command_not_sent，permit 保留 | BLOCKED `stale_action`，hw=0 | `/tmp/exp4_fault_injection.json` |
| S2 改 calibration hash | permit_hash_mismatch + permit revoked，后续拒绝 | BLOCKED + revoked，followup `permit_revoked` | 同上 |
| S3 Sandbox BLOCK | is_safe False（step_delta 799>50），command_not_sent | sandbox 拒绝，hw=0 | 同上 |
| S4 slave 无响应 | COMMUNICATION_LOST + permit revoked | FAILED `communication_lost`，arming=COMMUNICATION_LOST，revoked | 同上 |
| S5 杀 policy worker | permit 失效（worker_restart） | revoked 1，后续 BLOCKED `permit_revoked` | 同上 |
| S6 STATUS protection（模拟位） | verifier fault(estop) + failure event + permit revoked | FAILED `execution_fault`，arming=FAULT，revoked | 同上 |
| S7 Ctrl+C | DISARMED + interrupted 摘要 + 进程退出 | interrupted=true，estop_seen=true | `/tmp/exp4_fault_injection_s7.json` |
| S8 拔 USB（sysfs 电气） | 停止后续动作 + permit revoked + COMMUNICATION_LOST + 恢复 | unbind→`communication_lost`+revoked；bind→恢复并完成 noop | `/tmp/exp4_fault_injection_s8.json` |

## 过程中发现并修复的问题（已纳入 harness）

1. **S7 子进程未转发右手参数**（v1.0.0 遗留）：`s7_ctrl_c()` 以左手默认配置拉起
   exp3 子进程，导致右手验收时中断摘要缺失。已修复为转发
   `--transport-profile/--calibration/--body-id/--hand`，并新增 `--only s7` 与失败时
   `output_tail` 诊断。修复后右手 S7 PASS（returncode 1，interrupted，estop）。
2. **S8 sudo 票据**：无 TTY 环境下 `sudo -n` 票据按调用方 PPID 隔离，shell 预缓存对
   harness 子进程无效；改用 harness 设计的 `EXP4_SUDO_PASS`（stdin 喂给 sudo -S，
   不落任何文件）。
3. **S8 前置位姿**：S8 的基线 noop 命令开放位姿 `[1000]*6`，permit 步长上限 50 raw，
   手部若处于中间位姿（本例由 S7 中断遗留 index 899/thumb 932/thumb_rot 759）则基线
   必然 `step_delta_exceeded`。以 10-raw 块、speed/force 400 走回开放位姿后 S8 PASS。
4. **适配器重编号**：bind 后适配器可能枚举为 `/dev/ttyUSB2`（旧节点未完全释放），
   harness 的 `_find_adapter_node` 已按 sysfs id 正确重定位；验收后用一次额外的
   unbind/bind 循环恢复 `/dev/ttyUSB0`（当日循环数仍 ≤4）。

## 备注

- S6 的保护位为模拟注入（STATUS_PROTECTION_MASK 位仿真），真机物理保护触发
  （300 g 硬限）由内核 executor 路径的既有 fault 试验覆盖（左手 v1.0.0 +
  rh56-kernel-executor-force-cap 记录）。
- 每个场景独立 fresh harness（permit/transport 隔离），与 v1.0.0 左手规程一致。
