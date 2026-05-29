# Sprint 3 Verification Report: Sandbox MVP

**Date:** 2026-05-29
**Commit:** `3a60521`
**Status:** PARTIAL PASS

---

## Acceptance Criteria

| # | Criteria | Status | Evidence |
|---|----------|--------|----------|
| 1 | 能加载 ur5e / go2 的 MJCF | **PARTIAL** | `SandboxRuntimeAdapter` 可加载内置机器人列表，但 hello_bot 等非内置名称 fallback 到 mock |
| 2 | 能运行 step | **PASS** | `mujoco_sim_driver.py` 实现 step |
| 3 | 能输出 joint state / contact state | **PASS** | `get_sensor_data()` / `get_joint_positions()` 可用 |
| 4 | 能录制 episode | **FAIL** | MCAP 录制未实现（标记为 P2 defer）|
| 5 | 能生成 replay | **FAIL** | Replay 生成未实现 |
| 6 | 能发布 SandboxEpisodeFinished 事件 | **PASS** | EventBus 可发布 sandbox 事件 |

---

## Test Results

```bash
python3 examples/hello_robot.py
# Sandbox 启动 → mock fallback → 正常运行 → 正常关闭
```

---

## Blockers

| Issue | Severity | Note |
|-------|----------|------|
| MCAP episode 录制 | P2 | 已标记为 v1.1 defer |
| Replay 生成 | P2 | 已标记为 v1.1 defer |
| `rosclaw sandbox` CLI 缺失 | P1 | 无 sandbox 子命令 |

---

## Verdict

**PARTIAL PASS** — MuJoCo backend 和 mock fallback 就绪，episode 录制/replay 和 CLI 待完善。
