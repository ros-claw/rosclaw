# ROSClaw v1.0 — Phase 4: 故意失败测试 — PID 振荡

## 测试时间
2026-06-03

## 任务目标
验证系统不是只能跑成功 demo，而是能处理失败、记录失败、解释失败、恢复失败。

## 执行方式

```bash
rosclaw demo mobile-pid --target 1.0 --kp 10.0 --ki 0.0 --kd 0.0
```

## 执行结果

| 指标 | 结果 |
|------|------|
| Kp | 10.0 (故意过高) |
| Ki | 0.0 |
| Kd | 0.0 |
| Steps | 9 |
| Final error | 0.0156m |
| Status | success |

**注意**: Mock 模式下没有真实物理振荡，任务仍然收敛成功。需要在真实 MuJoCo 物理仿真中测试才能看到振荡。

## How 恢复测试

### rosclaw how explain ep_fail_001
⚠️ **ep_fail_001 不存在**
- 返回 Failure: N/A, Root Cause: N/A
- How 模块 CLI 可用，但需要真实存在的失败 episode

### rosclaw how recover ep_fail_001
⚠️ **ep_fail_001 不存在**
- 返回通用恢复建议（Review episode logs and retry）
- 需要真实存在的失败 episode 才能生成具体 patch

### rosclaw memory explain
⚠️ **未找到 ep_fail_001**
- 但 artifact fallback 机制可用
- 其他真实 episode 可正常查询

## 测试结论

⚠️ **部分通过** — Mock 模式无法真实模拟 PID 振荡，但 How CLI 和 Memory CLI 都正常工作。

**建议**: 使用 MuJoCo backend 进行真实物理仿真以验证振荡场景。

**已验证通过**:
- ✅ How explain CLI 入口可用
- ✅ How recover CLI 入口可用
- ✅ Memory query 能找到 episode 记录
- ⚠️ 需要真实失败 episode 才能生成具体恢复建议
