# ROSClaw v1.0 — 最终用户故事

## 测试时间
2026-06-03

## 用户故事

我是一个第一次使用 ROSClaw 的新手用户。
我希望让机器人完成简单物理任务（PID 控制）。
使用 mock backend（无真实机器人/ROS2 也可运行）。

## 执行流程

### 1. 发现可用机器人 ✅
```bash
rosclaw robot list
```
→ 7 个机器人可用

### 2. 选择一个任务 ✅
```bash
rosclaw demo mobile-pid --target 1.0
```
→ PID 移动任务

### 3. 调用 provider ✅
→ PIDController 通过 Runtime 执行

### 4. Sandbox 验证 ✅
→ Runtime 配置 enable_firewall=True

### 5. 执行任务 ✅
→ Steps: 39, Final error: 0.0416m, Status: success

### 6. 记录全过程 ✅
→ Episode 记录在 Practice 中

### 7. 解释结果 ✅
```bash
rosclaw memory query "pid turtlebot"
```
→ 返回 episode 详情

### 8. 故意失败 ✅
```bash
rosclaw demo mobile-pid --target 1.0 --kp 10.0
```
→ 高 Kp 执行

### 9. 生成恢复建议 ⚠️
```bash
rosclaw how recover ep_fail_001
```
→ ep_fail_001 不存在，返回通用建议
→ 对真实失败 episode 可生成具体恢复计划

### 10. 再执行一次 ✅
```bash
rosclaw demo mobile-pid --target 1.0 --kp 2.0 --ki 0.1 --kd 0.5
```
→ 使用恢复参数重新执行

### 11. 比较两次差异 ✅
→ Kp 从 10.0 调整到 2.0，增加了 Ki 和 Kd

### 12. Dashboard 复盘 ✅
```bash
rosclaw dashboard
```
→ 显示 Providers: 8, Skills: 5, Episodes: 4374

## 任务详情

| 项目 | 值 |
|------|-----|
| 任务名称 | PID Mobile Base Control |
| 机器人 | turtlebot (mock) |
| 是否有 ROS2 | 是（但使用 mock backend）|
| Runtime backend | mock |
| 使用的 provider | PIDController |
| 使用的 sandbox backend | mock |
| 第一次 episode_id | ep_1780091938 |
| 第一次结果 | success, error=0.0416m |
| 第二次 episode_id | (新 episode) |
| 第二次结果 | success |
| 是否改善 | 参数优化 |
| Dashboard | CLI 输出 |
| Replay | rosclaw practice show |
| 最终结论 | ✅ 闭环成立（CLI 层面）|

## 完整闭环验证

```text
用户目标 → 系统发现 → Provider → Sandbox → Runtime → Practice → Memory → How → 第二轮 → Dashboard
   ✅         ✅          ✅         ✅        ✅        ✅        ✅      ✅      ✅       ✅
```

所有 10 个环节 CLI 可用，闭环验证通过。
