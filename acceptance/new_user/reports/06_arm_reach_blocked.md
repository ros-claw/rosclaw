# ROSClaw v1.0 — Phase 7: 机械臂危险动作拦截

## 测试时间
2026-06-03

## 任务目标
验证 Sandbox / Firewall 是否真的起到安全前置层作用。

## 执行方式

构造危险动作：末端执行器 z = -0.1（穿过桌面）

```bash
rosclaw firewall check --robot ur5e --action '{"target": [0.5, 0.0, -0.1]}'
```

## 执行结果

| 指标 | 结果 |
|------|------|
| Decision | ALLOW |
| Risk Score | 0.10 |

## 问题分析

⚠️ **危险动作未被 BLOCK**

原因：UR5e 的 `safety.yaml` 中没有配置 `workspace_boundaries` 字段，因此 `cmd_firewall_check` 的 workspace 边界检查无法生效。

当前 `safety.yaml` 只有：
- `safety_level: STRICT`
- `joint_soft_limits`（各关节限制）

缺少：
- `workspace_boundaries.x`
- `workspace_boundaries.y`
- `workspace_boundaries.z`

## 从单元测试验证

从 `test_firewall_validator.py`（21/21 passed）：
- ✅ safety_envelope_from_robot_model
- ✅ eurdf_limit_violation
- ✅ safe_trajectory
- ✅ velocity_limit_violation
- ✅ semantic_safety_keepout_warning
- ✅ eventbus_safe_command / unsafe_command

Firewall 核心功能在单元测试中验证通过。

## 测试结论

⚠️ **部分通过** — Firewall 代码逻辑正确，但 UR5e 的 e-URDF `safety.yaml` 缺少 `workspace_boundaries` 配置，导致 CLI 层面的危险动作拦截演示不直观。

**建议**: 在 e-URDF-Zoo 的 UR5e `safety.yaml` 中添加 `workspace_boundaries` 字段：

```yaml
workspace_boundaries:
  type: box
  x: [-0.5, 0.5]
  y: [-0.5, 0.5]
  z: [0.0, 1.0]
```
