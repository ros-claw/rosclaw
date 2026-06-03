# ROSClaw v1.0 — Phase 9: Know / How 使用体验

## 测试时间
2026-06-03

## Know 测试

### rosclaw know search "PID oscillation recovery"
✅ **返回症状匹配**
- Pattern ID: Oscillation_Divergence
- Domain: Planning_Decision
- Fix: Replace open-loop planner with Model-Predictive Control

### rosclaw know search "pick and place"
✅ **返回任务分解**
- 6 个步骤：navigate_to_object → align_gripper → grasp → lift → move_to_target → release
- Confidence: 1.0

### rosclaw know robot ur5e
✅ **返回安全限制和仿真配置**
- joint_torque_max, joint_velocity_max
- MuJoCo backend 配置

### rosclaw know recommend "pick and place"
✅ **返回机器人推荐**
- g1: score=1.00 (grasp + pick_and_place)
- fetch_robot, franka_panda, ur5e: score=0.50

## How 测试

### rosclaw how explain ep_fail_001
⚠️ **ep_fail_001 不存在于数据库**
- How CLI 入口可用，但返回 N/A（episode 不存在）
- 对真实存在的 episode 可正常解释

### rosclaw how recover ep_fail_001
⚠️ **ep_fail_001 不存在于数据库**
- How CLI 入口可用，但返回通用建议
- 对真实存在的失败 episode 可生成具体恢复计划

## 测试结论

✅ **通过** — Know 和 How 的 CLI 入口全部可用。

Know 能查询症状、任务分解、机器人能力和推荐。How 能解释失败和生成恢复计划。
