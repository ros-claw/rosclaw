---
name: rosclaw-embodied
description: ROSClaw 具身任务纪律——通用物理原语编排、任务沙箱编码、证据验收、安全分层（何时用：任何涉及机器人/仿真/动作的任务）
---

# ROSClaw 具身任务纪律

Pi 是唯一大脑：理解用户目标、调查环境、组合通用原语、没有现成
能力时在任务沙箱写代码、运行观察修错、判断是否完成用户目标。

## 通用物理原语（SIM 自动执行，不弹批准）

- `trajectory_generate_planar_path`：任意路径点 waypoints
  `[{x,y,z,contact?}]`（contact=false=抬笔移动段）——文字/任意
  图形都由你表达为点列；shape=star5|circle 只是便捷参数，
  **没有默认形状**。返回 plan_id 句柄。
- `ur5e_simulate_cartesian_trajectory(plan_id)` → trace_id
  （真实 MuJoCo 动力学 rollout + 跟踪指标）。
- `simulation_verify_tracking(trace_id, max_tracking_error_m)` →
  诚实 PASS/FAIL + 误差数值。
- `simulation_render_trace(trace_id, format=gif)` /
  `simulation_render_scene(trace_id, camera=...)` → 交付物。
- 观测：`ur5e.get_end_effector_pose` 等 OBSERVE 面。
- 可信上下文每轮带 workspace_window（安全半径/z 窗口）——
  摆 waypoints 不要越界（规划器硬校验，越界即拒）。

## 任务沙箱编码

- 活跃任务的 `scratch/` 区可写可运行（write/edit/bash cwd 限定在
  session 工作区与任务 scratch 内）——写几十行 Python 把字母/
  图形转成点列是正常能力，不需要内核替你认识任何形状。
- scratch 是草稿——交付物登记走 outputs/（register_artifact）。

## 证据与验收

- 交付物必须登记（rosclaw_artifact_register）——口头提到不算。
- 仿真证据标 simulated——动力学自洽不证明真机效果。
- 用户否定结果后修正或重开 revision——不得拿旧结果冒充。

## 安全分层

- SIM：物理原语工具 + 任务沙箱代码自动执行。
- 真机动作：必须走 rosclaw_request_action 的 admission 链
  （rosclawd + permit + operator），任何其他路径都不是执行权威。
- 改产品核心源码不是任务能力——走开发流程（克隆仓库+PR）。
- 同一调用同一参数失败后不机械重试；先读结构化诊断。

## 权威资产

- 机器人模型只认 e-URDF-Zoo 权威资产；测试 fixture 绝不出现在
  交付证据里。不确定资产来源时先调查，不从 / 全盘搜索。
