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

## 动力学与策略合同

- 编译后按名称解析 joint/body/geom/actuator，检查所有 ID >= 0；
  attach 前缀会改变实际名称。策略的关节顺序、obs/action 维度、
  增益、单位、初始化姿态必须与部署合同对照，不靠数组位置猜。
- MuJoCo freejoint：qpos 的姿态是 wxyz；用 jnt_qposadr /
  jnt_dofadr 定位，平移 qvel 是世界系、角速度 qvel 是局部系。
  不要二次旋转局部角速度。原生 observe 的 body_pose 为 wxyz、
  site_pose 为 xyzw、body_velocity 为 raw cvel（角/线顺序），
  不可直接当 freejoint qvel 或局部线速度；先读 sim observe --help。
- data.ctrl 是控制请求；实际力用 actuator_force / qfrc_actuator，
  按 actuator / dof 映射。保持官方策略文件 hash；外层反馈另记。
- 接触前速度在 mj_step 前取，接触后速度在 episode 末取。
  保存逐步接触、力和时间；积分时区分法向标量冲量与有方向的
  世界系冲量。mj_contactForce 返回接触局部系 wrench，需用
  contact.frame 转换，并依据 geom1/geom2 确定对目标物体的力向。
  用实际几何薄轴确定拍面法向，不默认 local Z。先用独立碰撞
  校验动量变化与全部接触力及重力积分；恢复系数不从已经
  碰撞过的“入射速度”估计。参考官方 API：
  https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce
- 拍面/工具坐标不是机器人原点：目标需扣实际安装偏移。先用
  小脉冲实测速度/转向符号，再加反馈。走路转停止时在当前位置
  重捕获 hold 参考点，不能把机器人拉回原来的站立起点。
- 分开起立、稳定和正式验收窗口；漂移取同一窗口起末的平面
  位置，停止检查完整平面速度并披露周期摆动，不能只验 vx。
  无样本、缺 seed、短轨迹、坏维度、NaN 或缺证据均不得 PASS。

## 实验与视频

- 先保存物理遥测，再离线渲染；长运行分段 checkpoint。每次
  协议/参数变更使用唯一 run 目录，保存失败、源码及依赖、模型、
  策略的完整 SHA-256；已登记交付物不再覆写。
- 初始化/发球事件明确记录；飞行中写球速度、途中重置机器人
  不可冒充击球或运动能力。独立试验用 fresh state 和完整速度
  初值；选择协议与验证用不同 seed，并披露筛选条件和失败分母。
- 初始化参数不能代替实际状态：全部 setup 与 mj_forward 后按名
  读取根坐标，检查是否被另一层默认值覆盖。每回合有界回位后
  验实际位置、速度和姿态，未达到条件记 INITIALIZATION_FAIL；
  回位阶段需明确停放上一回合的球，防止残留活球触发拦截。
- 轨迹至少保存全模型 qpos/qvel/ctrl 与实际力、维度、时间；
  分段使用唯一编号，manifest 列出每段 hash/帧数/时间范围，
  不用反复覆写的 partial 文件冒充完整轨迹。缺拍关节或球状态
  的遥测不能还原为该次实验的完整动作视频。
- 先验证来球物理可达性，再评价拦截；弹跳按 contact episode
  计数，不按接触帧数。过网验球心高度、球半径和真实触网事件。
  没有可达样本就报告失败，不自动挑零成功率家族当成功协议。
- 同一拍反复接触不是双方回合：交替序列需核对真实拍的归属、
  有效冲量、事件时间、弹跳/触网及期间无重置。策略比较先验
  两种设置实际执行不同的控制逻辑，再核对位移与控制请求；
  设置了一个未被 controller 读取的字段不证明策略生效。
- 视频只读冻结轨迹；标签按实际协议时间及按名解析的根坐标
  生成，给可读近景/接触慢放。预览成功不等于物理验收成功。
- 优先通用仿真工具及 --help。缺少在线策略/多机器人原生能力
  时如实声明；沙箱 worker 实验不能冒充原生闭环执行回执。
  Bash 搜索用有范围的 rg，网络/探索命令设显式 timeout；后台
  结果自动通知，独立工作可继续，但不得修改后台操作的输入。

## 安全分层

- SIM：物理原语工具 + 任务沙箱代码自动执行。
- 真机动作：必须走 rosclaw_request_action 的 admission 链
  （rosclawd + permit + operator），任何其他路径都不是执行权威。
- 改产品核心源码不是任务能力——走开发流程（克隆仓库+PR）。
- 同一调用同一参数失败后不机械重试；先读结构化诊断。

## 权威资产

- 优先 e-URDF-Zoo 等项目已登记的权威资产。任务需要新增模型时，
  可在 SIM 任务沙箱调查官方厂商资产，固定版本及完整 hash，
  标为实验性；局部验证不提升原生支持状态或真机能力。测试
  fixture 不可冒充交付物的物理证据。不确定来源时先调查，
  不从 / 全盘搜索。
