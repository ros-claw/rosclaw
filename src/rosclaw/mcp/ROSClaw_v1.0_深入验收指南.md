# ROSClaw v1.0 深入验收指南

## 一句话原则

**不要问“功能都写完了吗”，要问：一个真实用户接入 Claude Code 后，能不能用 ROSClaw 完成一个物理智能任务，并且全过程可观测、可回放、可记忆、可恢复、可迭代。**

ROSClaw v1.0 的验收不应该停留在：

```text
模块能 import
接口能返回
demo 能跑
README 写得好
```

而应该验证：

```text
用户能不能从零启动系统？
Claude Code 能不能通过 MCP / Agent Runtime 调用 ROSClaw？
Provider 能不能被路由？
Sandbox 能不能预演和拦截？
Practice 能不能记录全过程？
Memory 能不能回答“刚才发生了什么”？
How 能不能给出恢复策略？
Forge 能不能生成新能力？
Dashboard 能不能看清整条链路？
失败后能不能形成下一轮改进？
```

附件中也强调，`rosclaw-sandbox` 不只是 firewall，而是物理试验场与安全执行前置层；`provider` 是具身能力总线；`sdk_to_mcp` 应升级为 Asset Forge，并提供 CLI、Skill、MCP Tool 等入口。

---

# 一、给主管的直接表述

下面这段可以直接发给他们主管。

我建议 ROSClaw v1.0 不要只按“开发完成”验收，而要进入一次完整的用户视角深度验收。

原因是这个项目不是普通 Web 系统，也不是单一机器人 SDK，而是一个 Physical AI Infrastructure Runtime。它包含 Runtime、Sandbox、Provider、Practice、Memory、How、Forge、Dashboard、e-URDF、MCP 等多个模块。单独模块能跑，不代表整体系统成立。

这次验收建议用 Claude Code 作为真实用户入口，通过 ROSClaw MCP / Agent Runtime 接入系统，让 Claude Code 真实调用各模块完成任务，例如小车 PID 运动控制、机械臂仿真抓取、Unitree / G1 行走仿真、巡检任务、失败恢复、记忆查询、SDK-to-MCP 能力生成等。

验收重点不是看某个 demo 能不能跑，而是看完整闭环是否成立：

用户任务 → Agent Runtime → Provider → Sandbox / Firewall → Runtime 执行 → Practice 记录 → Memory 沉淀 → How 恢复 → Dashboard 可观测 → Forge 扩展能力。

只有这个闭环在多个真实任务中跑通，并且有日志、事件、轨迹、回放、记忆、失败恢复证据，才能认为 ROSClaw v1.0 具备发布质量。

---

# 二、验收总目标

ROSClaw v1.0 验收目标分为 6 层。

```text
L0：安装启动验收
L1：模块契约验收
L2：Claude Code 接入验收
L3：单机器人任务验收
L4：失败恢复与记忆验收
L5：自扩展与自进化验收
```

真正通过 v1.0，不是每层都有炫酷效果，而是每层都有**可复现证据**。

---

# 三、P0 发布门槛

下面这些是 **P0，不通过就不建议宣布 v1.0 完成**。

## P0-1：从零安装与启动

验收命令：

```bash
git clone <rosclaw-v1.0-repo>
cd rosclaw-v1.0

./scripts/install.sh
rosclaw init
rosclaw doctor
rosclaw start
rosclaw status
```

必须看到：

```text
runtime: healthy
event_bus: healthy
seekdb: healthy
registry: loaded
mcp_gateway: healthy
provider_router: healthy
sandbox: healthy
practice: healthy
memory: healthy
dashboard: healthy
```

不接受：

```text
需要开发者手动改路径
需要开发者口头解释
需要进入某个 tmux 手动启动七八个服务
README 命令跑不通
demo 依赖本机隐藏配置
```

---

## P0-2：Claude Code 能作为真实用户接入

验收目标：

```text
Claude Code 不是读代码，而是作为 ROSClaw 用户调用系统。
```

需要验证：

```text
1. Claude Code 能看到 ROSClaw MCP tools
2. Claude Code 能查询机器人 registry
3. Claude Code 能调用 provider
4. Claude Code 能启动 sandbox 任务
5. Claude Code 能查询 practice 记录
6. Claude Code 能查询 memory
7. Claude Code 能触发 how recovery
8. Claude Code 能调用 forge 生成或验证 bundle
```

示例用户提示词：

```text
你现在作为 ROSClaw 用户，请检查当前系统可用机器人、可用 provider、可用 skill，并选择一个最简单的机器人任务运行。
```

Claude Code 应该能返回：

```text
可用机器人：ur5e / go2 / g1 / turtlebot 等
可用 provider：llm、vlm、skill、critic、embedding 等
可用 sandbox backend：mujoco / mock / isaac 可选
可用任务：pid_move、tabletop_reach、g1_walk 等
```

如果 Claude Code 只能读 README，而不能调用 ROSClaw，那不算接入完成。

---

## P0-3：Event Bus 真实工作

附件中明确要求模块之间不能互相乱调，sandbox 应只发布事件，Practice、Memory、Dashboard、Darwin、How 等模块订阅事件，否则模块会互相缠死。

验收事件：

```text
rosclaw.runtime.started
rosclaw.provider.inference.completed
rosclaw.sandbox.episode.started
rosclaw.sandbox.action.blocked
rosclaw.practice.event.created
rosclaw.memory.write.completed
rosclaw.how.recovery.generated
rosclaw.dashboard.trace.updated
```

验收方法：

```bash
rosclaw events tail
```

运行任何任务时，必须看到事件流。

失败标准：

```text
模块之间直接 import 互调
事件总线只是装饰
Dashboard 直接扫文件而不是订阅状态
Memory 不是订阅 Practice / Sandbox 事件，而是被某模块硬调用
```

---

## P0-4：Practice 记录全过程

附件中把 Practice 定位为统一时空轴、MCAP、PraxisEvent、artifact URI 的记录系统；一次任务结束后，应能回放 Agent 意图、Provider 选择、Sandbox 检查、Runtime 执行、Critic 判断、Memory 写入。

每次任务至少生成：

```text
episode_id
task_id
trace_id
robot_id
agent_request
provider_trace
sandbox_result
runtime_action
critic_result
memory_write_result
artifact_uri
```

目录示例：

```text
.rosclaw/artifacts/episodes/ep_0001/
├── metadata.json
├── events.jsonl
├── provider_trace.jsonl
├── trajectory.jsonl
├── sandbox_replay.json
├── critic_result.json
└── memory_write.json
```

验收命令：

```bash
rosclaw practice list
rosclaw practice show ep_0001
rosclaw practice replay ep_0001
```

不接受：

```text
只有日志，没有结构化 episode
只有控制结果，没有 agent/provider/sandbox/memory 链路
失败后无法复现
```

---

## P0-5：Memory 能回答真实问题

验收提示词：

```text
刚才那个任务发生了什么？
为什么失败？
哪个 provider 被调用了？
sandbox 有没有拦截？
上一次类似任务是怎么成功的？
下次应该怎么调整？
```

合格回答必须包含：

```text
episode_id
失败阶段
关键事件
证据 artifact
相似历史
恢复建议
置信度或依据
```

如果 Memory 只能做普通 RAG，不能查询 episode / robot / skill / failure / success pattern，那不算 ROSClaw Memory。

---

## P0-6：How 能从失败中给恢复策略

验收条件：

```text
故意制造一次失败。
系统必须记录失败。
Memory 必须能解释失败。
How 必须生成恢复建议。
下一轮任务必须能应用这个建议。
Practice 必须记录“恢复前 / 恢复后”差异。
```

示例：

```text
第一次机械臂抓取失败：夹爪位置偏高，接触不足。
How 输出：下次 approach z 降低 2cm，夹爪闭合力增加 15%，横向速度降低。
第二次执行：参数被 patch，任务成功率提升。
```

不接受：

```text
失败后只打印 error
How 没有被触发
恢复建议没有来源
下一次执行没有应用建议
```

---

# 四、Claude Code 用户视角验收流程

建议让一个没有参与开发的 Claude Code 实例做“用户验收官”。

## Step 1：环境发现

提示词：

```text
你是第一次使用 ROSClaw 的用户。请不要看开发者解释，只根据 CLI、README、MCP tools 和系统状态，判断当前 ROSClaw 是否可以启动，并列出可用机器人、provider、skill、sandbox backend、memory、practice、dashboard。
```

期望它调用：

```bash
rosclaw doctor
rosclaw status
rosclaw robot list
rosclaw provider list
rosclaw skill list
rosclaw sandbox list-worlds
rosclaw memory status
rosclaw practice list
```

验收点：

```text
用户不需要找开发者问路径
CLI 返回清晰
错误提示可操作
```

---

## Step 2：最小闭环任务

提示词：

```text
请运行一个最小 ROSClaw 闭环任务：选择一个可用机器人，调用 provider 生成动作建议，先进入 sandbox 验证，再由 runtime 执行 mock/sim 动作，practice 记录全过程，memory 写入结果，dashboard 展示 trace。
```

合格闭环：

```text
Agent Runtime
  ↓
Provider Router
  ↓
Sandbox
  ↓
Runtime
  ↓
Practice
  ↓
Memory
  ↓
Dashboard
```

这条链路正好对应附件中的 v1.0 最小完整闭环。

---

## Step 3：故意失败任务

提示词：

```text
请故意构造一个会失败或被拦截的机器人动作，例如越界移动、碰撞桌面、速度过高、目标不可达。然后验证 sandbox/firewall 是否拦截，practice 是否记录，memory 是否能解释，how 是否能给出恢复建议。
```

必须看到：

```text
SandboxActionBlocked / RuntimeExecutionFailed
risk_score
blocked_reason
replay_id
memory failure record
how recovery hint
```

---

## Step 4：第二轮改进任务

提示词：

```text
请基于上一次失败的 memory 和 how 建议，重新执行任务，并比较两次 episode 的差异。
```

必须输出：

```text
ep_0001: failed
ep_0002: recovered / improved
parameter_patch
success_delta
evidence
```

这一步是验证 ROSClaw 是否有“实践—记忆—恢复—再实践”的核心能力。

---

# 五、实际测试场景设计

下面是建议必须做的 6 个验收场景。

---

# 场景 A：小车 PID 运动控制

## 目标

验证 ROSClaw 是否能处理最基础的机器人闭环控制任务。

## 任务描述

```text
让一个差速小车 / TurtleBot / mock mobile base 从 x=0 移动到 x=1.0m，使用 PID 控制速度，要求误差小于 5cm。
```

## 涉及模块

```text
rosclaw-runtime
rosclaw-provider
rosclaw-sandbox
rosclaw-practice
rosclaw-memory
rosclaw-dashboard
```

## 用户提示词

```text
请使用 ROSClaw 控制小车完成一个 1 米直线运动任务。要求先在 sandbox 中验证 PID 参数，再执行仿真控制，最后记录 episode，并告诉我最终误差、超调量、稳定时间。
```

## 验收指标

```text
1. 能加载 mobile_base e-URDF / mock robot profile
2. Provider 能生成或选择 PID 参数
3. Sandbox 能仿真运动
4. Runtime 能执行 velocity command
5. Practice 记录 setpoint / actual position / error / command
6. Memory 写入 PID 任务结果
7. Dashboard 显示轨迹曲线
```

## 量化指标

```text
最终位置误差 <= 0.05m
最大超调 <= 0.15m
控制频率 >= 10Hz，模拟即可
episode 记录完整率 = 100%
```

## 故意失败测试

```text
把 Kp 设置过大，制造振荡。
```

期望结果：

```text
Practice 记录振荡
Critic 判断控制不稳定
Memory 记录 failure_type=pid_oscillation
How 建议降低 Kp 或增加 Kd
第二轮执行误差降低
```

---

# 场景 B：机械臂 MuJoCo / MoveIt 仿真 reach

## 目标

验证 e-URDF、Sandbox、Firewall、Runtime、Practice 的基本物理任务能力。

## 任务描述

```text
UR5e / Franka / mock arm 从初始位姿移动到桌面上方目标点。
```

## 用户提示词

```text
请使用 ROSClaw 让机械臂在仿真中执行 reach 任务：目标点是桌面上方 20cm。要求先进行 sandbox 安全预演，检查碰撞和关节限制，再执行动作，并把全过程记录到 practice。
```

## 涉及模块

```text
e-URDF-Zoo
rosclaw-sandbox
rosclaw-provider
rosclaw-runtime
rosclaw-practice
rosclaw-memory
rosclaw-dashboard
```

## 验收指标

```text
1. rosclaw robot inspect ur5e 能输出 DOF、joint limits、collision links
2. rosclaw sandbox validate ur5e 能通过
3. reach action 先进入 firewall check
4. 合法动作 ALLOW
5. 危险动作 BLOCK
6. Practice 有 trajectory
7. Dashboard 能回放轨迹
```

## 量化指标

```text
目标点误差 <= 3cm
关节限制 violation = 0
碰撞 violation = 0
sandbox replay 可打开
```

## 故意失败测试

```text
让末端目标点穿过桌面或超出工作空间。
```

期望结果：

```json
{
  "decision": "BLOCK",
  "reason": "workspace_boundary or collision",
  "replay_id": "sandbox://..."
}
```

并且：

```text
practice 收到 FirewallActionBlocked
memory 能回答“为什么刚才机械臂动作被拦截”
how 给出替代目标或路径建议
```

附件中对 firewall mode 的验收也明确要求危险动作被拦截、原因可解释、replay 可回放、practice 收到事件、memory 记录失败原因。

---

# 场景 C：机械臂桌面抓取红杯子

## 目标

验证 Provider + Skill + Sandbox + Critic + Memory 的完整链路。

## 任务描述

```text
桌面上有红杯子，系统需要定位红杯子，生成抓取动作，在 sandbox 中验证，再执行仿真抓取，critic 判断成功。
```

## 用户提示词

```text
请让 ROSClaw 完成桌面红杯子抓取任务。你需要调用 VLM/感知 provider 定位目标，调用 grasp skill provider 生成抓取方案，进入 sandbox/firewall 检查，执行仿真动作，最后用 critic 判断是否成功，并把 episode 写入 memory。
```

## 涉及模块

```text
vlm provider
skill provider
critic provider
sandbox
runtime
practice
memory
how
dashboard
```

## 验收指标

```text
1. Provider Router 能选择 vlm.object_grounding
2. Skill Provider 能输出 grasp candidate
3. Sandbox 能检查抓取路径
4. Runtime 能执行 mock / sim 动作
5. Critic 能判断 success / failure
6. Memory 记录 success pattern 或 failure memory
7. Dashboard 展示 provider latency、sandbox result、trajectory
```

## 故意失败测试

```text
目标识别错误
抓取点偏移
夹爪闭合太早
夹爪力度不足
```

期望恢复：

```text
How 给出 retry plan
第二次抓取参数被调整
Memory 能关联两次 episode
```

---

# 场景 D：Unitree Go2 / 移动机器人巡检

## 目标

验证 ROSClaw 作为具身巡检基础设施，而不只是单动作 demo。

## 任务描述

```text
机器人在仿真环境中巡检 3 个点位：
A：仪表盘
B：阀门
C：门口
要求导航到每个点，拍照或感知，记录状态，发现异常后生成处置建议。
```

## 用户提示词

```text
请使用 ROSClaw 运行一个三点巡检任务：从起点出发，依次巡检仪表盘、阀门、门口。每个点需要记录观测、判断是否异常、写入 practice 和 memory。如果发现异常，请调用 how 生成处置建议。
```

## 涉及模块

```text
agent runtime
provider router
vln / navigation provider
vlm / inspection provider
sandbox
practice
memory
how
dashboard
```

## 验收指标

```text
1. Task 被分解成 waypoint / inspect / report
2. 每个点位形成 PraxisEvent
3. 每个观测有 artifact URI
4. Memory 能查询“上次阀门状态”
5. How 能对异常生成建议
6. Dashboard 能显示巡检 timeline
```

## 量化指标

```text
3 个点位全部访问
每个点位至少 1 条 observation event
异常识别结果写入 memory
任务最终报告可生成
```

## 失败测试

```text
B 点无法到达
仪表盘图像模糊
阀门状态不确定
```

期望结果：

```text
系统不是直接失败退出，而是记录原因、请求重新观测或给出替代路径。
```

---

# 场景 E：G1 人形机器人行走仿真

## 目标

验证 ROSClaw 对更复杂本体的支持能力。

## 任务描述

```text
加载 G1 / humanoid profile，在 sandbox 中执行短距离 walking policy 或 mock gait，记录稳定性指标。
```

## 用户提示词

```text
请使用 ROSClaw 加载 G1 人形机器人模型，运行一个 3 米前向行走仿真任务。要求记录质心、高度、速度、跌倒事件、接触状态，并在 dashboard 中展示 episode。
```

## 涉及模块

```text
e-URDF-Zoo
sandbox
provider
runtime
practice
memory
dashboard
```

## 验收指标

```text
1. G1 robot profile 能被 registry 识别
2. capabilities.yaml 能描述 walking / balance / sensors
3. sandbox 能运行 humanoid task，哪怕是 mock policy
4. practice 记录 contact / base pose / fall event
5. memory 能回答“G1 上次为什么跌倒”
```

## 量化指标

```text
行走距离 >= 3m，仿真或 mock 均可
跌倒检测可触发
episode 记录完整
```

## 故意失败测试

```text
设置过高速度或不稳定地形。
```

期望结果：

```text
SandboxEpisodeFailed
failure_type=fall_detected / unstable_gait
How 建议降低速度、缩短步长或切换 gait provider
```

---

# 场景 F：Forge / sdk_to_mcp 自扩展能力

## 目标

验证 ROSClaw 能否自我扩展，而不是永远依赖人工写适配器。

附件中明确提出 `sdk_to_mcp` 应作为 Asset Compiler，可以生成 MCP Server、Skill Package、Provider Manifest、e-URDF Patch、Sandbox Validation Spec、Tests / CI 等。

## 任务描述

```text
给 ROSClaw 一个简单硬件 SDK 文档或模拟 SDK，让 Claude Code 调用 Forge 生成 MCP bundle，并安装到 staging。
```

## 用户提示词

```text
这里有一个简单传感器 SDK 文档，请使用 ROSClaw Forge 生成一个 ROSClaw-native MCP bundle。要求生成 MCP Server、Skill Manifest、Provider Manifest、测试文件和 README。不要直接启用生产环境，只安装到 staging，并运行 sandbox / critic validation。
```

## 验收指标

```text
1. Forge 能读取 SDK 文档
2. 能生成 bundle
3. Critic 能检查 async、schema、safety、preemption、firewall hook
4. validate 能通过
5. install --staging 能成功
6. Claude Code 能看到新 MCP tool
7. Practice 记录 forge 过程
8. Memory 记录新增能力
```

## 失败测试

```text
故意给 SDK 文档缺少安全限制。
```

期望结果：

```text
Critic 阻止启用
生成修复建议
不允许进入 production runtime
```

附件中也明确要求：Agent 可以生成能力，但不能绕过 sandbox、critic、approval 直接启用物理执行能力。

---

# 六、模块级验收清单

## 1. Runtime

必须通过：

```bash
rosclaw start
rosclaw stop
rosclaw restart
rosclaw status
rosclaw doctor
rosclaw logs
```

验收点：

```text
生命周期统一
配置统一
日志统一
服务健康检查统一
模块注册统一
```

失败标准：

```text
某模块只能靠 tmux 手工启动
某模块配置散落在个人目录
Runtime 不知道模块状态
```

---

## 2. MCP / Agent Runtime

验收点：

```text
Claude Code 能通过 MCP tools 使用 ROSClaw
AgentContext 包含 goal、robot、world、memory、skills、tools
工具调用有 trace_id
错误返回结构化
```

必须测试：

```text
list_robots
list_providers
run_sandbox_task
query_memory
explain_failure
compile_asset_bundle
```

---

## 3. Provider

验收点：

```text
Provider Manifest 存在
Provider 能注册
Provider health 可检查
Provider 输入输出 schema 可验证
Provider latency 可记录
Provider 失败可 fallback
```

必须有至少 4 类 provider：

```text
llm.summary / task_planning
vlm.object_grounding 或 mock perception
skill.pid / reach / grasp
critic.success_detection
```

---

## 4. Sandbox / Firewall

验收点：

```text
能加载 e-URDF
能创建 world
能运行 episode
能生成 replay
能做安全检查
能 ALLOW / BLOCK / MODIFY / REQUIRE_CONFIRMATION
```

必须测试：

```text
合法动作通过
危险动作拦截
拦截原因可解释
replay 可回放
```

---

## 5. Practice

验收点：

```text
所有任务都有 episode_id
所有事件有 trace_id
所有 artifact 有 URI
能 list / show / replay
能导出 timeline
```

必须包含：

```text
agent request
provider trace
sandbox result
runtime action
critic result
memory write
```

---

## 6. Memory / SeekDB

验收点：

```text
能写入 episode
能写入 failure
能写入 success pattern
能检索相似任务
能解释最近失败
能查询机器人能力
```

必须问答：

```text
刚才发生了什么？
为什么失败？
有没有类似成功案例？
下次怎么改？
```

---

## 7. How

验收点：

```text
能订阅失败事件
能读取 Memory
能生成 RecoveryHint
能输出参数 patch
能被下一轮任务使用
```

必须验证：

```text
失败 → 解释 → 恢复建议 → 重试 → 对比提升
```

---

## 8. Know

这是最容易“做了但不会用”的模块。

验收方式：

```text
让 Claude Code 遇到一个不会的机器人 / SDK / skill。
要求它先查 Know，再决定是否调用 Forge 或 Provider。
```

用户提示词：

```text
我想接入一个新的机器人底盘 SDK。请先查询 ROSClaw Know 中是否有相关知识、接口模板、安全注意事项和已有 bundle，再决定是否生成新 MCP。
```

通过标准：

```text
Know 能返回相关文档 / 模板 / 约束
Claude Code 能引用 Know 的结果做决策
Practice 记录 Know 查询
Memory 记录该知识被使用
```

不通过：

```text
Know 只是文档库，主流程完全不用它。
```

---

## 9. Dashboard

验收点：

```text
Runtime Overview
Provider Health
Robot Registry
Sandbox Replay
Firewall Blocks
Practice Timeline
Memory Browser
Event Bus Monitor
Forge Bundle Viewer
```

必须看到：

```text
一次任务从 Agent 到 Memory 的完整 trace。
```

不接受：

```text
Dashboard 只有静态页面
不能看到事件流
不能看到 episode
不能打开 replay
```

---

# 七、验收评分标准

建议采用 100 分制。

```text
A. 安装与启动：10 分
B. Claude Code / MCP 接入：15 分
C. Runtime / Event Bus 架构：15 分
D. Sandbox / Firewall：15 分
E. Provider / Skill：10 分
F. Practice / Replay：15 分
G. Memory / How：10 分
H. Dashboard / Observability：5 分
I. Forge / sdk_to_mcp：5 分
```

发布建议：

```text
>= 85 分：可以进入 v1.0 RC
70 - 84 分：只能内部试用，不建议对外发布
< 70 分：说明仍然是模块集合，不是 v1.0 Runtime
```

P0 阻塞项无论总分多少都不能放行：

```text
1. 不能 clean install
2. Claude Code 不能通过 MCP 使用系统
3. Runtime 不能统一管理模块
4. Event Bus 没有真实事件流
5. Sandbox 不能拦截危险动作
6. Practice 不能记录完整 episode
7. Memory 不能解释失败
8. Dashboard 看不到完整 trace
```

---

# 八、验收报告模板

每个场景都要求开发提交下面格式的证据。

````markdown
# ROSClaw v1.0 Acceptance Report

## Scenario
机械臂桌面抓取红杯子

## User Prompt
<原始用户提示词>

## Environment
- Commit:
- Branch:
- Machine:
- GPU:
- Simulator:
- Robot:
- Providers:
- ROS / ROS2 version:

## Execution Command
```bash
rosclaw demo tabletop_pick --robot ur5e --world tabletop
````

## Result

* Status: PASS / FAIL / PARTIAL
* Episode ID:
* Trace ID:
* Dashboard URL:
* Replay URI:
* Artifact Directory:

## Module Evidence

| Module    | Evidence          | Status |
| --------- | ----------------- | ------ |
| Runtime   | status / logs     | PASS   |
| Provider  | provider trace    | PASS   |
| Sandbox   | replay / decision | PASS   |
| Practice  | episode record    | PASS   |
| Memory    | query result      | PASS   |
| How       | recovery hint     | PASS   |
| Dashboard | trace screenshot  | PASS   |

## Metrics

* Success rate:
* Runtime latency:
* Provider latency:
* Sandbox check time:
* Final error:
* Memory write latency:

## Failure Analysis

<如果失败，必须说明是哪一层失败>

## Reproducibility

```bash
<一条命令复现>
```

## Conclusion

是否满足 v1.0 发布标准？

````

---

# 九、建议安排一次“验收日”

不要让开发继续自己说做完。你可以安排一次集中验收。

## 角色分工

```text
你：验收负责人 / 用户代表
主管：发布决策人
主集成开发者：负责修复阻塞问题
模块开发者：只回答本模块证据
Claude Code：真实用户代理
Qwen / 第二 Claude：红队审查
````

## 验收顺序

```text
上午：
1. clean install
2. rosclaw doctor
3. Claude Code MCP 接入
4. Runtime / Event Bus 检查

下午：
5. 小车 PID
6. 机械臂 reach
7. 机械臂抓取
8. 巡检任务
9. G1 行走
10. Forge 生成 bundle

最后：
11. 故意失败
12. Memory 解释
13. How 恢复
14. 第二轮重试
15. Dashboard 全链路复盘
```

---

# 十、最关键的验收结论标准

ROSClaw v1.0 如果真的完成，应该能做到下面这件事：

```text
一个不参与开发的用户，打开 Claude Code，
通过 ROSClaw MCP 提出一个物理任务，
系统能自动发现机器人能力，
选择 provider，
进入 sandbox 预演，
安全后执行，
记录全过程，
失败后解释原因，
给出恢复策略，
第二次执行有所改进，
最后在 dashboard 上看到完整 trace。
```

如果做不到，那就不能说 v1.0 完成。
最多只能说：

```text
核心模块已开发完成，但尚未通过用户闭环验收。
```

我建议你对团队使用这个最终判据：

**ROSClaw v1.0 不是看“代码是否完成”，而是看“一个真实用户能否通过 Agent 使用它，让物理任务形成可验证、可回放、可记忆、可恢复、可扩展的闭环”。**
