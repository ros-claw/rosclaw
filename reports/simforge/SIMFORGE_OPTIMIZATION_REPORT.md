# ROSClaw SimForge 验证与优化报告

日期：2026-07-22（Asia/Shanghai）
工作目录：`/code/rosclaw/rosclaw_test`
上游：`https://github.com/ros-claw/rosclaw.git`
最终集成的 `origin/main`：`daf13a0c98a533e0f6f46a92177421f29b376f10`
SimForge 开发起点：`1ce343a369750bef0449c1bf750cf7b4be18704f`

## 结论

本轮把 ROSClaw 从“闭环接口存在”推进到可在本机复现的仿真证据闭环：

- P0-P3 已实现并形成 UR5e Shield Evolution 的真实 MuJoCo 执行、Practice 记录、同 seed 重试、Physics Darwin 和 fail-closed 晋级链；
- P4 已在 4×RTX A6000 上跑通 MJWarp 四进程一致性验证，以及 Isaac Lab 4-rank NCCL/Newton/MJWarp 训练；
- P5 已跑通 ROS 2 Humble rosbridge/turtlesim 安全发现与订阅，以及 Gazebo Fortress `/clock` 真桥接；
- Hub 本地签名发布、远端 dry-run 下载、安装、卸载闭环通过；
- 社区 Isaac Sim MCP 完成真实 stdio→MCP server→Isaac extension 协议调用；
- 未执行任何真实机器人动作，未进行公开 Hub/ClawHub 上传。

合入最终上游后，普通测试集：`4806 passed, 57 skipped, 24 deselected`，耗时 746.79 秒；新增 LeRobot 定向测试另行复核为 `31 passed`。本次变更的 78 个 Python 文件通过 Ruff 规则与格式检查，`src tests scripts` 规则检查通过，mypy 指定的 118 个源文件通过，`git diff --check` 和 Skill 校验均通过。

## 机器与环境

- Ubuntu 22.04.5 LTS；约 125 GiB RAM；
- 4×NVIDIA RTX A6000，单卡约 49 GiB；驱动 595.71.05；
- ROS 2/Gazebo 使用 Docker 隔离安装，避免依赖宿主机 sudo；
- Isaac Lab 使用独立 `.venv-isaaclab`；MJWarp 使用独立 `.venv-mjwarp`；
- 用户级 ROSClaw 已从旧 1.0.0 更新到本工作树构建的 1.0.1，包含 `hub`、MCP 和 daemon 命令。

## P0：真实性与证据

- 新增 `EvidenceDomain`，将 FIXTURE、SIMULATION、REPLAY、SHADOW、HARDWARE 与证据强度解耦；
- mock runner 显式标记，始终 `valid_for_promotion=false`；
- Auto、Skill Evaluation 和 Promotion Gate 在缺少真实 physics、多 seed、replay、artifact hash 或 regression evidence 时 fail closed；
- 静态检查正名为 `StaticActionGate`，明确 `physics_executed=false`；
- SHA-256 稳定 seed 替代跨进程不稳定的 Python `hash()`；
- `TASK_VERIFIED + SIMULATION` 不再可能被误读为真实硬件证据。

## P1：Sandbox 2.0

实现 backend protocol/registry、ScenarioSpec、CPU MuJoCo backend、完整 waypoint 插值验证、executor registry、SimulationReceipt 和 strict replay。

轨迹后端会记录碰撞、速度、加速度、tracking error、contact force、NaN/不稳定和 deadline miss。黄金反例的起终点均安全，但中途穿过桌面；旧的终点检查会漏判，新后端在 `geom11` 与 `tabletop_surface` 间检出 `COLLISION_DETECTED`。篡改 artifact 后 strict replay fail closed。

## P2-P3：Practice、Retry 与 Physics Darwin

`SandboxPracticeBridge` 将物理执行事件写入 Practice，校验 artifact hash，并发布 physics-backed episode terminal。`RetryOrchestrator` 只接受白名单 immutable patch，第一轮使用相同 scenario/seed，并有 retry 上限。规则有效性更新必须具备 physics receipt、数据质量通过和 SIMULATION domain。

UR5e Shield Evolution 正式结果：

| 指标 | Baseline | Candidate |
| --- | ---: | ---: |
| Success rate | 0.0 | 1.0 |
| Collision rate | 1.0 | 0.0 |
| Worst-seed success | 0.0 | 1.0 |
| Strict replay success | 1.0 | 1.0 |
| Receipt completeness | 1.0 | 1.0 |

Physics Darwin 在 seed 42/43 间交替 baseline/candidate 顺序，Promotion Gate 八项检查通过，结果为 `promote_to_sim`，最高声明严格限制为 `SIM_CHAMPION`。

## P4：四卡 GPU 与 Isaac Lab

### MJWarp

四个隔离进程分别绑定 `CUDA_VISIBLE_DEVICES=0,1,2,3`。共执行 8 worlds、2800 world-steps；碰撞场景均命中预期桌面接触，所有状态有限，单 shard 额外显存约 68 MiB。共享 GPU 的小规模正确性运行观测吞吐为 962.26 world-steps/s；safe control 为零桌面碰撞。

### Isaac Lab

- 官方 IsaacLab develop commit：`b634245535dd7572f13a5699e0ff2fd2542b33c7`；
- Isaac Lab 12.0.0、PyTorch 2.11.0+cu128、Newton 1.5.0.dev0、MJWarp 3.10.0.2；
- 单卡 RSL-RL `Isaac-Cartpole-Direct`，Newton/MJWarp，5 iterations：通过；
- 4 卡 `train_multigpu.py`，4 ranks 分别绑定 `cuda:0..3`，NCCL 参数同步，2 iterations/512 steps：通过；
- 新 Skill 的 `verify_isaaclab.sh` 再次完成单卡与四卡验证：`ROSCLAW_ISAACLAB_VERIFY_OK`。

## P5：ROS 2、turtlesim 与 Gazebo

宿主机没有 ROS 2 且 sudo 不可用，因此使用官方 ROS 2 Humble Docker 镜像。Docker daemon 不能直接走宿主 loopback proxy，使用 checksum 校验过的 `crane v0.21.7` 拉取并载入基础镜像。

ROS 2/turtlesim：

- deployment 4/4：容器、doctor、ROS graph discovery、capability compile；
- live integration 6/6：ping、发现、manifest 编译、直连 legacy velocity 阻断、unsafe velocity 阻断、pose 订阅；
- 没有向 turtlesim 或真实机器人发布运动命令；
- 修复 Compose build proxy/host-network 传递及 websocket-client 对 loopback rosbridge 的 502 代理问题。

Gazebo：

- 新增 `docker/ros2-humble-gazebo.Dockerfile`；
- 按 Humble 的官方配对安装 Gazebo Fortress / Ignition Gazebo 6.18.0 与 `ros_gz`；
- headless server 启动 `grid.sdf`，`ros_gz_bridge` 将 Gazebo `/clock` 桥接至 ROS 2，`ros2 topic echo /clock --once` 收到真实时钟样本；
- `verify_gazebo.sh` 输出 `ROSCLAW_GAZEBO_VERIFY_OK`。

## Hub 上传/下载与生态探索

使用隔离的本地 HTTP fake registry 和测试签名密钥完成：

1. manifest validate；
2. trusted signature verify；
3. login/whoami；
4. private signed publish；
5. sync/search；
6. remote dry-run download；
7. install/list/uninstall。

单测 7/7，Skill 脚本输出 `ROSCLAW_HUB_VERIFY_OK`。这里验证的是开发者本地 registry，不代表生产公共 Hub；签名密钥仅为 fixture。没有公开上传，因为没有获得公共凭据/发布授权。

OpenClaw 2026.5.7 的隔离 workspace 中检索了 ClawHub ROS 路径，安装并审计 `ros2-introspection@1.0.1`。该 skill 为只读 ROS introspection，wrapper 使用 allowlist 与 `shell=False`；它可作为诊断辅助，但安全模型弱于 ROSClaw daemon/MCP 的 canonical `request_action` 路径。未发现 Isaac skill。

## Isaac Sim 与社区 MCP

- 使用现有 `nvcr.io/nvidia/isaac-sim:6.0.0-dev2` 在 GPU1 执行 DynamicCuboid 64 physics steps，z 从 2.0 降到约 -4.0168；物理本身通过；
- dev2 在正常 `SimulationApp.close()` 时触发 TaskGroup abort，因此该容器生命周期只记为 PARTIAL；正式 6.0.1 镜像因 Docker daemon 无代理未能拉取；
- 社区 `whats2000/isaacsim-mcp-server` commit `4704503bfa4ba4a7fd8a4f7d0e8eb036b6d85d31` 安装成功，扩展在 Isaac Sim 中注册 42 handlers；
- 包单测 `8 passed, 43 skipped`；带真实 stage 的 scene tests 5/5；非机器人 live integration 35 passed/2 failed；
- 两个失败均为 Isaac Sim 6 的 `clear_scene` 后过期 prim 生命周期兼容问题；robot list 返回 208 项，但社区测试仍断言旧键 `franka`，而 v6 使用 `frankapanda`/`frankafr3`；
- 完整协议链成功完成 MCP initialize、list_tools=42、`get_scene_info`，结果 `is_error=false`。

该 MCP 是社区项目，不是 NVIDIA 官方 MCP，未写入全局 Codex 配置。当前 `codex mcp list` 中只有 ROSClaw MCP。

## 新增可复用 Skill

`.agents/skills/rosclaw-simforge/` 包含：

- `SKILL.md` 与 `agents/openai.yaml`；
- `references/verified-stack.md`；
- `verify_ros2.sh`；
- `verify_gazebo.sh`；
- `verify_isaaclab.sh`；
- `verify_hub.sh`。

Skill 结构通过 `quick_validate.py`，所有 shell 脚本通过 `bash -n`，四条功能脚本均已实际执行通过。脚本保持真实机器人禁区，并在 ROS 路径中只做发现、订阅和安全阻断验证。

## 本轮额外发现并修复的严重问题

### 1. Loopback 被环境代理劫持

MCP HTTP、DeepSeek 本地 provider、Cosmos reasoner、ROS rosbridge 和测试 Outbox 都曾把 `127.0.0.1` 送至环境代理，返回 502。

修复：

- websocket loopback 使用直连 socket；
- 新增 `rosclaw.utils.http.urlopen_with_loopback_bypass`，只对 127.0.0.1/localhost/::1 禁用代理，远程端点继续尊重代理；
- 测试用错误代理且清空 NO_PROXY，确定性验证直连行为。

### 2. RecoveryHint 递归膨胀导致 128 GB 写盘

完整测试发现 recovery metadata 中的 dict 被 `str()` 后作为下一次 memory analogy 的 hint，再次嵌套并转义，形成指数增长。单个 episode `trajectory.jsonl` 达约 1.3 GB，event rotation 单文件最高约 23.4 GB，`~/.rosclaw` 达 128 GB。

修复：

- Memory 只提取 `hint/action/recovery_hint` 纯文本，最大 4096 字符；
- HOW 只接受有界 scalar analogy action；
- event persistence 单记录最大 1 MiB，超限写摘要；轮转改为写前判断；
- EpisodeRecorder 超大 trajectory/provider record 写有界摘要；
- `RuntimeConfig.workspace_home` 统一传播至 events、traces、artifacts 和 outbox；
- Scenario D 显式使用隔离 workspace 与 SQLite 路径。

效果：Scenario D 从 5 分钟仍未完成并持续 GB 写盘，降为 `7 passed in 11.10s`；完整套件安全跑到 100%。本任务产生的 128 GB 可再生测试文件已精确清理，较早用户数据保留。最终合并后全量回归仅使 `~/.rosclaw` 从 545,014,734 字节增加至 555,769,212 字节（约 10.3 MiB，主要是保留的测试证据），未再出现异常膨胀。

## 最终验证矩阵

| 范围 | 结果 |
| --- | --- |
| 普通 pytest 全套件 | 4806 passed / 57 skipped / 24 deselected；746.79s |
| 上游新增 LeRobot 定向测试 | 31 passed；37.50s |
| Ruff（本次变更 78 文件；`src tests scripts`） | 规则 PASS；本次变更格式 PASS |
| mypy（AGENTS.md 指定范围） | 118 source files PASS |
| git diff whitespace | PASS |
| SimForge Skill schema/shell | PASS |
| CPU MuJoCo + strict replay | PASS |
| Physics Darwin + Promotion Gate | PASS |
| 4×A6000 MJWarp | PASS |
| Isaac Lab 单卡 + 四卡 | PASS |
| ROS 2 deployment | 4 passed |
| turtlesim live safe integration | 6 passed |
| Gazebo Fortress `/clock` bridge | PASS |
| Hub CLI upload/download/install/uninstall | PASS |
| Isaac community MCP protocol | PASS；部分 v6 integration 有兼容问题 |
| Isaac Sim 6.0.0-dev2 physics | PASS；shutdown lifecycle PARTIAL |
| 真实机器人 | 未执行（明确排除） |

## 已知边界

- GPU 同时有用户工作负载，结果是正确性资格验证，不是峰值吞吐 benchmark；
- 全仓 `ruff check .` 仍会在未由本轮修改的 `examples/rh56_rps` 中报告 244 个既有问题；本次变更文件及 `src tests scripts` 均通过规则检查；
- generic mobile-base executor 尚未实现，导航请求仍正确 fail closed；
- Effective Body hash 已绑定，但 actuator-disable/calibration-offset 的穷举 mutation campaign 未完成；
- Isaac Sim 使用 dev2 镜像，需在可拉取 6.0.1/正式镜像后复测 shutdown 与社区 MCP 兼容性；
- 未做视觉模型训练或真实相机闭环；
- 未提交、推送或创建 PR，工作树保留为可审阅修改。
