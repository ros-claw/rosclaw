# W00 基线报告（ROSClaw_ClaudeCode_实施规格_2026-09-08 §4）

**日期**：2026-09-09
**基线 commit**：`f05aafd`（大道至简 R3，#519）
**脏文件**：`scripts/a0901_live_gate.py`（untracked，0901 真实 K3 Gate 脚本——保留，不动）
**机读清单**：`docs/implementation/baseline-versions.json`

## 1. 实际运行版本（实测，非报告转述）

| 项 | 值 | 来源 |
|---|---|---|
| Python | 3.11.15（.venv/bin/python） | 实测 |
| rosclaw 分发 | 1.0.1（editable，源自源码 checkout） | `pip list` |
| CLI | `.venv/bin/rosclaw` → `rosclaw.entrypoint` | 实测 |
| MuJoCo | 3.11.0（dev venv）/ 3.12.0（bundle venv） | 实测 |
| Pillow / NumPy / pydantic / mcp | 12.3.0 / 2.4.6 / 2.13.4 / 1.29.0 | `pip list` |
| Node | 系统 v24.19.0（dev）；**bundle 自带 v22.19.0 arm64** | 实测 |
| Pi pins | coding-agent / ai / tui 全 **0.83.0** | package.json |
| package-lock | sha256[:16]=`e2c557e386b10500`，241 resolved | 实测 |
| 渲染后端 | **osmesa**（egl n/a，xvfb n/a） | `probe_os_isolation` |
| OS 隔离 | **未就绪**（bwrap present-but-broken：RTM_NEWADDR 被拒） | 同上 |
| 资产 | e-urdf-zoo/（树内，wheel force-include） | 实测 |
| 秘密 | 三个 key 均 absent——只记录"已配置/未配置"，从不打印值 | env 探测 |

## 2. §2.2 逐条核对（当前代码实证）

| 发现 | 状态 | 证据（当前代码） | 复现测试 |
|---|---|---|---|
| `_artifact_register` 无 active task + 最近 SUCCEEDED → 拒绝 | **存在（有条件）** | tool_dispatch.py：`active=None` 且 `latest=SUCCEEDED` → `TASK_ALREADY_COMPLETED`。但 dispatcher 层 `ensure_task_for_effect` 会先把任务复活成 RUNNING revision 2 遮掩——0907 的拒绝只在 handler 直达/无新动机时咬人；**复活本身就是规格 §9.2 标记的反模式（"不采用数据库直接改回 RUNNING"）** | `test_w00_baseline_repro.py` R1（xfail strict） |
| sim_render 原子 IPC/后端探测/RenderSpec/overlay | 存在，保留 | R2-2 已建 | — |
| 回放 `data.qpos[:model.nu]` 截断 | **存在** | sim_render.py:563——nq≠nu 模型丢状态 | R2（xfail strict） |
| 无 spec 默认 UR5e / 相机距离固定 / 按索引采样 12fps | **存在** | sim_render.py:503 `robot_id = "ur5e"`；cam.distance=1.2/0.9/1.4 写死；fps=12 按帧数索引采样（非时间戳） | W04 处理 |
| 同一 trace 固定结果文件名 + `candidates[:2]` | **存在** | sim_render.py:233 | R3（xfail strict） |
| render_profiles 仅注册 sim/ur5e | **存在** | render_profiles.py:16 | W02 处理 |
| entrypoint open alias + artifact dispatcher | 存在，保留 | 已验证 isolated install 内可用 | — |
| Pi pins 0.83.0 + postinstall patches | 存在 | package.json + patches/apply-upstream-patches.mjs | W01 盘点 |
| build-info 写死 pi_version/pi_commit | **存在** | build_release.sh：`"pi_version": "0.83.0"`, `"pi_commit": "588915ec…"` | W11 处理 |
| wheel 不含 packages/rosclaw-agent（JS/Node） | **存在** | pyproject.toml force-include 只有 zoo/configs/policies/benchmarks/worker_plugins——wheel 与 tar bundle 内容不一致 | W11 处理 |
| pytest 默认排除 integration/deployment | **存在** | pyproject.toml addopts `-m 'not integration and not deployment'` | W09/W12 处理 |

## 3. 基线分发物与独立安装测试

- 构建：`scripts/build_release.sh` → `dist/rosclaw-1.2.0-linux-arm64.tar.gz`（6534 文件验签通过，77 vendor wheels）。
- 隔离安装：`/tmp/w00-pkgtest`（**不挂源码、不继承 dev PYTHONPATH、不改名 worktree**）→ `install.sh --offline --allow-untrusted-dev` exit 0，健康检查通过。
- Smoke：`rosclaw --version` = 1.2.0；`rosclaw doctor` 健康（MuJoCo 3.12.0、zoo 在 bundle site-packages 内）。

## 4. 四路复现结果

| 路径 | 结果 | 说明 |
|---|---|---|
| unsafe rollout 错误成功表述 | **NOT_RUN** | 无 ROSCLAW_KIMI_API_KEY——真实模型回归不可执行（skip 标记，不合成冒充） |
| 追加交付 | **复现（R1 xfail）** | handler 直达必现 TASK_ALREADY_COMPLETED；另发现 admission 复活 RUNNING 遮掩（同列为 W05 修复面） |
| 非 UR5e 状态回放 | **复现（R2 xfail）** | `qpos[:nu]` 截断在源码实证 |
| 无 Node/npm 启动 | **通过（无缺陷）** | PATH 无 node：bundle 自带 Node 使 chat 正常进 TUI，stdin EOF 干净退出，无 traceback |

## 5. 用户可见输出 / 耗时 / 费用 / Artifact 可达性（基线）

- 安装：exit 0；doctor 全绿（除预期 pytest 缺失/workspace config 提示）。
- chat 启动：TUI 正常渲染（80 列）；启动警告文案"shell 类操作将在会话内弹确认卡降级运行"**已过时**（R1-2b 后 SIM 不再弹卡——W07 文案修正项）。
- Artifact：isolated install 内 `rosclaw artifact list` 可达（0901 P0-1 已证，本轮未重跑真实产物）。
- 耗时/费用：真实模型路径 NOT_RUN，无数据（不编造）。

## 6. NOT_RUN 清单（明示，后续批次回填）

1. 0907 unsafe rollout 真实主模型回归（§16.2-1）——待 key。
2. 真实 A/B（W10）——待 key（R3 harness 就绪）。
3. 十类 Agent 测试的真实模型段（W09）——待 key。
4. 用户盲测——待 operator。

## 7. 后续工作包使用该映射

- W01：Pi 补丁盘点（patches/apply-upstream-patches.mjs）+ 0.83.0 基线固定。
- W02：render_profiles 从力学模型生成 + body 契约。
- W03：qpos/qvel/ctrl 完整状态记录与回放（解 R2 xfail）。
- W04：后端降级全候选 + 时间驱动采样 + operation 目录（解 R3 xfail）。
- W05：追加交付与复活反模式（解 R1 xfail——注册既有 revision，不改回 RUNNING）。
- W07：启动警告文案修正。
- W09/W11/W12：pytest integration 收集、wheel JS/Node 打包、build-info 动态化。
