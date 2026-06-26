# ROSClaw Body 模块 P2 实施报告与移交文档

> 文件名关键词：`body-p2`、`implementation-report`、`handover`、`fleet`、`mcp`、`dashboard`、`multi-body-registry`  
> 生成时间：2026-06-26  
> 作者：Claude Code（Body P2 实施者）  
> 关联计划：`/home/ubuntu/.claude/plans/lexical-tinkering-dawn.md`  
> 关联 PR：https://github.com/ros-claw/rosclaw/pull/13（已合并）

---

## 1. 文档目的

本报告用于：

1. **完整记录** Body 模块 P2 里程碑的开发内容、验证结果和当前状态。
2. **明确边界**：由于本工作与另一位 **e-URDF 开发者** 的内容存在交集，本文档对双方的职责范围、重合点、接口契约进行划分。
3. **本地移交**：为后续维护者、 reviewer 或 e-URDF 开发者提供一份可落地的上下文说明，避免重复开发或冲突修改。

---

## 2. 我实施的内容（Body P2）

### 2.1 范围总览

依据已批准的 P2 计划，本次实现了：

- 多 body registry 与 resolver
- MCP body registry / compatibility tools
- Dashboard body 页面
- 跨 body 技能兼容性聚合
- Fleet 级 CLI 操作（status / stop）
- 相关测试、文档、CI 清理

### 2.2 新增 / 修改的核心文件

| 类别 | 文件 | 说明 |
|------|------|------|
| **Registry / Resolver** | `src/rosclaw/body/registry.py` | `BodyRegistryManager`：创建、切换、删除、列出、统计 body |
| | `src/rosclaw/body/resolver.py` | `BodyResolver`：按 `body_id` 路由，兼容 legacy `workspace/body/` |
| | `src/rosclaw/body/schema.py` | `BodyRegistryEntry`、`BodyRegistry`、`FleetCompatibilityReport`、`EffectiveBody.readiness` 等 |
| | `src/rosclaw/body/fleet.py` | `FleetCompatibilityAggregator`、`discover_skill_manifests()` |
| | `src/rosclaw/body/compiler.py` | `EffectiveBodyCompiler`（已存在，本次少量调整 readiness） |
| **CLI** | `src/rosclaw/body/cli.py` | 新增 `fleet-compat`、`switch`、`remove`、`list`、`history`、`export`、`diff`、`update-state`、`note` 等子命令 |
| | `src/rosclaw/cli.py` | 新增 `rosclaw fleet status`、`rosclaw fleet stop`；扩展 `rosclaw skill check` 支持 workspace manifest |
| **MCP** | `src/rosclaw/mcp/adapters/runtime_client.py` | 新增 6 个 body tools 方法，含 fixture 降级 |
| | `src/rosclaw/mcp/tools/__init__.py` | 注册 6 个新 tools 并加入 `P0_TOOLS` |
| **Dashboard** | `src/rosclaw/dashboard/server.py` | 新增 `get_body_summary()`，广播 snapshot 含 body 字段 |
| | `src/rosclaw/dashboard/web_server.py` | 新增 `/api/body`、`/body` 页面与 WebSocket body 数据 |
| **兼容性** | `src/rosclaw/body/compatibility.py` | `SkillCompatibilityChecker`（已存在，本次持续调用） |
| | `src/rosclaw/skill_manager/executor.py` | fail-closed body compatibility gate（与 e-URDF 编译结果交互） |
| **文档** | `docs/body/FLEET_OPERATIONS.md` | Fleet 操作文档 |
| | `docs/body/BODY_REGISTRY.md` | Registry 与迁移文档 |
| | `docs/body/BODY_HISTORY_EXPORT.md` | History/export 文档 |
| | `docs/body/SKILL_COMPATIBILITY.md` | Skill/body 兼容性文档 |
| | `docs/body/URI_SCHEME.md` | Body URI scheme 文档 |
| | `docs/README.md`、`CHANGELOG.md`、`CLAUDE.md` | 更新索引、变更日志、P0 安全合约表 |
| **测试** | `tests/mcp/test_body_tools.py` | MCP body tools 测试 |
| | `tests/dashboard/test_body_page.py` | Dashboard body 页面测试 |
| | `tests/body/test_fleet_compatibility.py` | Fleet 兼容性聚合测试 |
| | `tests/body/test_multi_body_registry.py` | Registry CRUD、迁移测试 |
| | `tests/body/test_end_to_end_workflow.py` | 端到端 body workflow（含 `skill check`） |
| | `tests/mcp/test_e2e.py` | 扩展 P0 tool calls 列表 |
| | `tests/test_skill_manager.py` | 适配 fail-closed executor 行为 |

### 2.3 新增 MCP Tools（P2）

| Tool | 安全等级 | 功能 |
|------|----------|------|
| `list_bodies` | S0 read-only | 列出 workspace 中所有注册 body |
| `get_body` | S0 read-only | 获取指定 body 的 registry entry + effective body |
| `switch_body` | S0 config | 切换当前 active body（仅改指针，不运动） |
| `list_body_history` | S0 read-only | 列出 body snapshot 历史 |
| `check_skill_compatibility` | S0 read-only | 对当前 body 检查所有 skill 兼容性 |
| `fleet_skill_compatibility` | S0 read-only | 聚合所有 body 的 skill 兼容性报告 |

### 2.4 新增 CLI 命令（P2）

```bash
# Body registry
rosclaw body create --robot unitree-g1 --name g1-sim
rosclaw body switch <body_id>
rosclaw body remove <body_id> [--no-archive]
rosclaw body list [--json]

# History / export / diff
rosclaw body history [--json]
rosclaw body export --format {zip,tar}
rosclaw body diff [--from-gen N] [--to-gen M]

# State / note
rosclaw body update-state --key ... --value ... --reason ...
rosclaw body note --message "..."
rosclaw body update-state --from-ros  # ROS 2 实时 introspection

# Fleet
rosclaw body fleet-compat [--workspace ...] [--json]
rosclaw fleet status [--workspace ...] [--json]
rosclaw fleet stop --reason "..."
```

### 2.5 验证结果

- **PR #13 CI**：全部通过
  - Lint ✅、Type Check ✅、Test (3.11/3.12/3.13) ✅、Integration Test ✅、ROS Docker Deployment Test ✅、First Boot Acceptance ✅、Build Package ✅
- **本地最新 main 全量测试**：`3107 passed, 5 skipped, 23 deselected`
- **本地 lint / mypy**：`ruff check .` clean；focused `mypy` clean

---

## 3. e-URDF 开发者的工作范围（已知 / 推断）

另一位开发者主要负责 **e-URDF 物理 DNA 层** 及由其驱动的 **P0 本体初始化/渲染/校验**。对应文件和目录：

| 类别 | 文件 / 目录 | 说明 |
|------|-------------|------|
| **e-URDF 加载** | `src/rosclaw/eurdf/loader.py` | `EurdfLoader`：加载 e-URDF profile |
| | `src/rosclaw/eurdf/models.py` | `EurdfProfile` 等数据模型 |
| | `src/rosclaw/eurdf/registry.py` | e-URDF profile registry/发现 |
| | `src/rosclaw/runtime/eurdf_loader.py` | Runtime 层 e-URDF 加载 |
| **Profile Zoo** | `e-urdf-zoo/`（项目根目录） | 各机器人 profile 目录，如 `unitree-g1/` |
| | `src/rosclaw/runtime/eurdf_zoo/` | 内置 e-URDF zoo（若存在） |
| **Body 初始化 / 渲染** | `src/rosclaw/body/cli.py` 中的 `cmd_body_init`、`cmd_body_validate`、`cmd_body_render`、`cmd_body_show`、`cmd_body_state`、`cmd_body_query` | P0 本体认知闭环 |
| | `src/rosclaw/body/renderer.py`（若存在）或 compiler 中的渲染逻辑 | `EMBODIMENT.md` / `BODY.md` 渲染 |
| | `src/rosclaw/body/validator.py`（若存在） | body.yaml / calibration.yaml / maintenance.log 校验 |
| **数据模型** | `src/rosclaw/body/schema.py` 中的 `BodyYaml`、`CalibrationYaml`、`MaintenanceEvent`、`EurdfProfile` | 本体四层真源的数据结构 |
| **编译** | `src/rosclaw/body/compiler.py` | `EffectiveBodyCompiler` 将 e-URDF + body.yaml + calibration + maintenance 编译为 `EffectiveBody` |
| **ROS 集成** | `src/rosclaw/body/ros_introspection.py` | `--from-ros` 实时 introspection（近期新增） |

### 3.1 与 `rosclaw_body.md` 指南的对应关系

`rosclaw_body.md`（项目根目录）是一份完整的 **P0 实施指南**，目标包括：

- `rosclaw body init --robot unitree-g1` 生成 `EMBODIMENT.md`、`BODY.md`、`body.yaml`、`calibration.yaml`、`maintenance.log`
- `rosclaw body validate / render / show --agent / state --json / query`
- Fault / maintenance / calibration / capability / retrofit 更新机制
- MCP tools：`get_body_profile`、`get_body_state`、`list_body_capabilities`、`query_body`、`validate_body_action`、`get_calibration_status`
- 安全闭环：enabled ≠ executable，high/critical 必须 sandbox，critical forbidden 必须 block real robot

当前代码中：**`init`、`validate`、`render`、`show`、`state`、`query`、`fault`、`maintenance`、`calibration`、`retrofit`、`capability` 等命令已存在**，说明 e-URDF 开发者或更早的迭代已基本落地 P0。本次 P2 是在该基础上叠加 **多 body registry、fleet 聚合、MCP dashboard 工具**。

---

## 4. 重合点与边界划分

### 4.1 重合点 1：`EurufProfile` 的消费

- **e-URDF 开发者职责**：
  - `EurdfProfile` 的数据结构、字段语义、profile zoo 组织。
  - `EurdfLoader` 的加载逻辑、checksum、版本管理。
- **Body P2 职责**：
  - 通过 `BodyResolver` 调用 `EurdfLoader` 获取 `EurdfProfile`。
  - 在 `EffectiveBodyCompiler` 中将 `EurdfProfile` 与 `BodyYaml`、`CalibrationYaml`、`MaintenanceEvent` 合并为 `EffectiveBody`。
- **接口契约**：
  - `EurdfProfile` 必须提供 `profile_id`、`profile_version`、`joints`、`sensors`、`actuators`、`frames`、`identity`、`capability_hints`、`safety`、`provider_interfaces`、`sandbox` 等字段。
  - Body P2 不修改 `EurdfProfile` 的定义，只读取。

### 4.2 重合点 2：`rosclaw body init` vs `rosclaw body create`

- **`init`（e-URDF 侧主导）**：
  - 默认在当前 workspace 生成单 body 文件（`EMBODIMENT.md`、`body.yaml` 等）。
  - 负责从 e-URDF 渲染初始 body 状态。
- **`create`（Body P2 新增）**：
  - 在多 body registry 中创建独立 body 实例，目录为 `workspace/bodies/<body_id>/`。
  - 内部也会加载 e-URDF profile，但额外写入 `body_registry.yaml` 并维护 `active` 指针。
- **边界**：
  - e-URDF 开发者可继续优化 `init` 的渲染、校验、P0 MCP tools。
  - Body P2 负责 `create/switch/remove/list` 等多 body 生命周期。
  - 未来可考虑将 `init` 统一收敛到 `create`（即 `init` 是 `create` 在当前目录的别名）。

### 4.3 重合点 3：`body.yaml` 与实例状态

- **e-URDF 侧**：提供静态结构来源（joints、sensors、capabilities）。
- **Body P2 侧**：管理实例级状态变更：
  - `update-state`、`note`、`fault add/resolve`、`capability enable/disable/degrade`
  - generation 递增、maintenance.log 追加、snapshot 历史

### 4.4 重合点 4：MCP Tools

| MCP Tool | 归属 | 说明 |
|----------|------|------|
| `get_body_profile` | e-URDF / P0 | 获取本体静态概要 |
| `get_body_state` | e-URDF / P0 | body.yaml + runtime overlay |
| `list_body_capabilities` | e-URDF / P0 | 列出能力 |
| `query_body` | e-URDF / P0 | 自然语言查询 |
| `validate_body_action` | e-URDF / P0 | action proposal 阶段 validation |
| `get_calibration_status` | e-URDF / P0 | 校准状态 |
| `list_bodies` | **Body P2** | 注册表列表 |
| `get_body` | **Body P2** | registry entry + effective body |
| `switch_body` | **Body P2** | 切换 active body |
| `list_body_history` | **Body P2** | snapshot 历史 |
| `check_skill_compatibility` | **Body P2** | 当前 body skill 兼容性 |
| `fleet_skill_compatibility` | **Body P2** | 跨 body 兼容性聚合 |

### 4.5 边界划分总结表

| 领域 | e-URDF 开发者 | Body P2（我） |
|------|---------------|---------------|
| e-URDF profile 格式与 zoo | ✅ 主责 | 只消费 |
| `EurdfLoader` / `EurdfProfile` | ✅ 主责 | 只调用 |
| `EffectiveBodyCompiler` | 共同依赖 | 共同依赖，少量调整 |
| `init` / `validate` / `render` / `show` / `state` / `query` | ✅ 主责 | 不主动改动 |
| `create` / `switch` / `remove` / `list` | 协同 | ✅ 主责 |
| `history` / `export` / `diff` | 协同 | ✅ 主责 |
| `fleet-compat` / `fleet status` / `fleet stop` | | ✅ 主责 |
| MCP P0 body tools | ✅ 主责 | 不改动 |
| MCP P2 body/fleet tools | | ✅ 主责 |
| Dashboard body 页面 | | ✅ 主责 |
| Skill compatibility check | 协同 | ✅ 主责 |
| Fail-closed executor gate | 协同 | ✅ 已加固 |

---

## 5. 移交清单

### 5.1 已合并到 main 的内容

- [x] 多 body registry（`BodyRegistryManager`、`BodyResolver`）
- [x] Body P2 CLI 命令（create/switch/remove/list/history/export/diff/update-state/note/fleet-compat）
- [x] Fleet CLI（status/stop）
- [x] 6 个 MCP body/fleet tools
- [x] Dashboard `/api/body`、`/body`、WebSocket body 数据
- [x] `FleetCompatibilityAggregator`
- [x] 相关测试与文档
- [x] CI lint / mypy / test 全绿

### 5.2 本地临时分支

- `feat/body-docs-and-fail-closed`：已合并，可删除。
- `feat/body-p2-postmerge-verify`：本次验证临时分支，无独占提交，可删除。

### 5.3 需要 e-URDF 开发者确认的事项

- [ ] `EurdfProfile` 后续字段变更时，请同步更新 `EffectiveBodyCompiler` 的读取点。
- [ ] `rosclaw body init` 与 `rosclaw body create` 是否统一？建议后续由 e-URDF 侧主导决策。
- [ ] P0 MCP body tools（`get_body_profile`、`get_body_state` 等）是否已完整实现？如未实现，需继续由 e-URDF 侧补齐。
- [ ] `body.yaml` / `calibration.yaml` schema 如有调整，请同步更新 `tests/body/test_schema.py` 与 `EffectiveBodyCompiler`。

### 5.4 已知限制

- `fleet_skill_compatibility` 当前实时计算，body 数量多时可能较慢；后续如需缓存，应在 body 状态变更事件（`SENSE_BODY_UPDATED` 或 registry 更新）时失效缓存。
- `switch_body` 仅修改 registry 指针，不会自动重新初始化 runtime/sense；若需要切换后自动加载，需 e-URDF/runtime 侧协同。
- 当前 `origin` remote 中的 PAT 已移除，但建议做一次全仓库 secret scanning。

---

## 6. 关键命令速查

### 6.1 验证 Body P2 功能

```bash
# 在最新 main 上
rosclaw body create --robot unitree-g1 --name g1-sim
rosclaw body list --json
rosclaw body switch g1-sim
rosclaw body fleet-compat --json
rosclaw fleet status --json
rosclaw fleet stop --reason "handover test"
```

### 6.2 运行相关测试

```bash
pytest tests/mcp/test_body_tools.py tests/dashboard/test_body_page.py \
       tests/body/test_fleet_compatibility.py tests/body/test_multi_body_registry.py \
       tests/body/test_end_to_end_workflow.py tests/mcp/test_e2e.py -v

# 全量
pytest -q

# Lint / Type
ruff check .
mypy --config-file .github/mypy-ci.ini src/rosclaw/mcp/adapters src/rosclaw/core/runtime.py src/rosclaw/cli.py src/rosclaw/body src/rosclaw/firstboot src/rosclaw/hub
```

### 6.3 查看 Dashboard

```bash
rosclaw dashboard start --port 8765
# 浏览器访问 http://localhost:8765/body
# API http://localhost:8765/api/body
```

---

## 7. 后续工作建议

1. **e-URDF 侧继续**（由另一位开发者负责）
   - 完善 `rosclaw body init` 的渲染输出，确保符合 `rosclaw_body.md` 的 P0 模板。
   - 实现/补齐 P0 MCP tools：`get_body_profile`、`get_body_state`、`list_body_capabilities`、`query_body`、`validate_body_action`、`get_calibration_status`。
   - 丰富 `e-urdf-zoo/` 中各机器人的 profile。

2. **Body P2 后续可选优化**
   - 为 `fleet_skill_compatibility` 增加缓存与增量更新。
   - 为 `switch_body` 增加 runtime/sense 重新加载 hook（需与 runtime 团队协同）。
   - 扩展 dashboard body 页面，支持历史曲线和故障时间线。

3. **工程治理**
   - 删除已合并的远程分支 `feat/body-docs-and-fail-closed`。
   - 运行 GitHub secret scanning，确认无 PAT 泄漏。
   - 考虑引入 `pytest-xdist` 缩短本地全量测试时间。

---

## 8. 联系与上下文

- 本报告对应工作已合并到 `main`（commit `aea131db` 及后续）。
- 如对本报告内容有疑问，优先查看：
  - `docs/body/FLEET_OPERATIONS.md`
  - `docs/body/BODY_REGISTRY.md`
  - `docs/help/body-p2-mcp-dashboard-fleet-help.md`
- 计划原文：`/home/ubuntu/.claude/plans/lexical-tinkering-dawn.md`

---

*本文档为本地移交文件，由 Claude Code 生成。如需提交到仓库，可将其纳入 `docs/help/` 目录并随后续 PR 合并。*
