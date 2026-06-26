# Body / e-URDF / Agent View 统一审计报告

> 文件名关键词：`body-eurdf-unification-audit`  
> 生成时间：2026-06-26  
> 关联文档：`rosclaw_eurdf_body合并开发.md`、`rosclaw_eurdf优化.md`、`rosclaw_body.md`、`docs/help/body-p2-implementation-report-and-handover.md`

---

## 1. 审计目的

根据 `rosclaw_eurdf_body合并开发.md` Phase A 要求，梳理当前代码中 Body、e-URDF、Agent View、MCP Tools、Dashboard 相关文件，标记重复实现与边界混乱点，为后续统一实施提供基线。

---

## 2. 文件清单

### 2.1 Body Core 文件

| 文件 | 职责 | 状态 |
|------|------|------|
| `src/rosclaw/body/__init__.py` | 导出公共 API | ✅ 保留 |
| `src/rosclaw/body/schema.py` | Body/e-URDF/EffectiveBody 数据模型 | ✅ 保留 |
| `src/rosclaw/body/references.py` | `rosclaw://` URI 解析 | ✅ 保留 |
| `src/rosclaw/body/resolver.py` | `BodyResolver`：统一加载、编译、渲染入口 | ✅ 保留 |
| `src/rosclaw/body/compiler.py` | `EffectiveBodyCompiler` | ✅ 保留 |
| `src/rosclaw/body/renderer.py` | `EMBODIMENT.md` 渲染 | ✅ 保留 |
| `src/rosclaw/body/diff.py` | body diff | ✅ 保留 |
| `src/rosclaw/body/notes.py` | `maintenance.log` JSONL | ✅ 保留 |
| `src/rosclaw/body/validators.py` | update-state 路径校验 | ✅ 保留（建议改名 `patch_validator.py`） |
| `src/rosclaw/body/validator.py` | body workspace 完整校验 | ✅ 保留（建议改名 `workspace_validator.py`） |
| `src/rosclaw/body/compatibility.py` | skill compatibility | ✅ 保留 |
| `src/rosclaw/body/registry.py` | 多 body registry | ✅ 保留（P2） |
| `src/rosclaw/body/fleet.py` | fleet compatibility | ✅ 保留（P2） |
| `src/rosclaw/body/query.py` | body query engine | ✅ 保留 |
| `src/rosclaw/body/cli.py` | body CLI | ✅ 保留，需统一入口 |
| `src/rosclaw/body/safety.py` | safety invariant engine | ✅ 保留 |
| `src/rosclaw/body/ros_introspection.py` | `--from-ros` introspection | ✅ 保留 |
| `src/rosclaw/body/README.md` | body 模块说明 | ✅ 保留 |

**新增文件建议**：
- `src/rosclaw/body/service.py`：统一 `init`/`create`/`link-eurdf` 的 `BodyInstanceService`
- `src/rosclaw/body/agent_view.py`：Agent View 渲染逻辑抽取
- `src/rosclaw/body/summaries.py`：generated summaries 生成逻辑抽取
- `src/rosclaw/body/mcp_tools.py`：P0 MCP body tools 实现（如较复杂）
- `src/rosclaw/body/fleet_cache.py`：fleet compatibility 缓存
- `src/rosclaw/body/hooks.py`：`switch_body` runtime hooks

### 2.2 e-URDF 文件

| 文件 / 目录 | 职责 | 状态 |
|------------|------|------|
| `e-urdf-zoo/` | 各机器人 e-URDF profile | ✅ 保留 |
| `src/rosclaw/eurdf/loader.py` | `EurdfLoader` | ✅ 保留 |
| `src/rosclaw/eurdf/models.py` | `EurdfProfile` 等模型 | ✅ 保留 |
| `src/rosclaw/eurdf/registry.py` | e-URDF registry | ✅ 保留 |
| `src/rosclaw/runtime/eurdf_loader.py` | Runtime 层加载 | ✅ 保留 |

### 2.3 Agent View 文件

| 文件 | 职责 | 状态 |
|------|------|------|
| `src/rosclaw/body/renderer.py` | EMBODIMENT.md 渲染 | ✅ 保留 |
| `src/rosclaw/body/query.py` | body query | ✅ 保留 |
| `src/rosclaw/body/validator.py` | 校验 EMBODIMENT.md / BODY.md | ✅ 保留 |
| `docs/body/EMBODIMENT_FORMAT.md` | EMBODIMENT.md 格式说明 | ✅ 保留 |
| `docs/body/MIGRATION.md` | 迁移指南 | ✅ 保留 |
| `docs/body/TESTING.md` | body 测试指南 | ✅ 保留 |

### 2.4 MCP Body Tools 文件

| 文件 | 职责 | 状态 |
|------|------|------|
| `src/rosclaw/mcp/adapters/runtime_client.py` | RuntimeClient body tools方法 | ✅ P2 已添加，P0 待补齐 |
| `src/rosclaw/mcp/tools/__init__.py` | MCP tool 注册 | ✅ P2 tools 已注册，P0 tools 未注册 |
| `tests/mcp/test_body_tools.py` | MCP body tools 测试 | ✅ P2 测试存在，P0 测试待添加 |

### 2.5 Dashboard Body 文件

| 文件 | 职责 | 状态 |
|------|------|------|
| `src/rosclaw/dashboard/server.py` | `get_body_summary()`、snapshot | ✅ 保留（P2） |
| `src/rosclaw/dashboard/web_server.py` | `/api/body`、`/body` 页面 | ✅ 保留（P2） |
| `tests/dashboard/test_body_page.py` | Dashboard body 页面测试 | ✅ 保留（P2） |

---

## 3. 当前命令清单

### Body CLI（已存在）

```text
rosclaw body init --robot unitree-g1
rosclaw body create --robot unitree-g1 --name g1-sim
rosclaw body switch <body_id>
rosclaw body remove <body_id>
rosclaw body list [--json]
rosclaw body validate [--json]
rosclaw body render
rosclaw body show [--agent]
rosclaw body state [--json]
rosclaw body query "..."
rosclaw body fault ...
rosclaw body maintenance ...
rosclaw body calibration ...
rosclaw body retrofit ...
rosclaw body capability ...
rosclaw body link-eurdf unitree-g1
rosclaw body inspect
rosclaw body diff
rosclaw body update-state ...
rosclaw body note ...
rosclaw body history
rosclaw body export
rosclaw body fleet-compat
```

### Fleet CLI（已存在）

```text
rosclaw fleet status [--json]
rosclaw fleet stop --reason "..."
```

### 命令问题

- `init` / `create` / `link-eurdf` 当前在 `cli.py` 中各自实现，需要统一收敛到 `BodyInstanceService`。

---

## 4. 重复实现 / 边界混乱点

### 4.1 `validators.py` 与 `validator.py` 命名重复

**问题**：两个文件名称相近，容易混淆。

**当前职责**：
- `validators.py`：`validate_update_path()`、`parse_set_expression()`，负责 update-state 路径校验；
- `validator.py`：`BodyValidator` 类，负责完整 workspace 校验、EMBODIMENT.md 检查、安全 invariant。

**建议**：
- `validators.py` → `patch_validator.py`
- `validator.py` → `workspace_validator.py`

### 4.2 `init` / `create` / `link-eurdf` 三个入口未统一

**问题**：实施文档指出三个命令存在分叉风险，应统一内部服务。

**当前状态**：各自在 `cli.py` 中实现。

**建议**：新增 `src/rosclaw/body/service.py`，所有命令调用 `BodyInstanceService.create_or_init()`。

### 4.3 P0 MCP Body Tools 缺失

**问题**：`rosclaw_body.md` 与 `rosclaw_eurdf_body合并开发.md` 要求的 6 个 P0 tools 未实现：

| P0 Tool | 当前实现 | 缺口 |
|---------|---------|------|
| `get_body_profile` | ❌ 无 | 需新增 |
| `get_body_state` | ⚠️ `_get_robot_state` 部分覆盖 | 需确认是否等价，可能需增强 |
| `list_body_capabilities` | ❌ 无 | 需新增 |
| `query_body` | ❌ 无（`body query` CLI 存在但无 MCP tool） | 需新增 |
| `validate_body_action` | ❌ 无 | 需新增 |
| `get_calibration_status` | ❌ 无 | 需新增 |

### 4.4 Agent View 未完全抽取

**问题**：`EMBODIMENT.md` 渲染、generated summaries、agent query 分散在 `renderer.py`、`resolver.py`、`query.py`。

**建议**：按 `rosclaw_eurdf_body合并开发.md` 建议抽取：
- `src/rosclaw/body/agent_view.py`
- `src/rosclaw/body/summaries.py`

但当前代码功能已存在，属于代码组织优化，非阻塞。

### 4.5 跨模块 adapter stub 缺失

**问题**：实施文档要求 sandbox/provider/skill/memory/dashboard 通过 `EffectiveBody` 消费 body 数据，但各模块测试目录缺少 adapter stub 验证。

**当前状态**：仅在 `tests/body/test_cross_module_references.py` 有通用 StubAdapter。

**建议**：在 `tests/sandbox/`、`tests/provider/`、`tests/skill_manager/`、`tests/memory/`、`tests/dashboard/` 各加 `test_body_reference_contract.py`。

### 4.6 Provider / Sandbox / Memory 集成未实现

**问题**：Phase G 要求：
- `src/rosclaw/provider/body_binder.py`
- `src/rosclaw/sandbox/body_adapter.py`
- `src/rosclaw/memory/body_events.py`

**当前状态**：这三个文件均不存在。

---

## 5. 保留 / 迁移 / 废弃建议

| 项目 | 决策 | 理由 |
|------|------|------|
| `src/rosclaw/body/` 全部文件 | 保留 | P0/P2 核心实现 |
| `src/rosclaw/body/service.py` | 新增 | 统一 `init`/`create`/`link-eurdf` |
| `src/rosclaw/body/agent_view.py` | 新增 | 抽取 Agent View 逻辑 |
| `src/rosclaw/body/summaries.py` | 新增 | 抽取 generated summaries 逻辑 |
| `validators.py` / `validator.py` | 重命名 | 避免混淆 |
| `src/rosclaw/provider/body_binder.py` | 新增 | Phase G |
| `src/rosclaw/sandbox/body_adapter.py` | 新增 | Phase G |
| `src/rosclaw/memory/body_events.py` | 新增 | Phase G |
| P0 MCP body tools | 新增 | Phase E |
| 跨模块 contract tests | 新增 | Phase F |

---

## 6. 当前测试清单

### Body 测试

```text
tests/body/test_schema.py
tests/body/test_link_eurdf.py
tests/body/test_inspect.py
tests/body/test_effective_body.py
tests/body/test_diff.py
tests/body/test_update_state.py
tests/body/test_note.py
tests/body/test_skill_compatibility.py
tests/body/test_cross_module_references.py
tests/body/test_capability_management.py
tests/body/test_end_to_end_workflow.py
tests/body/test_fault_lifecycle.py
tests/body/test_fleet_compatibility.py
tests/body/test_history_export.py
tests/body/test_incremental_skill_recheck.py
tests/body/test_list.py
tests/body/test_maintenance_calibration.py
tests/body/test_multi_body_registry.py
tests/body/test_update_state_from_ros.py
```

### MCP / Dashboard 测试

```text
tests/mcp/test_body_tools.py
tests/dashboard/test_body_page.py
tests/mcp/test_e2e.py
```

### Phase A 验收测试结果

```bash
pytest tests/body tests/mcp/test_body_tools.py tests/dashboard/test_body_page.py -q
```

结果：

```text
93 passed, 1 warning in 30.13s
```

✅ Phase A 验收通过。

---

## 7. 关键结论

1. **P0 本体闭环已实现大部分**：e-URDF → body.yaml → EffectiveBody → EMBODIMENT.md 链路已跑通。
2. **P2 多 body / fleet / dashboard 已落地**：由前一位工程师完成。
3. **主要缺口**：
   - `init`/`create`/`link-eurdf` 入口未统一；
   - P0 MCP body tools 未实现；
   - Provider / Sandbox / Memory body 集成未实现；
   - 跨模块 contract tests 未分散到各模块测试目录；
   - `validators.py`/`validator.py` 命名需整理。
4. **未发现独立“本体认知”模块**：没有 `rosclaw_cognition`、`rosclaw_embodiment_docs` 等独立模块，重复实现风险较低。

---

## 8. 下一步建议

按 `rosclaw_eurdf_body合并开发.md` 8 个阶段推进：

1. Phase B：统一 `init`/`create`/`link-eurdf`；
2. Phase C：整理 schema 与命名；
3. Phase D：抽取 Agent View / summaries；
4. Phase E：补齐 P0 MCP body tools；
5. Phase F：跨模块 contract tests；
6. Phase G：Provider / Sandbox / Memory 集成；
7. Phase H：Dashboard / Fleet / Runtime hooks / 文档收敛。
