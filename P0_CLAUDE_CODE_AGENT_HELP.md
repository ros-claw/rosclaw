# ROSClaw P0 Claude Code Agent 集成 — 工作进展与 Help 清单

> 本地状态快照，用于对照实施指南 `rosclaw_agent接入优化.md` 检查剩余工作、问题与所需支持。
> 生成时间：2026-06-23

---

## 1. 我做了什么

1. **确认我推送的 P0 Agent PR 状态**
   - PR #14 (`feat/p0-claude-code-agent-followup`) — **已合并**到 `main`（merge commit `0dd3c3d`，2026-06-20）。
   - PR #19 (`fix(ci): add pytest-cov to dev dependencies`) — **已合并**到 `main`（merge commit `8ba5af9`，2026-06-20）。
   - PR #20 (`docs/p0-managed-blocks-refresh`) — **已合并**到 `main`（merge commit `7143cd3`，2026-06-21）。
   - 本地 `main` 已更新到最新（当前 HEAD 在 PR #37 之后）。

2. **清理错误文件**
   - 删除了之前误生成的 `HARDWARE_MCP_ONBOARDING_HELP.md`（它属于 Hardware MCP onboarding，不是 P0 Agent 工作）。

3. **在最新 `main` 上跑通 P0 Agent 相关 CI 关卡**
   - `ruff check src tests` — **All checks passed**。
   - `pytest tests/agent tests/mcp tests/security` — **162 passed, 1 skipped**。
   - 针对 P0 Agent/MCP 改动模块的 `mypy`：`mypy --config-file .github/mypy-ci.ini src/rosclaw/mcp/adapters src/rosclaw/agent src/rosclaw/mcp/schemas src/rosclaw/mcp/server.py src/rosclaw/mcp/tools/__init__.py` — **Success, no issues found in 27 source files**。
   - 修复了 mypy 在 `src/rosclaw/mcp/tools/__init__.py:65` 报 `has no attribute "__signature__"` 的问题：把 `wrapper` 显式 `cast(Any, ...)` 后再赋值 `__signature__/__annotations__/__name__`。

4. **本地手动验收 P0 Agent CLI**
   - `python3 -m rosclaw.cli agent init claude-code --dry-run` → 输出 5 个待生成文件路径，正常。
   - `python3 -m rosclaw.cli agent init claude-code` → 生成 `.mcp.json`、`CLAUDE.md`、`ROSCLAW.md`、`.claude/settings.json`、`.rosclaw/agent/context.snapshot.json`。
   - `python3 -m rosclaw.cli agent doctor claude-code` → onboarding files OK，expected tools 7。
   - `python3 -m rosclaw.cli agent test claude-code --quick` → 文件检查通过，advertised 7 tools。

5. **对照 `rosclaw_agent接入优化.md` 逐条核对 P0 DoD**
   - 已实现：CLI 入口、`rosclaw mcp serve`、7 个 P0 tool 函数、doctor/test 命令、文件生成、managed block 合并、JSON 合并/备份、审计日志、secrets scan、FastMCP 注册、fixture fallback。
   - 发现偏差：见第 3 节。

---

## 2. 当前状态

| 维度 | 状态 |
|------|------|
| PR #14/#19/#20 | 已合并到 `main` |
| 本地 lint | `ruff check src tests` 通过 |
| 本地 type check（聚焦模块） | `mypy` 通过 |
| 本地测试 | `tests/agent tests/mcp tests/security` 162 passed, 1 skipped |
| 手动 CLI smoke | `agent init/doctor/test claude-code` 通过 |
| 代码实现 | CLI、文件生成、MCP server、RuntimeClient facade、7 个 P0 tools、6 个 body tools、audit log、doctor/test 均已落地 |

---

## 3. 对照实施指南还缺少什么 / 偏差

### 3.1 `.mcp.json` 格式与指南不一致

**指南要求：**

```json
{
  "mcpServers": {
    "rosclaw": {
      "type": "stdio",
      "command": "rosclaw",
      "args": ["mcp", "serve", "--profile", "${ROSCLAW_PROFILE:-default}", "--project", "${PWD}"],
      "env": {
        "ROSCLAW_HOME": "${ROSCLAW_HOME:-~/.rosclaw}",
        "ROSCLAW_PROFILE": "${ROSCLAW_PROFILE:-default}",
        "ROSCLAW_AGENT_CLIENT": "claude-code",
        "ROSCLAW_MCP_AUDIT": "1"
      },
      "timeout": 300000
    }
  }
}
```

**当前实现：**

```json
{
  "version": "1.0.0",
  "servers": {
    "rosclaw-p0": {
      "command": "rosclaw-mcp-serve",
      "args": ["--transport", "stdio"],
      "env": {
        "ROSCLAW_PROJECT_ROOT": "...",
        "ROSCLAW_ROBOT_ID": ""
      }
    }
  },
  "rosclaw": { "schema_version": "p0.2025-06-19", ... }
}
```

**偏差：**

- 顶层键是 `servers`，不是 `mcpServers`。
- server 名是 `rosclaw-p0`，不是 `rosclaw`。
- 没有 `"type": "stdio"`。
- `command` 是独立 entrypoint `rosclaw-mcp-serve`，不是 `rosclaw` + args `mcp serve ...`。
- 缺少 `--profile`、`--project` 参数、环境变量 `ROSCLAW_AGENT_CLIENT`、`ROSCLAW_MCP_AUDIT`、`timeout`。
- 当前 `.mcp.json` 在 Claude Code 官方格式下可能无法被正确识别为项目级 MCP server。

### 3.2 MCP server / FastMCP 名称与指南不一致

- 当前 FastMCP name = `rosclaw-p0`；指南要求 `rosclaw`。
- 当前 entrypoint `rosclaw-mcp-serve` 存在；指南要求通过 `rosclaw mcp serve` 启动。

### 3.3 `P0_TOOLS` 混入了 P2 body 工具

- 当前 `P0_TOOLS` 包含 13 个工具：7 个 P0 + 6 个 body 工具（`list_bodies`、`get_body`、`switch_body`、`list_body_history`、`check_skill_compatibility`、`fleet_skill_compatibility`）。
- 指南明确 P0 只暴露 7 个工具；body 工具属于 P2/body 范畴。
- `tools/list` 目前返回 13 个，Claude Code 会发现超出 P0 范围的工具。
- 需要把 body 工具拆到独立 `BODY_TOOLS` 列表，由 body 相关 server 或后续 P2 server 注册。

### 3.4 工具输入 schema 与指南不一致

当前签名过度简化：

| 工具 | 当前签名 | 指南期望 |
|------|----------|----------|
| `get_robot_state` | `()` | `robot_id`, `include: ["all"]`, `max_age_ms: 1000` |
| `list_skills` | `(skill_type=None, full_ids=False)` | `robot_id`, `domain`, `verified_only`, `include_parameters`, `include_disabled` |
| `query_memory` | `(instruction, limit=5, outcome_filter=None)` | `query`, `robot_id`, `task_id`, `memory_type`, `top_k`, `min_confidence`, `time_range`, `include_raw` |
| `validate_trajectory` | `(trajectory: list[list[float]], safety_level="MODERATE")` | `robot_id`, `candidate: {type, frame_id, trajectory/skill_plan}`, `mode`, `constraints`, `current_state_required` |
| `sandbox_run` | `(joint_positions: list[float])` | `robot_id`, `scenario_id`, `candidate`, `duration_s`, `speedup`, `seed`, `record`, `return_artifacts` |
| `practice_query` | `(episode_id=None, limit=10)` | `query`, `robot_id`, `task_id`, `skill_id`, `modality`, `limit`, `include_artifacts` |
| `emergency_stop` | `(reason: str)` | `robot_id`, `scope`, `reason`, `source` |

这会导致：

- FastMCP 暴露的 `inputSchema` 与指南定义的 JSON Schema 不符。
- Agent 调用时无法传入指南推荐的参数（如 `max_age_ms`、`domain`、`candidate` 结构体）。
- 测试用例（如 `tests/mcp/tools/test_p0_smoke.py`）使用的是当前简化参数，需要同步更新。

### 3.5 公共 envelope schema version 不一致

- 当前 `SCHEMA_VERSION = "p0.2025-06-19"`。
- 指南要求 `"rosclaw.mcp.v1"`。
- 测试（`test_p0_smoke.py`）断言 `schema_version.startswith("p0.")`，需同步改。

### 3.6 审计日志字段不完整

指南要求每行包含：

```json
{
  "trace_id": "uuid",
  "timestamp": "...",
  "agent_client": "claude-code",
  "project_root": "...",
  "runtime_profile": "...",
  "tool": "...",
  "input_redacted": {},
  "ok": true,
  "latency_ms": 123,
  "safety_level": "S0_READ_ONLY"
}
```

当前实现只记录：

```json
{
  "trace_id": "...",
  "timestamp": "...",
  "tool": "...",
  "arguments": {...},
  "ok": true
}
```

缺少：`agent_client`、`project_root`、`runtime_profile`、`latency_ms`、`safety_level`，且字段名是 `arguments` 而不是 `input_redacted`。

### 3.7 业务错误未显式设置 MCP `isError: true`

- 指南要求业务错误返回 MCP protocol 的 `isError: true`。
- 当前工具 wrapper 只是把 envelope 序列化为 JSON 字符串返回，没有调用 FastMCP 的错误返回方式。需要确认 FastMCP 版本下如何设置 `isError`（FastMCP 2.x/3.x 行为不同，已有 `ros-mcp-fastmcp-version-agnostic-fixture.md` 的处理经验）。

### 3.8 `CLAUDE.md` / `ROSCLAW.md` managed block 边界与指南不一致

- 当前使用 `<!-- ROSCLAW-MANAGED-BEGIN -->` / `<!-- ROSCLAW-MANAGED-END -->`。
- 指南要求：
  - CLAUDE.md: `<!-- ROSCLAW:BEGIN CLAUDE-CODE-CONTEXT -->` / `<!-- ROSCLAW:END CLAUDE-CODE-CONTEXT -->`
  - ROSCLAW.md: `<!-- ROSCLAW:BEGIN GENERATED -->` / `<!-- ROSCLAW:END GENERATED -->`
- 内容结构也与指南完整模板有差距（例如缺少 Required Startup Behavior、Available ROSClaw MCP Tools、Safety Rules、Recommended Workflows 等完整章节）。

### 3.9 `.claude/settings.json` 格式与指南不一致

- 当前生成 `version`、`rosclaw`、`permissions.deny`、`autoMemoryEnabled`。
- 指南要求：包含 `$schema`、明确 deny secrets 路径、推荐 allowed MCP server、推荐 hooks、推荐 telemetry trace env，并且 `permissions.deny` 里要包含直接 ROS 发布命令等。

### 3.10 `context.snapshot.json` 结构与指南不一致

- 当前 `schema_version` 是 `p0.2025-06-19`。
- 当前顶层结构是 `project` / `runtime` / `tools` / `policies`。
- 指南要求 `schema_version: "rosclaw.agent.context.v1"`，并包含 `project_root`、`runtime_profile`、`rosclaw_home`、`robot`、`services`、`mcp` 等字段。

### 3.11 CLI 参数与指南有差距

- 当前 `agent init claude-code` 缺少指南中的 `--server-name`、`--server-url`、`--force`、`--include-settings`、`--include-snapshot`。
- `--check` 当前只是加了一个 `check_mode` flag，没有真正执行指南列的 4 项验证（JSON schema、list_tools、get_robot_state smoke、CLAUDE.md import 检查）。
- `agent doctor claude-code` 只检查文件存在和工具数量，没有完整跑指南列的 16 项检查。
- `agent test claude-code` 没有分层 Level 0-5 和性能测试入口。

### 3.12 输出 envelope 缺少 `runtime_profile` 字符串

- 指南要求 `runtime_profile` 是字符串（如 `"default"`）。
- 当前 `make_response` 中 `runtime_profile` 默认 `{}`（dict），且工具 wrapper 没有传入实际 profile。

---

## 4. 需要什么 / 外部支持

| 需求 | 说明 |
|------|------|
| 产品/架构决策 | body 工具（`list_bodies` 等 6 个）是立即拆出 `P0_TOOLS`，还是保留在 P0 但更新指南？按 `rosclaw_agent接入优化.md` 应拆到 P2。 |
| `.mcp.json` 格式决策 | 是否严格对齐 Claude Code 官方项目级 MCP 格式（`mcpServers` / `type` / `command` / `args` / `env`）？这是自动加载的关键。 |
| FastMCP `isError` 语义 | 需要确认当前项目依赖的 FastMCP 大版本（2.x vs 3.x），因为设置 `isError: true` 的 API 不同。 |
| 工具 schema 对齐优先级 | 把 7 个工具的输入参数补齐到指南 JSON Schema 是一项较大改动，需确认是否在本次 follow-up PR 中全部完成，还是分阶段（先结构、再字段）。 |
| 测试预期更新 | 拆分 body tools / 改 schema version / 改 `.mcp.json` 都会动到现有测试和 golden files，需要同步更新。 |
| 全局 `rosclaw` entrypoint | 当前环境 PATH 中的 `rosclaw` 来自旧安装，直接运行 `rosclaw agent init` 可能报错；需要 `pip install -e .` 或确认 CI 使用 `python3 -m rosclaw.cli`。 |

---

## 5. 困惑 / 待确认点

1. **P0 vs P2 工具边界**
   - 代码和 `CLAUDE.md` managed block 已经把 6 个 body 工具视为可用，但实施指南明确 P0 只有 7 个。这里应该服从指南还是更新指南？用户已明确“实施指南是 `rosclaw_agent接入优化.md`”，所以应把 body 工具从 P0 server 拆出去。

2. **`.mcp.json` 官方格式 vs 自定义格式**
   - 当前格式（`servers.rosclaw-p0` + `rosclaw` 元数据）可能是为内部 dashboard 设计的。但指南按 Claude Code 官方约定使用 `mcpServers`。需要统一。

3. **FastMCP 版本**
   - `pyproject.toml` 里 `mcp` 依赖版本是多少？需要确认才能写正确的 `isError` 处理和测试 fixture。

4. **schema version 命名**
   - 当前 `p0.2025-06-19` 是内部临时名。指南要求 `rosclaw.mcp.v1`。是否直接全局替换，还是保留向后兼容？建议直接对齐指南。

5. **审计日志路径里的 `~/.rosclaw`**
   - 当前硬编码 `Path.home() / ".rosclaw" / "logs" / "mcp"`。是否需要尊重 `ROSCLAW_HOME` 环境变量？

---

## 6. 下一步建议

1. **立即修复低争议偏差**（适合一个 focused follow-up PR）：
   - 把 `SCHEMA_VERSION` 改为 `"rosclaw.mcp.v1"`。
   - 拆分 `P0_TOOLS` / `BODY_TOOLS`；server 只注册 P0 工具；保留 body 工具导出供测试/P2 使用。
   - 补齐审计日志字段：`agent_client`、`project_root`、`runtime_profile`、`input_redacted`、`latency_ms`、`safety_level`。
   - 把 `.mcp.json` 对齐到 `mcpServers.rosclaw` + `type` + `command` + `args` + `env` + `timeout`。
   - 同步修改 `doctor.py` / `test_claude_code.py` / `validate.py` 里对 `rosclaw-p0` / `servers` 的硬编码检查。
   - 更新 `tests/mcp/test_server.py`、`tests/mcp/tools/test_p0_smoke.py`、`tests/agent/test_init_claude_code.py` 的预期。

2. **第二批次：工具输入 schema 对齐**（工作量更大，可单独 PR）：
   - 按指南 JSON Schema 重新定义 7 个工具的函数签名。
   - 更新 `RuntimeClient` 方法签名和 adapters 的调用方式。
   - 更新所有相关测试和 fixture。

3. **第三批次：上下文文件模板对齐**（文档/Polish PR）：
   - `CLAUDE.md` / `ROSCLAW.md` 使用指南指定的 managed block 标记和完整章节。
   - `.claude/settings.json` 对齐指南格式。
   - `context.snapshot.json` 对齐指南结构。

4. **验证与提交**
   - 每批次都跑 `ruff check src tests`、`mypy` 聚焦模块、`pytest tests/agent tests/mcp tests/security`。
   - 在 temp dir 跑 `rosclaw agent init claude-code --check`、`doctor`、`test --quick`。
   - 分支 → commit → push → 开 follow-up PR。

---

## 7. 已验证命令

```bash
cd /home/ubuntu/rosclaw/rosclaw/rosclaw-v1.0-hub
ruff check src tests
pytest tests/agent tests/mcp tests/security -q
mypy --config-file .github/mypy-ci.ini \
  src/rosclaw/mcp/adapters \
  src/rosclaw/agent \
  src/rosclaw/mcp/schemas \
  src/rosclaw/mcp/server.py \
  src/rosclaw/mcp/tools/__init__.py

# 手动验收（使用 python3 -m rosclaw.cli，因为全局 rosclaw entrypoint 可能不是最新）
python3 -m rosclaw.cli agent init claude-code --dry-run
python3 -m rosclaw.cli agent init claude-code
python3 -m rosclaw.cli agent doctor claude-code
python3 -m rosclaw.cli agent test claude-code --quick
```

当前结果：lint / type / test 全绿；手动 CLI 正常；但与 `rosclaw_agent接入优化.md` 在 `.mcp.json`、工具集合、schema version、审计日志、工具输入 schema、上下文模板等方面存在系统偏差，需要 follow-up 修复。
