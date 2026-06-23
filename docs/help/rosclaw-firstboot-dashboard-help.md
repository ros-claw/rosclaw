# ROSClaw First Boot 产品化开发求助清单

> 生成时间：2026-06-24
> 当前分支：`feat/feedback-telemetry-v1`
> 上游 `origin/main` HEAD：基于 PR #28 合并后的 main
> 参考计划：`/home/ubuntu/.claude/plans/lazy-puzzling-dawn.md`

---

## 1. 本次完成的工作

在 PR #28（Dashboard firstboot page）已合并到 `main` 的基础上，继续实现 **Feedback / Telemetry v1** 模块，为 ROSClaw 提供可审计、可撤销的产品遥测与反馈上传能力。

### 1.1 新增/修改文件

| 文件 | 说明 |
|------|------|
| `src/rosclaw/feedback/` | 新增：遥测/反馈核心包（config、consent、installation、store、telemetry_client、upload、redactor、cli） |
| `src/rosclaw/firstboot/telemetry.py` | 重写：使用 `TelemetryConfig` / `FeedbackConfig` 生成 telemetry.yaml / feedback.yaml |
| `src/rosclaw/firstboot/wizard.py` | 修改：交互/非交互流程中增加 telemetry/diagnostics/rich-feedback 选项，并记录 `firstboot_completed` 事件 |
| `src/rosclaw/firstboot/workspace.py` | 修改：初始化 workspace 时创建 feedback / telemetry 目录 |
| `src/rosclaw/cli.py` | 修改：dispatch 接入 `telemetry_command_hook`，新增 `feedback` 子命令，firstboot 新增 `--diagnostics` / `--no-diagnostics` / `--rich-feedback` / `--no-rich-feedback` |
| `src/rosclaw/dashboard/firstboot.py` | 修改：dashboard 向导增加 diagnostics / rich-feedback 选项，默认 telemetry 关闭（offline 默认） |
| `tests/feedback/` | 新增：反馈/遥测单元测试 |
| `tests/test_firstboot.py` | 修改：补充 diagnostics / rich-feedback 参数 |
| `tests/dashboard/test_firstboot_page.py` | 修改：补充 dashboard 默认命令断言 |
| `supabase/migrations/004_telemetry_feedback_schema.sql` | 新增：后端遥测/反馈事件表 |

### 1.2 功能说明

- `rosclaw firstboot --yes --profile offline --no-telemetry --no-diagnostics --no-rich-feedback`：非交互式首次配置，offline 默认关闭遥测与反馈上传。
- `rosclaw feedback telemetry {on|off|status|ping|reset-id}`：用户可随时启用/禁用/测试/轮换匿名安装 ID。
- `rosclaw feedback consent --diagnostics/--rich-feedback/--revoke-all`：管理诊断摘要与富媒体反馈上传同意。
- `rosclaw feedback export --redact` / `rosclaw feedback upload --redact`：本地导出或手动上传经过 redaction 的反馈包。
- Telemetry 事件本地先写入文件，上传为 fire-and-forget；失败不影响本地使用。
- 所有上传均包含 redaction：不发送 hostname、username、local_path、prompt、log、mcap、api_key、robot_serial 等敏感字段。

### 1.3 验证结果

| 检查项 | 结果 |
|--------|------|
| `ruff check .` | ✅ 通过 |
| `PYTHONPATH=src python3 -m pytest tests -q --tb=short` | ✅ 3203 passed, 3 skipped |
| `PYTHONPATH=src python3 -m pytest tests/dashboard/test_firstboot_page.py tests/feedback tests/test_firstboot.py -q` | ✅ 通过 |
| `python3 scripts/integration_test.py` | ✅ 通过 |
| `ROSCLAW_DEV_SOURCE=1 bash scripts/acceptance_firstboot.sh` | ✅ 通过 |

---

## 2. 当前状态

- PR #28 已确认合并（`state: MERGED`，合并时间 2026-06-22）。
- Feedback / Telemetry v1 已在本分支实现并验证，**待提交 PR**。
- 本地改动包括已合并 PR #28 的 dashboard 页面与新增 feedback 模块，将以 `feat/feedback-telemetry-v1` 分支推送。

---

## 3. 已提交/已合并的 First Boot 相关 PR

| PR | 状态 | 说明 |
|----|------|------|
| #13 Body docs + fail-closed executor + firstboot | ✅ 已合并 | Phase 1/2 核心 |
| #17 MCP CLI tests / onboarding docs | ✅ 已合并 | 验证时 main HEAD |
| #20 P0 agent docs refresh | ✅ 已合并 | 文档 managed blocks |
| #22 Phase 7 sense wiring | ✅ 已合并 | body-sense 别名、skill executor gating |
| #28 Dashboard firstboot page | ✅ 已合并 | First Boot Phase 3 首个子项 |

本次新增改动（Feedback / Telemetry v1）尚未提交 PR。

---

## 4. Phase 3 剩余开发项（更新后）

- [x] **Dashboard firstboot page** — 已合并（PR #28）。
- [ ] **Feedback / Telemetry v1** — 已实现，待 PR 合并。
- [ ] **Homebrew tap**：需要单独仓库 `ros-claw/homebrew-tap`。
- [ ] **签名 release artifacts**：需要 GPG/代码签名密钥决策。
- [ ] ~~**Offline wheel bundle**~~ — **已决定不做**。
- [ ] **Cloud login**：需要后端 API 规范与 OAuth client。

---

## 5. 需要的外部支持 / 资源

| 需求 | 说明 |
|------|------|
| **PR 评审与合并** | 需要维护者 review `feat/feedback-telemetry-v1` PR。 |
| **macOS / Windows (WSL) 测试环境** | 当前仅在 Linux 验证。 |
| **Homebrew tap 仓库** | 需创建 `ros-claw/homebrew-tap`。 |
| **代码签名 / GPG 密钥** | Phase 3 签名 release 需要持有者决策。 |
| **Cloud login API 规范** | 若后续实现 cloud firstboot，需要后端接口。 |

---

## 6. 困惑 / 待确认的问题

1. **help 文档是否随 PR 一起提交？**
   本文件放在 `docs/help/` 下，用户此前要求“名字要包含你的工作关键词，放到 docs 目录下的help目录下”。是否应在 PR 中提交？还是仅作为本地辅助文件？

2. **是否继续 Phase 3 的下一项？**
   用户已明确 offline wheel bundle 不做。Phase 3 剩余项均需要外部资源：
   - Homebrew tap（需要仓库）
   - 签名 release artifacts（需要密钥）
   - Cloud login（需要 API 规范）
   请确认优先启动哪一项，或等资源到位后再继续。

3. **过时分支如何清理？**
   本地和远程存在大量已合并分支。是否可以批量删除？

4. **是否需要打 tag / release？**
   Feedback / Telemetry v1 是 P1 增强。是否要在 PR 合并后发布版本？

---

## 7. 下一步建议

1. **提交并推送 PR**：当前分支 `feat/feedback-telemetry-v1`，已验证 lint + 全量测试 + acceptance，建议 `git push -u origin feat/feedback-telemetry-v1` 并创建 PR。
2. **等待 review 并修复反馈**。
3. **PR 合并后**：Phase 3 剩余项均依赖外部资源，建议先确认优先级与可用资源，再启动下一项。
4. **分支清理**：在 PR 合并后统一删除已合并分支。

---

*本清单为本地辅助文档，用于跟踪当前模块开发状态与待确认事项。*
