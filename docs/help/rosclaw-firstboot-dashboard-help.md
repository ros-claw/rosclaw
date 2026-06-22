# ROSClaw Dashboard First Boot 页面开发求助清单

> 生成时间：2026-06-22
> 当前分支：`verify-main`（基于 `origin/main` 准备 rebase）
> 上游 `origin/main` HEAD：`7e2dda4f`
> 参考计划：`/home/ubuntu/.claude/plans/lazy-puzzling-dawn.md`

---

## 1. 本次完成的工作

实现了 **Dashboard firstboot page**（First Boot Phase 3 子项），为本地 dashboard 提供可视化的首次配置向导。

### 1.1 新增/修改文件

| 文件 | 说明 |
|------|------|
| `src/rosclaw/dashboard/firstboot.py` | 新增：First Boot 状态读取、配置预览、CLI 命令生成、HTML 向导页面 |
| `src/rosclaw/dashboard/server.py` | 修改：`DashboardServer.get_firstboot_state()` 委托给新 helper |
| `src/rosclaw/dashboard/web_server.py` | 修改：新增 `/api/firstboot`、`/api/firstboot/preview`、`/firstboot` 路由 |
| `tests/dashboard/test_firstboot_page.py` | 新增：API 与页面测试 |

### 1.2 功能说明

- `/firstboot`：单页 HTML 向导，展示当前 First Boot 状态、可交互选择 profile/robot/safety/模块、实时生成 `rosclaw firstboot --yes ...` 命令。
- `/api/firstboot`：返回当前安装/工作区/配置状态，供页面刷新使用。
- `/api/firstboot/preview`：接收用户选择，返回配置预览和可复制命令；**只生成命令，不在服务端执行 firstboot**，符合安全契约。

### 1.3 验证结果

| 检查项 | 结果 |
|--------|------|
| `ruff check .` | ✅ 通过 |
| `PYTHONPATH=src python3 -m pytest tests/dashboard/test_firstboot_page.py tests/dashboard/test_body_page.py -v` | ✅ 10 passed |
| `PYTHONPATH=src python3 -m pytest tests -q --tb=short` | ✅ 3123 passed, 5 skipped |
| `python3 scripts/integration_test.py` | ✅ 通过 |
| `ROSCLAW_DEV_SOURCE=1 bash scripts/acceptance_firstboot.sh` | ✅ 通过 |

---

## 2. 当前状态

- `verify-main` 分支在本地保存了上述改动，但尚未 rebase 到最新的 `origin/main`。
- `origin/main` 自上次验证以来新增了 3 个 docs 提交（`706058f0`、`6a7cd599`、`7e2dda4f`），与本次 dashboard 改动无冲突。
- 下一步：**rebase 到 `origin/main` → 推送到 `feat/dashboard-firstboot-page` → 创建 PR**。

---

## 3. 已提交/已合并的 First Boot 相关 PR

| PR | 状态 | 说明 |
|----|------|------|
| #13 Body docs + fail-closed executor + firstboot | ✅ 已合并 | Phase 1/2 核心 |
| #17 MCP CLI tests / onboarding docs | ✅ 已合并 | 验证时 main HEAD |
| #20 P0 agent docs refresh | ✅ 已合并 | 文档 managed blocks |
| #21? / #22? Phase 7 sense wiring | ✅ 已合并 | body-sense 别名、skill executor gating |

本次新增改动尚未提交 PR。

---

## 4. Phase 3 剩余开发项（更新后）

- [x] **Dashboard firstboot page** — 已实现，待 PR 合并。
- [ ] **Homebrew tap**：需要单独仓库 `ros-claw/homebrew-tap`。
- [ ] **签名 release artifacts**：需要 GPG/代码签名密钥决策。
- [ ] ~~**Offline wheel bundle**~~ — **已决定不做**。
- [ ] **Cloud login**：需要后端 API 规范与 OAuth client。

---

## 5. 需要的外部支持 / 资源

| 需求 | 说明 |
|------|------|
| **PR 评审与合并** | 需要维护者 review `feat/dashboard-firstboot-page` PR。 |
| **macOS / Windows (WSL) 测试环境** | 当前仅在 Linux 验证 dashboard 页面。 |
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
   本地和远程存在大量已合并分支（`feat/body-docs-and-fail-closed`、`feat/body-p2-postmerge-verify`、`feat/p0-claude-code-agent-*` 等）。是否可以批量删除？

4. **是否需要打 tag / release？**
   First Boot P0 已完整合入 `main`，Dashboard 页面是 P1 增强。是否要在 Dashboard PR 合并后发布版本？

---

## 7. 下一步建议

1. **rebase + push PR**：`git rebase origin/main`，切到 `feat/dashboard-firstboot-page`，`git push -u origin`，创建 PR。
2. **等待 review 并修复反馈**。
3. **Dashboard PR 合并后**：Phase 3 剩余项均依赖外部资源，建议先确认优先级与可用资源，再启动下一项。
4. **分支清理**：在 PR 合并后统一删除已合并分支。

---

*本清单为本地辅助文档，用于跟踪当前模块开发状态与待确认事项。*
