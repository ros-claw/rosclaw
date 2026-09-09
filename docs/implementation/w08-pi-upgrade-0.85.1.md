# W08 Pi 0.83.0 → 0.85.1 升级（规格 §12）

**分支**：a-w08-pi-upgrade（基于 main=06a8918b）
**日期**：2026-09-09

## 升级流程（§12.1 落地）

`scripts/release/pi_upgrade_candidate.sh <version>`：专用工作目录
内 bump pins（deps+overrides 两段对齐）→ `npm install
--ignore-scripts` 重生成 lock（与补丁应用分开，失败可归因）→
补丁重放（锚点漂移硬失败）→ 构建 → TS 套件 → JSON 报告
（lock 摘要前后/逐步状态/回退命令）。真实模型 A/B 标 NOT_RUN
（属 W10，不合成冒充）。

## 0.85.1 候选实测（一轮完整执行）

| 步骤 | 结果 |
|---|---|
| pin bump + lock 重生成 | ✅ lock sha256 e2c557e3→deeaa3cc |
| 补丁重放（0.85.1 dist） | ✅ patch-01/02 锚点全命中，retire 幂等 |
| 构建 | ❌→✅ 初始失败：`TUI` 在 0.85.1 变 type-only（拆分为接口 `TUI` + `TuiMainScreen`/`TuiAltScreen` 实现类） |
| 兼容适配 | `new TUI(terminal)` → `new TuiMainScreen(terminal)`（implements TUI，成员经 TuiBase 不变）：`packages/rosclaw-tui/src/app.ts`、`packages/rosclaw-agent/src/harness/pi/pi-picker.ts` |
| rosclaw-agent TS 套件 | ✅ 229 pass / 0 fail / 3 skip |
| rosclaw-tui TS 套件 | ✅ 27/27 |
| release bundle 构建（test_pna10_release，含 build-info） | ✅ 通过（含 wheel + Node bundle + SBOM 全链） |
| wheel journey（安装产物 PTY） | CI 门禁跑（本 PR 触发） |
| 真实模型 A/B | **NOT_RUN**（无 key——不合成冒充；W10 门禁） |

## 单源化改动（§12.1-6 manifest 诚实）

- `pi-upstream.lock.json` → 0.85.1 + tag commit d981de12 + 新
  lock 摘要（升级记录单源）。
- `build_release.sh` build-info 的 `pi_version`/`pi_commit` 从
  锁定记录读——不再写死 0.83.0（W00 基线标注的 W11 项顺手关闭）。
- pin 断言测试改为从单源读（test_pi_dependency_boundary /
  package-entry.test.ts / test_pna10_release）。
- 新增 `tests/test_w08_pi_upgrade.py`：pin 单源一致 + lock 摘要
  防漂移 + 不写死守卫 + 补丁钉版重放（无 node_modules 跳过）。

## 回退

`git checkout` 本 PR 涉及的 package.json/package-lock.json/
pi-upstream.lock.json/app.ts/pi-picker.ts 即回到 0.83.0（补丁
在 0.83.0 上同样可重放——W01 起既有性质）；会话/Artifact 不受
pin 影响（存储层无 Pi 版本耦合）。

## 边界

- §12.2 契约清单大部分由既有 journey/TS 套件覆盖（resume/
  compact/幂等/abort 等在 CI 门禁）；真实模型 continuation
  字段行为属 W10。
- 定时 CI 自动开 candidate PR（§12.1-1）属工具化增强，本轮
  交付可复现脚本 + 一轮实测；schedule 触发器留给后续。
