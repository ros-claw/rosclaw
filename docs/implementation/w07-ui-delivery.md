# W07 用户界面、路径和远程交付（规格 §11）

**分支**：a-w07-ui-delivery（基于 main=06a8918b，§17 依赖：W07←W04–W06 已合入）
**日期**：2026-09-09

## 现状盘点（保留已有可用实现）

- `rosclaw artifact list/show/path/open/export` + `rosclaw open`
  重写（同一 handler，无第二实现）——0901 P0-1 / 0902 R3-a 已建；
  安装产物可达性已有 journey 硬 Gate（list/open 不回落顶层帮助）。
- 无显示环境 open → 绝对路径 + 导出提示（OSC 8 不假装把远程
  文件带到本机）；桌面 xdg-open 只说「已交给系统默认程序」
  （不声称用户已看过）。
- §11.1 界面减法（只显示目标/进度/结果）已由大道至简 R2 落地。

## 发现的缺口（已修）

- `cmd_artifact_export` 目标已存在时**静默覆盖**（与 §6.4
  api.export 的 EXPORT_TARGET_EXISTS 纪律矛盾——导出是证据链
  动作）：改为拒绝并提示（rc=4）。

## 新增测试（tests/agentd/test_w07_ui_delivery.py，6 例）

真实 handler + 真实账本（MissionStore DB + TaskKernel 登记）经
`entrypoint._dispatch` 全链：
- help 列五个子命令；
- 合法 ID 无显示环境 → 绝对路径 + 导出提示，不声称已打开；
- 未知 ID rc=2；登记后文件被删 rc=3（诚实报缺失）；
- 导出内容一致 + 目标存在拒绝静默覆盖（红→绿：基线静默覆盖）；
- `rosclaw open <id>` 重写可达（同一 handler）。

## 验证

| 测试 | 结果 |
|---|---|
| test_w07_ui_delivery.py | 红 1（静默覆盖）→ 绿 6/6 |
| tests/cli + root_cli + a0902_r3a + a0901_p02 + a0902_review | 53 全过 |
| ruff check src tests | 全过 |

## 边界说明

- 多会话歧义/`latest` 作用域：legacy `explain` 的 run_reference
  是旧 CLI 面；产品面 artifact 命令以 artifact_id 为准（无
  latest 猜测语义需要修）。
- 真实模型 journey：**NOT_RUN**（无 key，不合成冒充）。
