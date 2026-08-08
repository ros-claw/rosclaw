# Native Agent 迁移说明（原 Pi Engine 迁移指南，NA-FIX-9 后）

日期：2026-08-06（取代 2026-08-05 Developer Preview 版）

## 当前阶段：Native Agent 为默认（阶段 3/4 已完成）

`rosclaw chat` 默认就是 Native Agent（Pi-backed harness）。公开界面
不再暴露 "engine" 概念——`--engine` / `--legacy` 均不在 help 中显示。
REAL 门禁不变（关闭；SHADOW/REAL 继续走独立安全 Gate）。

## 使用

```bash
rosclaw chat            # Native Agent（默认）
rosclaw chat --continue # 恢复最近一次会话
rosclaw chat --resume <session-id>
```

临时回退（隐藏、保留一个稳定版本后随 legacy 一起退役）：

```bash
rosclaw chat --legacy
```

优先级：`--legacy` > `--engine`（隐藏）> `config.agent.engine` > `pi`。

## Native Agent 包含什么

- Pi InteractiveMode（成熟 TUI：流式、working 动画、/model /login
  /compact /resume /fork /tree 等内建命令）；
- ROSClaw 品牌 header + 内联扩展（`!` bash 关闭、项目资源锁定）；
- 每轮具身上下文注入（stale 禁止动作）；
- rosclaw_* 工具：status / observe / verify / memory_query /
  fail_safe / delegate / request_action；
- ApprovalCard（Y/N/Esc，operatord 签名链）；
- SessionBinding + writer lease；认知事件 hash 镜像；
- Provider：Pi ModelRuntime（developer 文件凭据 / robot env-only；
  config.yaml 自动迁移）。

## 数据兼容（规格 §31 数据回滚）

- Mission DB 不迁移为 Pi 格式；Pi Session 是附加认知存储；
- legacy 回退仍能读取 Mission/Receipt；
- 新表（013–016）独立 migration，downgrade 不删 binding 数据。

## legacy 冻结与退役路线

旧 `rosclaw-tui`（packages/rosclaw-tui）自 2026-08-05 起冻结：
只修安全或阻断 bug，不新增功能。`--legacy` 隐藏回退保留一个稳定
版本；其后删除旧 TUI/model loop，保留数据迁移读取器（规格 §31）。

## 默认切换的前置条件（全部已于 2026-08-06 完成，证据见
## tests/agentd/test_product_journey.py 等 Gate E 测试）

- [x] 完整 Product Journey 全自动 PTY（clean install → chat →
  delegate → SIM action → approve → receipt → compact → exit →
  resume）——`tests/agentd/test_product_journey.py`；
- [x] PTY/IME 矩阵（中文输入、CJK 退格、bracketed paste、resize、
  tmux）——`tests/agentd/test_tui_ime.py`、`test_tmux_env.py`；
- [x] 性能门槛实测（§30：回显 p95=15.8ms、working 可见 p95=25.3ms、
  redraw p95=16.8ms、idle CPU≈0）——`test_interaction_perf.py`；
- [x] 100 次启动退出零孤儿、RSS/fd 趋势平坦——`test_start_exit_soak.py`；
- [x] SIM E3 不回退；REAL 门禁不受影响（仍为关闭状态）。
