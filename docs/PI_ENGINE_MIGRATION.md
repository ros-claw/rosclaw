# Pi Engine 迁移与切换指南（PNA-11）

日期：2026-08-05

## 当前阶段：Developer Preview（阶段 2）

`engine=pi` 已可用但**不是默认**。默认保持 `legacy`——完整 Product Gate
（PTY/IME 矩阵、产品旅程全自动 PTY、性能门槛实测）尚未全部通过，
REAL 门禁不变（关闭）。

## 切换方式

```bash
# 单次
rosclaw chat --engine pi

# 配置（Developer Preview）
# ~/.rosclaw/config.yaml
agent:
  engine: pi
```

回退：

```bash
rosclaw chat --legacy
```

优先级：`--legacy` > `--engine` > `config.agent.engine` > `legacy`。

## engine=pi 包含什么

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
- 删除 pi engine 不影响 Mission/Receipt；legacy 仍能读取 Mission；
- 新表（013–016）独立 migration，downgrade 不删 binding 数据。

## legacy 冻结说明

旧 `rosclaw-tui`（packages/rosclaw-tui）自 2026-08-05 起冻结：
只修安全或阻断 bug，不新增功能。`--engine legacy` 路径在 PNA-11
完成前保持完全可用。

## 切换默认 engine 的前置条件（规格 §31 阶段 3/4）

- [ ] 完整 Product Journey 全自动 PTY（clean install → chat → login →
  model → ask → observe → delegate → SIM action → approve → receipt →
  compact → exit → resume）；
- [ ] PTY/IME 矩阵（中文输入法、退格、粘贴、resize、tmux/SSH）；
- [ ] 性能门槛实测（§30 表格）；
- [ ] 100 次启动退出零孤儿；
- [ ] SIM E3 不回退；REAL 门禁不受影响。
