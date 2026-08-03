# Pi attribution (ADR-0008)

ROSClaw depends on two npm packages from the Pi project (MIT License,
Copyright (c) 2025 Mario Zechner):

| Package | Version (exact pin) | Upstream commit | Use |
|---|---|---|---|
| `@earendil-works/pi-tui` | 0.83.0 | 0e633790c5a007f6d4bf35ba67ced457287c25ac | rosclaw-tui 终端组件（editor/markdown/select/loader/IME/CJK） |
| `@earendil-works/pi-ai` | 0.83.0 | 0e633790c5a007f6d4bf35ba67ced457287c25ac | rosclaw-modeld Provider 目录与流式调用（批次 D 引入） |

- 只使用包公开 exports；不 import 私有内部路径。
- `pi-coding-agent` / `pi-agent-core` / Hermes / OpenCode 的 Agent 运行时
  均被 ADR-0008 禁止进入生产路径（见
  `tests/architecture/test_pi_dependency_boundary.py`）。
- `packages/rosclaw-tui/package-lock.json` 已提交，升级依赖前必须先跑
  upstream characterization tests。
- LICENSE 原文见同目录 `LICENSE`。
