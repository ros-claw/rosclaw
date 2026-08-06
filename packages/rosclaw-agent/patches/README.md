# ROSClaw 上游薄补丁（NA-FIX-3，规格 §5/§23.4）

目标包：`@earendil-works/pi-coding-agent@0.83.0`（commit 588915ec）。
预算：< 60 LOC。只加通用扩展点，不 fork AgentLoop/Provider/Session 格式。

## patch-01：AppIdentity / resumeCommandFormatter

- 位置：`dist/modes/interactive/interactive-mode.js` 的 `formatResumeCommand`
- 内容：退出 resume 提示输出 `rosclaw chat --resume <id>`（绝不输出
  `pi --session` / `--session-dir`——用户必须经 ROSClaw runtime 恢复：
  agentd kernel + SessionBinding + lease + ResourcePolicy + 工具表）。

## patch-02：BuiltinCommandPolicy（dispatch 前置）

- 位置：`defaultEditor.onSubmit` 内建命令 dispatch 前
- 内容：`globalThis.__rosclawBuiltinPolicy.disabled` 集合中的内建命令
  在任何内建 dispatch 之前被拦截（ROBOT profile 的 /trust /share
  /import /reload 强制语义——P0-8：input hook 在 dispatch 之后，单靠
  InputGuard 拦不住内建）。

应用方式：`npm install` 后 postinstall 自动执行
`node patches/apply-upstream-patches.mjs`。锚点文本在上游升级后若
漂移，补丁应用器**硬失败**（拒绝产出静默不生效的包）——升级 Pi 时
必须人工复核锚点并重新生成。
