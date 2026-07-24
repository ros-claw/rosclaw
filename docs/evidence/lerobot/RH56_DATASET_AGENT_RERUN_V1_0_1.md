# Dataset Agent 重跑证据（v1.0.1 清单 §6）

- 日期： 2026-07-23
- 运行： `scripts/acceptance/lerobot_agent_blackbox.sh --agent claude --mode dataset`
- Agent： Claude Code 2.1.218（模型 qwen3.7-max），全新临时项目上下文，仅 MCP 工具
  （无 shell，forbidden scan 0 违规）
- 证据目录： `~/.local/state/rosclaw/evidence/lerobot_agent_blackbox/run_20260723_dataset/`

## 结论：PASS

| 验收项 | 结果 |
|---|---|
| Agent 识别 practice ID | ✅ `prac_20260723T155556Z_502b2f`（预置 fixture 会话） |
| verify --strict | ✅ 操作员按 Agent 指示运行 pinned CLI：`passed: true`，0 issues（重预置会话 `prac_20260723T161539Z_264bcc`，同构造） |
| export --format lerobot --profile physical | ✅ 按预期**诚实失败**：rollout 会话无 `physical_feedback_event`（Agent 预先从源码推断出这一点，与实际报错逐字一致） |
| 确定性 P2.1 导出 | ✅ `run_dataset_export(profile=physical, dataloader=True)`：`load_ok / index_ok / dataloader_ok = True`（`cli_export_deterministic.json`） |
| 不训练、不上传 | ✅ Agent 无相关动作（transcript 可查） |

## Agent 侧行为亮点

- MCP `practice_query` 查不到该会话时，Agent 正确定位原因（MCP 默认数据根 ≠
  `practice_demo`，需 `--data-root`）。
- 对照 `verifier.py` 做文件级审查，报告 3 处需注意的 fixture 痕迹（录制期 ID
  `prac_bb` 与束 ID 不一致、`end_time` 字段自相矛盾、manifest 引用的 mcap/artifact
  目录不在盘上）。严格验证器最终 0 issues —— Agent 的三点是更保守的文件级观察，
  已如实记录。
- Agent 无 shell（黑盒设计），明确拒绝手写 parquet 伪造"数据集"，将 CLI 步骤
  交还操作员 —— 与 §5 同一边界哲学：Agent 不越过自己的通道。
- 首次运行 20 min 超时（Agent 工具链过长被截断），第二次以 45 min 窗口完成；
  两次均零违规。

## 与 §5 的关系

清单规定 `agent_ready: true` 的前置是 §5（授权 REAL 黑盒）+ §6（本项）同时通过。
两者均于 2026-07-23 通过；黑盒性质均为 `developer_agent_blackbox,
independent: false`（不声明独立 H5）。
