# W09 十类本地任务验收层（规格 §13）

**分支**：a-w09-eval-cases（基于 main=06a8918b）
**日期**：2026-09-09

## 交付

- `tests/eval/cases/*.yaml`：L01–L10 十个 case 定义（§13.2
  schema：id/tier/fixture/variants/prompt_source/allowed_mode=
  SIM/allowed_outputs/oracle/limits/grading）。
- `tests/eval/test_w09_cases.py`：三层纪律执行器
  （pytest 即 runner——§13.2「合并到现有测试目录」）：
  - **契约层**：十 case schema 完整 + 全限 SIM（REAL 门禁）。
  - **物理/媒体层真跑**（真实 MuJoCo，无模型）：
    - L01 变体结构真值（nq≠nv≠nu，freejoint/ball/被动 vs
      执行器）+ 完整状态记录（W03 链接）；
    - L02 任意路径 rollout + 时间单调 + GIF 可解码 + receipt
      标 agent_generated（不冒充受信）；
    - L07 阻尼实验 3 条件×3 次、衰减率从数据重算、单调——
      **实证发现**：damping=0.8 过阻尼改变指标定义（§13.9
      警告成真），夹具参数改欠阻尼区（0.02/0.1/0.3）后真实通过；
    - L10 同 trace 双视角独立产出 + 重启（重读）不重 rollout。
  - **agent 层**（L03/L04/L05/L06/L08/L09）：无 ROSCLAW_KIMI_API_KEY
    逐例 skip NOT_RUN——不合成冒充；有 key 走 W10 真实驱动。

## 与既有资产的关系

simforge 联赛基准（contact_push 等）是训练/演化评测，本层是
**产品验收十类**——引用其 fixture 概念但不复用其评分器（评分器
与答案隔离原则）。L04/L06 的 agent 侧（模型写控制器）属 W10。

## 验证

| 项 | 结果 |
|---|---|
| test_w09_cases.py | 6 passed / 6 skipped(NOT_RUN) |
| L07 过阻尼实证 | damping 0.8 六秒峰值 <3（指标失效）→ 欠阻尼区通过 |
| ruff | 全过 |

## 边界

- §13.13 密封任务/30 次样本/分母报告：属发布前 W10 真实驱动
  轮的产物，本轮不立。
- 每类 ≥3 变化样本：L01 实现 2 变体 + L07 3 条件；其余在
  agent 层驱动时扩展（NOT_RUN 标注）。
