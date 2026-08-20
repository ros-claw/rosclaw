# Phase 8 / C16：离散触球岛门控与微秒级边界闭环

日期：2026-08-09

边界：`SIM_ONLY`，CPU MuJoCo 动力学；4 张 A6000 仅用于并行进程隔离与 EGL，不宣称 GPU 物理加速、真实机器人效果或候选晋升。

## 1. 结论先行

本轮没有把连续 RL 继续盲目外推，而是先解决 C15 暴露的 Stability–Plasticity 问题：同一套动作只改变接触脉冲时长，球路会在极窄区域发生不连续跳变。

新增 ROSClaw 产品模块 `growth ballistic-contact-island-gate`，从严格 MuJoCo 回放中学习“可用的离散接触事件坐标”，在连续 actor 训练前拒绝跨岛插值。第一版单调阈值模型被新回放证伪后已废弃；最终实现改为保守的 replay-anchored event atlas，失败样本不会被成功均值抹平。

最终 gate：

- 训练数据：16 条同 Body、同实现、同实验上下文的严格回放；7 条 qualified、9 条 rejected；
- 事件特征：`lead_duration_sec`；
- 已知 qualified 坐标：`0.08 / 0.12 / 0.14 / 0.16 s`；
- `0.16 s` 邻域自动授权半径：`0.0001 s`，即保守使用约 `[0.1599, 0.1601] s`；
- 留一 balanced accuracy：`0.785714`；
- 留一 rejected recall：`1.0`，没有把已知失败误放行；
- 留一 qualified recall：`0.571429`，不足覆盖的成功岛宁可拒绝，不做乐观插值；
- `training_ready=true` 只表示可以作为岛内 actor 的前置门，不表示策略已晋升。

artifact：`/code/rosclaw/phase8_evidence/g1-growth-c16-contact-island-gate-v5.json`

gate hash：`sha256:a173676ae22565a29c5ea726ef6ff284fadd7f95c0516f48a57166e7728e21d9`

文件 SHA-256：`0cd801f72144d83623f8b9eb9d1ffefeba8aed02027e0462962631791f9d896f`；artifact 自身还由 canonical gate hash 绑定全部字段和 16 条源证据哈希。

## 2. 为什么连续 RL 会“越学越差”

C15 的初始样本看起来近似单调：

| lead | 门平面 z | 目标误差 |
|---:|---:|---:|
| 0.080 s | 0.7828 m | 0.5788 m |
| 0.120 s | 0.8518 m | 0.5061 m |
| 0.140 s | 0.8885 m | 0.4676 m |
| 0.160 s | 0.9562 m | 0.40525 m |
| 0.162 s | 0.3188 m | 1.0331 m |

只看这些点，线性或二次 critic 很容易推断“在 0.160 左侧都属于同一连续曲面”。4 卡临界复测直接否定了这个假设：

| lead | 门平面 y / z | 目标误差 | 结果 |
|---:|---:|---:|---|
| 0.1590 s | 0.9349 / 0.3156 m | 1.03641 m | 低球失败岛 |
| 0.1600 s | 0.9042 / 0.9562 m | 0.40525 m | 高球岛 |
| 0.1605 s | 0.9375 / 0.3160 m | 1.03589 m | 低球失败岛 |
| 0.1610 s | 0.9378 / 0.3169 m | 1.03493 m | 低球失败岛 |
| 0.1615 s | 0.9381 / 0.3179 m | 1.03400 m | 低球失败岛 |

这不是机器人摔倒造成的假象：所有条目均为零触球前停顿、零触球后后退、无跌倒。变化来自足—球接触几何与 20 ms 控制脉冲包络的混合动力学切换。对这种问题，连续 critic 的平滑外推是假设错误，不是“训练轮数还不够”。

## 3. 失败被怎样回灌

最初实现的单调 stump 在 12 条样本上得到训练 balanced accuracy `1.0`、留一 `0.9`，并错误预测 `0.159 / 0.1605 / 0.161` 属于 qualified 区域。未见临界回放全部否定该预测，因此：

1. 旧 gate artifact 没有接入 actor，也没有提交为最终能力；
2. 四条反例加入 replay；
3. 模型由“单一阈值的一侧”改成“成功事件坐标及极窄验证邻域”；
4. loader 会从 probes 重新计算特征、容差、支持域、训练指标和留一指标；即使篡改字段并重算普通 JSON hash，也会 fail-closed；
5. 输出继续硬顶 `SIM_ONLY`，无直接力矩、无 hot-swap、无 promotion、无 hardware authority。

另有 4 条临界回放曾误用了内部哈希为 `647588…` 的另一份 authority calibration。上下文检查发现训练集要求的是 `5e0b53…` 后，这 4 条被明确排除并用正确配置重跑，未污染 gate。

## 4. 微尺度 holdout

gate 训练后又运行 4 条不参与训练的微尺度 development holdout：

| lead | 门平面 y / z | 目标误差 | gate v5 判定 | 稳定性 |
|---:|---:|---:|---|---|
| 0.1599 s | 0.9041 / 0.9560 m | 0.40548 m | qualified 边界 | 零停顿、零后退、无跌倒 |
| 0.1601 s | 0.9042 / 0.9565 m | 0.40502 m | qualified 边界 | 同上 |
| 0.1602 s | 0.9042 / 0.9567 m | 0.40479 m | unseen，拒绝自动使用 | 同上 |
| 0.1603 s | 0.9043 / 0.9569 m | **0.40456 m** | unseen，拒绝自动使用 | 同上 |

`0.1603 s` 把 development 最佳误差从 `0.40525 m` 降到 `0.40456 m`，仅改善约 `0.17%`。它仍远未通过 `0.16 m` 严格精度门，而且位于自动授权带之外，所以只记录为下一轮扩岛候选，不更新冠军或宣传为精度突破。

这组 holdout 证明两点：

- gate 放行的 `[0.1599, 0.1601] s` 两端都得到真实物理回放支持；
- gate 对同样成功的 `0.1602/0.1603 s` 仍选择弃权，说明其目标是控制假阳性，而非尽可能扩大动作域。

## 5. ROSClaw 代码内容

新增 `ballistic_contact_island_gate.py`：

- 读取严格 replay evidence，并校验轨迹文件哈希；
- 强制物理接触 claim，检查足端速度、触球峰值、法向和世界系接触力数组；
- 将 JSON 中的触球峰值与 NPZ 实测峰值逐值绑定；
- 禁止混合 Body hash、implementation hash、实验上下文和重复 controls；
- 只从接触帧、lead、trail 三个事件特征中选择可解释轴；
- 保留成功与失败事件坐标，不跨过失败坐标做连续插值；
- 重新计算全部几何和留一指标，防止自报分数或边界被篡改；
- 输出明确的 `QUALIFIED_CONTACT_ISLAND / REJECTED_CONTACT_ISLAND / UNSEEN_CONTACT_EVENT / OUTSIDE_QUALIFIED_REPLAY_SUPPORT` 判定。

Growth CLI 新增：

```text
rosclaw growth ballistic-contact-island-gate \
  --evidence-json ... \
  --output /code/rosclaw/phase8_evidence/g1-growth-c16-contact-island-gate-v5.json \
  --source-checkout /code/rosclaw/rosclaw_phase8_exploration
```

CLI 只有 gate 满足类别数与保守留一门时才返回 0，否则返回 3。写出 artifact 不等于学习成功。

## 6. 本轮证据范围

- 新运行 20 条 MuJoCo 回放：16 条正确上下文，4 条配置串线后排除；
- gate 训练使用 16 条：12 条 C16 正确上下文新数据，加 4 条 C15 最终实现哈希上的 trail 对照；
- 另有 4 条正确上下文 micro holdout，不参与 gate 训练；
- 所有最终数据 implementation hash：`sha256:650473314cbd9bf23c46e7ac1f5111757809dcde8f1cb9579b5a06eb433fafeb`；
- 实验上下文 hash：`sha256:e1e44d374a6a37954a6eb66bcb997533a19047914c95bc9dfb7e025e9d87f793`。

## 7. 仍未完成与下一轮

1. 当前 gate 只解决“先选哪个接触事件岛”，尚未把它强制串入 actor-critic CLI；下一轮要让 actor proposal 必须携带并通过 gate hash。
2. 当前 7 个成功样本中有 3 个是孤立坐标，因此留一 qualified recall 只有 `57.14%`；要围绕 `0.08/0.12/0.14 s` 补邻域回放，而不是降低安全阈值。
3. 需要在最终实现上重新采集多关节 action probes；旧 actor 数据 implementation hash 不一致，不能复用。
4. 需要按 planner seed、球初速、球位置做分组 holdout。当前 seed 0 的窄岛不能声称跨初态泛化。
5. 精度仍约 `0.405 m`，严格 `0.16 m` 门未通过；下一步应在 gate 允许的时序岛内学习足端接触状态到球初速的逆模型，而不是继续扫描 lead。
6. 恢复稳定时间仍约 `3.422 s`；本轮保持零后退、零跌倒，但没有宣称恢复小脑取得新突破。

一句通俗解释：以前教练只看到“把腿再抬久一点，球会更高”，于是自然会继续加；现在发现地面上其实有很多很窄的石阶，相差不到一毫秒就可能踩空。新 gate 先把已经踩实的石阶圈出来，连续 RL 只能在圈内练球；想走到下一块，必须先派仿真探路并把成功和失败都记入地图。
