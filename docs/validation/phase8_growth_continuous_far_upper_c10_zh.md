# Phase 8 C10：连续跑射、数据运动先验与 5 米上角课程报告

日期：2026-08-08
范围：G1 / MuJoCo / SIM_ONLY。主机已核验为 4×RTX A6000，并以四路任务并发开展课程扫描；当前 MuJoCo 物理和 SONIC ONNX 明确使用 CPU，A6000 用于 EGL 视频渲染，未把 `CUDA_VISIBLE_DEVICES` 误写成 GPU 物理训练。未连接真实机器人，未发送硬件命令。

## 1. 结论先行

本轮针对“跑到球前明显停住，再走一步踢球”的主观问题补上了可量化定义，并完成了数据驱动运动先验、连续跑射和更远上角课程的闭环。

1. **连续跑射已经取得实质改善。** 4.5 m 高目标最终候选从控制器交接到触球由 `1.448 s` 降至 `0.914 s`，缩短 **36.9%**；交接后最低前向速度从 `0.116 m/s` 提高到 `0.265 m/s`，提高 **128%**；低于 `0.20 m/s` 的最长区间从 `0.12 s` 降为 **0 s**。机器人不再先站定再挪一步，新的视觉连续性硬门为 `true`。
2. 球从 `x=1.0 m` 射向 `x=5.5 m`，射程 **4.5 m**，在门线通过 `(y,z)=(1.023, 0.670) m`，目标误差 **3.07 cm**；无跌倒、无关节越界，严格双回放一致。
3. **5.0 m 左上角挑战已经实际执行，但没有假装成功。** 它仍保持连续跑射，门线通过 `(0.940, 0.598) m`，距声明上角 `0.925 m`。当前瓶颈不是横向瞄准，而是垂向触球能力。
4. 新弹道遥测测得该上角候选触球点低于球心约 `4.82 cm`，早期出球速度为 `(vx,vy,vz)=(9.477,1.782,4.336) m/s`，最高球心仅 `0.937 m`。在 5 m 射程下，这个垂向速度不可能命中 `z=1.35 m`；继续盲目增加抬腿或教师力已被反事实实验否定。
5. 当前两个展示候选仍分别有 22/23 个 2 ms 执行器饱和步，且上角未命中，因此都保持 **REJECTED DEVELOPMENT CANDIDATE**，没有进入 promotion 或硬件路径。

## 2. 用户提出的四项要求落实情况

| 要求 | 实施结果 | 状态 |
|---|---|---|
| 跑动到触球不中断 | 新增前向速度保持、低速持续时间和速度保留率；P210 + 0.8 m/s run-through + 数据运动先验实现 0.914 s 连续触球 | 已突破，仍需多种子 |
| 球离球门更远 | 保留 4.5 m 精确高球，同时实际推进到 5.0 m 上角挑战 | 已实施 |
| 左上角或右上角 | 选择左上角 `(y,z)=(1.0,1.35) m`，开展相位、抬升协同、教师、瞄准、摆速扫描 | 已尝试，尚未命中 |
| 下一轮一起开展 | 接入训练集运动先验、弹道遥测、Growth 新失败签名和下一轮学习接口 | 已实施基础层 |

## 3. 为什么旧指标没有发现“看起来停住”

旧指标 `pre_contact_motion_pause_sec` 只在整机平移速度几乎为零时记为暂停。机器人即使原地调整身体、以很低速度挪动，也会得到 `0.000 s`，所以机器判定和人眼感觉冲突。

本轮新增四个互补指标：

- `handoff_min_forward_speed_mps`：交接至触球的最低前向速度；
- `handoff_low_forward_speed_duration_sec`：低于 `0.20 m/s` 的最长连续时间；
- `handoff_forward_speed_retention_ratio`：最低前向速度相对交接速度的保留比例；
- `perceptual_continuity_passed`：同时要求触球延迟、旧暂停指标、最低速度、低速持续时间和速度保留率全部达标。

基准样例虽然旧暂停仍为零，但最低前向速度只有 `0.116 m/s`、低速段 `0.12 s`、速度保留率 `33.6%`，因此新门正确判为失败。这和用户看到的“停下来站住”一致。

## 4. 数据驱动运动先验

### 4.1 数据来源和隔离

新增 `rosclaw growth football-motion-prior`，从本地 OmniContact 足球数据的 **train split** 中提取右脚触球事件：

- 实际读取 59 个训练文件；
- 找到 63 个满足速度/接触条件的事件；
- 选择 24 个高质量事件；
- 按触球时刻对齐 `[-0.18, -0.12, -0.06, 0, 0.06, 0.12] s` 的右腿六关节序列；
- 用中位数形成稳健参考，同时保存 IQR；
- held-out 分区只做内容承诺，`heldout_metrics_accessed=false`，没有偷看测试指标；
- G1 Body、关节顺序契约、split manifest、每个源文件和 artifact 全部内容哈希绑定。

Artifact：

`/code/rosclaw/phase8_evidence/g1-growth-c10-omnicontact-football-motion-prior-v1.json`

Prior hash：

`sha256:5378afbb000186231300233d837b13b6853a2b31f392387f766a2df3509e501a`

### 4.2 运行时如何使用

先验不是直接输出力矩的“万能小脑”。运行时只在触球附近 0.36 s 的有限窗口，以最多 0.5 的 blend 混入现有踢球策略目标；单关节修正硬限幅为 0.45 rad，仍经过 PD、硬力矩裁剪和踢后右踝边界保护。证据记录实际激活帧和峰值目标改变量。

这次 blend=0.25 的先验与 P210 连续相位配合，把连续性从失败变为通过，同时保留有效高速射门。它证明数据集已经真正进入物理控制闭环；但它只覆盖右腿接触附近，不能单独解决上角弹道、左右脚切换和全身恢复。

## 5. 71 条严格物理反事实实验

本轮生成 71 个 C10 free-kick evidence，71 个均为 `strict_replay=true`。主要扫描维度包括：

- 踢球相位 P170–P220；
- 摆腿速度 1.10–1.50、预摆加载压缩 1.0–1.5；
- 助跑刹车速度 0.55–0.80 m/s；
- 数据先验 blend 0/0.10/0.25/0.50 及触球对齐帧；
- 垂向瞄准偏置、脚踝俯仰、髋膝抬升协同；
- 3–7 m/s 操作空间抬升教师、最大 60–100 N、前向教师；
- 4.5 m 自定义高目标和 5.0 m 左上角。

关键结果：

| 方案 | 触球延迟 | 最低前速 | 低速段 | 速度保留率 | 门线位置 `(y,z)` | 目标误差 | 关节越界 | 饱和步 |
|---|---:|---:|---:|---:|---|---:|---|---:|
| C9/P180 视觉停顿基准 | 1.448 s | 0.116 m/s | 0.12 s | 33.6% | (1.037, 0.605) | 5.88 cm | 无 | 10 |
| C10/P210 连续 4.5 m 高球 | **0.914 s** | **0.265 m/s** | **0 s** | **67.0%** | **(1.023, 0.670)** | **3.07 cm** | 无 | 22 |
| C10/P210 连续 5.0 m 左上挑战 | 0.918 s | 0.265 m/s | 0 s | 67.0% | (0.940, 0.598) | 75.48 cm | 无 | 23 |

### 5.1 被否决的直觉办法

- 单纯把整个挥腿动作加速到 1.35–1.50：出现不触球、关节越界或跌倒；
- 只压缩 P185–P235 的加载动作：最多改善约 0.11 s，仍保留低速挪动；
- 只增大抬升协同到 0.22–0.30 rad：高度非单调下降，部分球不再过门线；
- 把旧短距离上角教师直接迁到 P210：机器人连续且关节安全，但球反而全部变为贴地；
- 在 P170 保留教师并暴力加速：垂向出球速度降至 0–0.36 m/s，或破坏身体稳定。

这些结果说明高球是“进场本体状态 × 触球相位 × 足端速度方向 × 接触点 × 支撑腿协调”的混合动力学问题，不是一个抬腿幅值旋钮。

## 6. 新弹道学习遥测

结果契约升级后，每次射门额外记录：

- `kick_contact_point_xyz_m`；
- `kick_contact_height_relative_ball_center_m`；
- `ball_launch_velocity_xyz_mps`：首次触球后 120 ms 内速度最大的真实物理子步；
- `ball_apex_height_m`。

为什么必须记录 2 ms 级别：球脚冲量分布在多个物理子步，用 20 ms 控制帧的第一个或最后一个样本都会随机漏掉真实出球速度。新的指标把“球为什么飞不高”转成 critic 可学习的连续标签，而不是只留下一个最终 miss。

5 m 左上失败被 Growth 新增失败签名 `insufficient_ballistic_loft`，贡献因子明确为：

1. `ball_launch_vertical_velocity`；
2. `foot_ball_contact_height`；
3. `support_leg_coordination`。

可复用事件为 swing/contact/follow-through，学习器建议为 motion tracking + residual SAC。它同时保留 `contact_mode_precision`、`declared_corner_miss` 和 `authority_projection_required`，并继续 `promotion_ready=false`。

## 7. 当前闭环

```text
训练集足球触球数据
  -> Body/关节契约和分区哈希校验
  -> 训练集接触对齐运动先验
  -> 连续 SONIC 助跑 + 速度匹配桥 + 踢球策略
  -> 单世界 MuJoCo 严格双回放
  -> 连续性 + 2 ms 冲击 + 触球几何 + 出球弹道 + 身体安全判分
  -> Growth 失败签名与学习器路由
  -> 成功冠军冻结；失败轨迹进入下一轮课程
  -> 任务、身体和泛化同时通过才允许替换
```

本轮的“成长”不是每次训练都覆盖旧策略。P210 连续候选在视觉连续性和 4.5 m 精度上优于旧冠军，因此保留；5 m 上角和过载问题仍失败，因此只作为下一轮训练数据。这是 Stability–Plasticity 的 fail-closed 实现。

## 8. 视频与证据

### 连续 4.5 m 高目标（当前展示冠军）

- 证据：`/code/rosclaw/phase8_evidence/g1-growth-c10-continuous-high-final-v2/g1-free-kick.json`
- 轨迹：`/code/rosclaw/phase8_evidence/g1-growth-c10-continuous-high-final-v2/g1-free-kick-trajectory.npz`
- Growth：`/code/rosclaw/phase8_evidence/g1-growth-c10-continuous-high-final-v2/growth-triage-c10.json`
- 视频：`/code/rosclaw/phase8_evidence/g1-growth-c10-continuous-high-final-v2/g1-growth-c10-continuous-4p5m-high-development.mp4`

### 连续 5.0 m 左上角挑战（真实失败样本）

- 证据：`/code/rosclaw/phase8_evidence/g1-growth-c10-5m-left-upper-best-eval-v2/g1-free-kick.json`
- 轨迹：`/code/rosclaw/phase8_evidence/g1-growth-c10-5m-left-upper-best-eval-v2/g1-free-kick-trajectory.npz`
- Growth：`/code/rosclaw/phase8_evidence/g1-growth-c10-5m-left-upper-best-eval-v2/growth-triage-c10.json`
- 视频：`/code/rosclaw/phase8_evidence/g1-growth-c10-5m-left-upper-best-eval-v2/g1-growth-c10-continuous-5m-upper-challenge.mp4`

### 双实验长版

- 41.8 s / 1254 帧 / 1280×720 / 30 fps：`/code/rosclaw/phase8_evidence/g1-growth-c10-two-challenge-development.mp4`
- SHA-256：`c7b86e29203369a86713e5584e25f87b2bc5e580239763a1ceb58180f2a12122`

视频由严格物理轨迹重放，画面像素不参与评分。候选未晋升，因此带 SIM_ONLY / REJECTED 水印。

## 9. 验证

- 两个最终 episode 均执行两次仿真，result 和 trajectory digest 完全一致，`strict_replay=true`；
- 产品 CLI 已实际运行 `growth football-motion-prior`、`goalforge free-kick-showcase run/export`、`growth free-kick-triage`；
- 项目 `.venv` 下针对连续性、运动先验、IQL、弹道遥测、联合边界保护和 Growth 分流：**56 passed，2 deselected**；
- Ruff 和 mypy 通过；
- 系统裸 `pytest` 指向 Python 3.10，无法提供项目所需的 `StrEnum`/`datetime.UTC`；产品 CLI 和有效测试均使用项目固定 `.venv`，没有把解释器错配算作代码失败；
- 所有数据运动先验和最终证据保持 SIM_ONLY，activation/promotion/hardware authorization 均为 false。

## 10. 下一轮实施计划

1. **把弹道标签接入在线 actor-critic。** 状态加入触球前 200 ms 的骨盆速度、支撑脚力、足端速度/姿态和球相对位姿；动作使用相位率、六维右腿接触残差、骨盆/摆臂协调和接触后恢复残差；reward 同时包含角点误差、出球垂向速度、连续性、饱和、关节边界、后退和 jerk。
2. **先学接触几何，再学上角。** 用当前 71 条成功/失败反事实和 OmniContact 先验训练 contact-conditioned motion tracker，目标是把触球点从球心下方约 4.8 cm 提升到接近球心下缘的可控区，并把垂向出球速度从 4.34 提高到约 5.5–6.0 m/s。
3. **4 卡并行课程。** 分别负责 P180 高弹道冠军回放、P210 连续冠军回放、触球几何探索和恢复/安全 adversary；开发/保留种子严格分开，至少 32 个规划种子后才评估泛化。
4. **解决 Stability–Plasticity。** 每批更新混入冻结冠军 replay 和人体运动蒸馏，critic 使用任务 reward + 安全 cost 双头；任何角点能力提高但连续性、跌倒率、过载或踢后稳定回退的 candidate 都不替换冠军。
5. **去教师化和力矩安全。** 操作空间教师只用于探索标签；最终 actor 必须在教师力为零时完成射门，并把 22/23 个饱和子步降到零，才进入 sealed evaluation。
6. 左上角稳定后再增加右上角、来球初速度、左右脚条件变量和多人传射；不能用右脚固定先验的镜像目标冒充换脚能力。
