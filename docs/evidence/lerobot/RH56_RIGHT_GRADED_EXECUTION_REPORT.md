# RH56 右手分级 REAL 执行报告（v1.0.1 清单 §4 Exp 2-7）

- 日期： 2026-07-23
- 身体： `rh56_right_01`（/dev/ttyUSB0，FTDI 5-1.2，slave 2）
- 传输 profile： `configs/rh56_right_rs485_v1.yaml`（STATUS/TEMP 6/6 寄存器，实测声明）
- 标定： `configs/rh56_right_01_calibration.yaml`（`measured_conservative`，validation.status=validated，mock=False）
- 执行器： `RH56Executor` → `SingleStepExecutor` → `RH56RealStepExecutor`（真机 REAL 路径，与产品 daemon 同款）
- 操作员在场，物理 E-Stop 可达；每级独立 arm（一次性 permit，max_step_delta 50 raw）

## 验收结果总览

| 级别 | 要求 | 实测 | 结果 | Practice 会话（verify --strict PASS） |
|---|---|---|---|---|
| Exp 2 noop（hold） | 20/20 | 20/20，hw_actions=20×steps | ✅ | `prac_20260723T090105Z_ab5a99` |
| Exp 3 micro（index −20 raw ×10） | 10/10 | 10/10 | ✅ | `prac_20260723T090139Z_5bf3e2` |
| Exp 4 motion（index −50 raw ×10） | 10/10 | 10/10 | ✅ | `prac_20260723T090314Z_ac87f1` |
| Exp 5 gesture（countdown 5→4→5 ×10） | 10/10 | 10/10 | ✅ | `prac_20260723T093632Z_083e01` |
| Exp 6+7 OK 接触 ×10 | ≥9/10 | **10/10**，contact 全部检出 | ✅ | `prac_20260723T142909Z_64107c` |

protection_events=0，emergency_over_contact=0，temperature_limited=0（全部级别）。

## OK 接触证据（Exp 6+7）

右手 OK 几何（v2 系列实验提升值，非左手参数迁移）：
- coarse `[1000,1000,1000,410,700,300]`，floor `[1000,1000,1000,410,340,300]`
- 两阶段接近：40-raw 块 → contact 窗口后 10-raw 微步
- 接触判据：max FORCE_ACT ≥ 70 g（运动基线 ~49 g 以上），中止 ≥250 g，硬限 300 g

10 次试验全部检出接触，thumb FORCE_ACT 峰值 89–121 g，两阶段 23–27 步到位，
回撤 64–69 步无接触；力-位移梯度与既有 ~12 g/raw 认知一致。温度 36–40 °C 无异常。

## 容差方法论（本次右手验收的决定性教训）

右手位置容差不能用手册值或左手值，必须**在真实运行包络**（速度、步长、整定时间、
多指同时运动）下实测。本次经历三轮实测修正，每轮都由失败试验的遥测驱动：

1. 速度 80 实测值（thumb_rot 19）在速度 400 执行时失败：`noop thumb_rot err 20 > 19`
   → 速度 400 复测得 thumb_rot 23。
2. 单指 40-raw 探针值在真实步态下失败：`gesture little err 11 > 10`
   （失败会话 `prac_20260723T090456Z_ee747d` 保留为证据）。
3. 试验包络复测（8 段连续同时 50-raw 块 + 800 ms 整定，取暂态最差 +5 余量）：
   **little 13, ring 9, middle 10, index 10, thumb 11, thumb_rot 28**
   → gesture 10/10，OK 10/10，全程零违例。

2 h 连续运行峰值温度 56–58 °C 记录在标定 notes（`thresholds_source: measured_conservative`）。

## 偏差与备注

- 首次 OK 运行（`prac_20260723T133217Z_fc8d61`，5/10 PASS 后）被 900 s `timeout`
  在 trial 5 中途 SIGTERM —— 每 trial ~3 min，10 次需 ~35 min；以 2700 s 重跑完整通过。
  中途被 kill 时手部无接触力、状态位健康，重跑前只做只读健康检查。
- `/tmp/exp3_graded_execution.json` 会被后续 exp3 调用覆盖（S7 子进程也写同一路径）；
  各级别权威证据以 practice 会话为准（上表）。
- 全部会话 `practice verify --strict` PASS（0 errors）。

## 复现

```bash
PYTHONPATH=src .venv/bin/python scripts/experiments/exp3_graded_execution.py \
  --levels noop,micro,motion,gesture,ok \
  --transport-profile configs/rh56_right_rs485_v1.yaml \
  --calibration configs/rh56_right_01_calibration.yaml \
  --body-id rh56_right_01 --hand right
# 注意：OK 级别 10 次试验约 35 min，timeout ≥ 2700 s
```
