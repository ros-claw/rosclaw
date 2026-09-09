# W04 渲染器补强（规格 §8）

**分支**：a-w04-renderer（基于 a-w03-state-replay，§17 依赖：W04←W00；
与 W03 同文件故栈叠避免冲突）
**日期**：2026-09-09

## 改动（`src/rosclaw/agentd/sim_render.py`）

### §8.1 渲染身份与幂等复用

- `render_identity_key()`：trace states digest + camera/尺寸/
  world/tool/spec digest + fps/playback + renderer build digest
  → 16 字符键。result 文件与 GIF/MP4 输出全部按键命名
  （`{trace_id}-{key}-scene.gif`——保留 `-scene.gif` 后缀约定，
  verifier 兼容）。
- 同渲染身份的已完成结果直接复用（`reused: true`，不重渲染）；
  同 trace 不同参数（视角/尺寸/fps）各自独立文件互不覆盖——
  e2e 实证两视角共存。

### §8.2 时间驱动与取景

- `select_frame_indices(states, fps, playback_rate)`：目标时刻
  网格 first+i/fps 取最近状态（替换计数均分）；首尾必含；
  playback rate 缩放输出时长。max_frames 降为安全上限
  （默认 60→480——10s 记录 12fps 需要 121 帧，旧默认会把
  时长砍半）。
- `apply_camera_framing()`：相机距离按轨迹包围盒半对角线缩放
  （follow/top/free 三预设各自系数与钳位）；无轨迹点回落旧
  固定参数。
- 单遍流式编码：imageio writer 逐帧写入 GIF+MP4——不再把
  PIL 与 NumPy 帧双份常驻内存。
- GIF 帧时长 = 1/fps（e2e 实证：输出 1× 时长与记录时长误差
  ≤ 一个输出帧量级）。

### §8.3 后端降级与探测缓存

- `_render_with_fallback()`：**全部候选**逐个真实尝试（不再是
  `candidates[:2]`）——EGL 渲染崩 + OSMesa 不可用 + Xvfb 可用
  时第三候选真正被探测并渲染成功；每个候选最多一次真实
  尝试；全失败聚合错误列出每个后端的真实原因
  （RENDER_BACKEND_EXHAUSTED）。
- 探测缓存：`_probe_cache_key()` = mujoco/Python 版本 +
  DISPLAY/MUJOCO_GL/xvfb-run/EGL vendor 环境。**成功才缓存**
  （同进程重复渲染零 smoke 开销）；失败不缓存——环境恢复后
  下次真实重探。

### W00 解标

`TestR3BackendFallbackDepth` xfail(strict) 解除——源码锚点 +
行为测试（monkeypatch 降级引擎 + 全失败聚合）双保险。

## 验证（红→绿）

| 测试 | 结果 |
|---|---|
| test_w04_renderer.py（12：第三候选到达/单候选单尝试+聚合错误/源码无 [:2]/探测缓存命中与失效重探/缓存键含环境/时间选帧/playback/短记录首尾/相机随包围范围/身份键稳定且敏感/重试复用+两视角共存 e2e/GIF 时长 e2e） | 红（10 failed）→ 绿 12/12 |
| W00 R3 解标 + 全量 baseline | 绿（R5 NOT_RUN skip；xfail 清零） |
| wp3/wp6/r05/a0902 渲染 e2e（PYTHONPATH=worktree——子进程解析坑） | 28 passed |
| tests/agentd + tests/sandbox 全量（除 PTY journey） | 见 PR（后台跑） |
| ruff check src tests | 全过 |

## 边界说明

- `_render_operation_child`（W02 api 路径）已是时间驱动 + 完整
  状态恢复，本轮未重复改；两条渲染路径的进一步合并留给
  后续工作包（不属于 §8 完成条件）。
- 「推物视频显示实际物体移动」：由 W03 完整状态记录 + 本条
  时间驱动回放联合保证（freejoint 物体 qpos 不再丢失）；
  专用推物场景 e2e 属 W09 测试类。
- 真实模型验收：**NOT_RUN**（无 key，不合成冒充）。
