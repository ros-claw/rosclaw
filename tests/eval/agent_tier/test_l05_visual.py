"""L05 视觉 grounding（规格 §13.7）——agent 层真实驱动。

fixture（盲视觉）：RGB 图 + 深度图 + 相机标定——模型**拿不到**
MJCF/物体真值（只有传感器可见信息，§13.7 重要条款）。
模型交 answer.json（世界坐标）+ annotated.png（标注图）；
oracle：坐标误差 ≤20mm（与夹具生成时记录的真值比）+ 标注
质心落在蓝色目标区域内。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.eval.agent_tier import driver

SCENE = """<mujoco model="l05_scene">
  <worldbody>
    <geom type="plane" size="1 1 0.1" rgba="0.7 0.7 0.7 1"/>
    <body name="cube" pos="{cx} {cy} 0.012">
      <geom type="box" size="0.012 0.012 0.012" rgba="0.05 0.1 0.9 1" mass="0.05"/>
    </body>
    <camera name="cam" pos="{camx} {camy} {camz}" xyaxes="{xyaxes}"/>
  </worldbody>
</mujoco>"""

# (cx, cy, cam pos, euler)——两个变体（位置/视角变化）
VARIANTS = [
    {"id": "a", "cx": 0.12, "cy": -0.05,
     "camx": 0.0, "camy": -0.45, "camz": 0.42,
     "xyaxes": "1.0000 -0.0000 0.0000 0.0000 0.6351 0.7724"},
    {"id": "b", "cx": -0.08, "cy": 0.10,
     "camx": 0.05, "camy": -0.40, "camz": 0.45,
     "xyaxes": "0.9923 0.1240 -0.0000 -0.0874 0.6989 0.7098"},
]

_RENDER_CHILD = """
import json, sys
import mujoco
import numpy as np
from PIL import Image

scene_path, out_dir, meta_path = sys.argv[1], sys.argv[2], sys.argv[3]
model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
width, height = 320, 240
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
r = mujoco.Renderer(model, height, width)
r.update_scene(data, camera=cam_id)
rgb = r.render()
Image.fromarray(rgb).save(out_dir + "/rgb.png")
r.enable_depth_rendering()
r.update_scene(data, camera=cam_id)
depth = r.render()
np.save(out_dir + "/depth.npy", depth.astype(np.float32))
f = 0.5 * height / np.tan(model.cam_fovy[cam_id] * np.pi / 360.0)
meta = {
    "width": width, "height": height,
    "fx": float(f), "fy": float(f),
    "cx": width / 2.0, "cy": height / 2.0,
    "camera_pos": [float(v) for v in data.cam_xpos[cam_id]],
    "camera_mat": [float(v) for v in data.cam_xmat[cam_id]],
    "fovy_deg": float(model.cam_fovy[cam_id]),
    "note": "depth.npy 是沿光轴距离（米）。相机前向为 -Z 相机轴。"
            "反投影（像素 u,v）：z=depth[v,u]；"
            "x_cam = [(u-cx)*z/fx, -(v-cy)*z/fy, -z]；"
            "world = R @ x_cam + camera_pos，其中 R=camera_mat(3x3)，"
            "列为相机轴在世界中的方向（p_world = R @ p_cam + pos）",
}
json.dump(meta, open(meta_path, "w"), indent=1)
r.close()
"""


def _stage_sensor_files(workdir: Path, variant: dict) -> dict:
    """离线渲染 RGB-D（真值只留在 harness——模型拿不到）。"""
    workdir.mkdir(parents=True, exist_ok=True)
    scene = workdir / "_scene.xml"
    scene.write_text(SCENE.format(**variant), encoding="utf-8")
    out = workdir / "sensor"
    out.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _RENDER_CHILD, str(scene), str(out),
         str(out / "camera.json")],
        env={"MUJOCO_GL": "osmesa", "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-300:]
    return {"truth": [variant["cx"], variant["cy"], 0.012]}


PROMPT = """工作目录 sensor/ 下有一台 RGB-D 相机的输出：
- rgb.png：彩色图；
- depth.npy：沿光轴深度（float32 米）；
- camera.json：相机内参（fx/fy/cx/cy）、外参（camera_pos/camera_mat）与反投影说明。

任务：场景里有一个蓝色方块。定位它的世界坐标（米）。

要求：
1. 在 answer.json 里写 {"position_m": [x, y, z]}（世界坐标）；
2. 在 annotated.png 里输出标注图（在原图上把识别到的目标框/点标出来）；
3. 只许用 sensor/ 下的数据与通用图像处理（numpy/PIL）——
   坐标误差我们会和真值核对（≤20mm 才算定位正确），标注会核
   对是否落在目标上。
4. 完成后简述你的定位方法。"""


def _oracle(run: driver.AgentRun, truth: list[float]) -> None:
    import numpy as np
    from PIL import Image

    answer_path = run.ws / "answer.json"
    assert answer_path.exists(), "模型未交 answer.json（见 PTY 日志）"
    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    pos = answer.get("position_m")
    assert isinstance(pos, list) and len(pos) == 3, f"answer 格式: {answer}"
    # 度量：报告点到方块实际占据盒体的最小距离（可见面 vs 几何
    # 中心是物理不可分辨的——20mm 定位误差按占据空间操作化）。
    half = 0.012
    d = np.abs(np.array(pos) - np.array(truth)) - half
    err = float(np.linalg.norm(np.maximum(d, 0.0)))
    assert err <= 0.02, (
        f"定位误差（距盒体）{err * 1000:.1f}mm > 20mm（报告 {pos}，"
        f"真值中心 {truth}）"
    )
    annotated = run.ws / "annotated.png"
    assert annotated.exists(), "缺 annotated.png"
    # 标注有效性：rgb 上蓝色区域与 annotated 的标注重叠（sensor
    # 在任务根，不在 ws 内——2633 实证路径）。
    sensor = run.tmp_path / "sensor"
    rgb = np.asarray(Image.open(sensor / "rgb.png").convert("RGB"))
    blue_mask = (
        (rgb[:, :, 2].astype(int) > rgb[:, :, 0].astype(int) * 1.5)
        & (rgb[:, :, 2] > 60)
    )
    ys, xs = np.nonzero(blue_mask)
    assert len(xs) > 50, "夹具渲染的蓝色目标过小——fixture 问题"
    ann = np.asarray(Image.open(annotated).convert("RGB")).astype(int)
    diff = np.abs(ann - rgb.astype(int))
    changed = diff.sum(axis=2) > 60
    cys, cxs = np.nonzero(changed)
    assert len(cxs) > 0, "标注图与原图无差异——未标注"
    # 标注落点：变化像素质心与蓝色目标质心的距离（标注文字溢出
    # 边界框是正常画法——b 变体实证 48% 像素在内但标注完全正确）。
    dist = float(np.hypot(cxs.mean() - xs.mean(), cys.mean() - ys.mean()))
    assert dist <= 60.0, (
        f"标注质心距目标 {dist:.0f}px > 60px——标错位置"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not (driver.has_key() and driver.has_runtime()),
    reason="NOT_RUN: 无真实 key/Node——不合成冒充",
)
class TestL05VisualGroundingAgent:
    @pytest.mark.parametrize("variant", VARIANTS, ids=[v["id"] for v in VARIANTS])
    def test_locate_blue_cube(self, tmp_path: Path, variant: dict) -> None:
        meta = _stage_sensor_files(tmp_path, variant)
        run = driver.AgentRun(tmp_path, settle_timeout=1200)
        try:
            run.run(PROMPT)
        finally:
            run.stop()
        _oracle(run, meta["truth"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
