"""tests/sim 共享夹具（PR-MH2/MH4）：tiny_arm 内联 MJCF、已加载后端、
红绿 fixture 模型工厂。"""

from __future__ import annotations

from pathlib import Path

import pytest

TINY_MJCF = """<mujoco model="tiny_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.5"/>
      <geom name="base_geom" type="capsule" size="0.05 0.2" mass="1.0"/>
      <body name="forearm" pos="0 0 0.4">
        <joint name="elbow" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="forearm_geom" type="capsule" size="0.03 0.2" mass="0.5"/>
        <site name="tool0" pos="0 0 0.3"/>
      </body>
    </body>
    <camera name="top" pos="0 0 2"/>
  </worldbody>
  <actuator>
    <position name="shoulder_servo" joint="shoulder" kp="10" ctrlrange="-1.57 1.57" forcerange="-50 50"/>
    <position name="elbow_servo" joint="elbow" kp="5"/>
  </actuator>
  <sensor>
    <jointpos name="shoulder_pos" joint="shoulder"/>
  </sensor>
</mujoco>
"""


@pytest.fixture
def tiny_task_root(tmp_path):
    (tmp_path / "arm.xml").write_text(TINY_MJCF, encoding="utf-8")
    return tmp_path


@pytest.fixture
def loaded_backend(tiny_task_root):
    """(MujocoBackend, ModelReference)：tiny_arm 已加载。"""
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    backend = MujocoBackend(tiny_task_root)
    return backend, backend.load_model("arm.xml")


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixture_backend(tmp_path):
    """fixture 模型加载工厂：fixture_backend("broken_models", "01_no_collision")
    → (MujocoBackend, ModelReference)。"""

    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    backend = MujocoBackend(tmp_path)

    def _load(folder: str, name: str):
        source = FIXTURES_DIR / folder / f"{name}.xml"
        (tmp_path / f"{name}.xml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return backend, backend.load_model(f"{name}.xml")

    return _load
