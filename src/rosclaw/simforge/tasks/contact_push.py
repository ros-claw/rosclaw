"""ContactPush task backed by real MuJoCo contact and force measurements."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

_MODEL = """
<mujoco model="contact_push">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05" friction="0.7 0.01 0.001"/>
    <body name="object" pos="0.10 0 0.05">
      <freejoint/>
      <geom name="object_geom" type="box" size="0.04 0.04 0.04" mass="0.35"
            friction="0.55 0.01 0.001"/>
    </body>
    <body name="pusher" pos="-0.12 0 0.05">
      <joint name="pusher_x" type="slide" axis="1 0 0" range="-0.2 0.8" damping="1"/>
      <geom name="pusher_geom" type="box" size="0.025 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="push_velocity" joint="pusher_x" kv="150" ctrlrange="0 0.5"/>
  </actuator>
</mujoco>
"""


@dataclass(frozen=True)
class ContactPushEvidence:
    backend: str
    physics_executed: bool
    contact_observed: bool
    object_displacement_m: float
    peak_contact_force_n: float
    final_object_speed_mps: float
    steps: int
    success: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_contact_push(
    *,
    command_velocity_mps: float = 0.25,
    duration_sec: float = 1.5,
    required_displacement_m: float = 0.08,
    force_limit_n: float = 200.0,
) -> ContactPushEvidence:
    command_velocity_mps = _bounded_number(
        command_velocity_mps,
        name="command velocity",
        minimum=0.0,
        maximum=0.5,
        minimum_inclusive=False,
    )
    duration_sec = _bounded_number(
        duration_sec,
        name="duration",
        minimum=0.1,
        maximum=10.0,
    )
    required_displacement_m = _bounded_number(
        required_displacement_m,
        name="required displacement",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    force_limit_n = _bounded_number(
        force_limit_n,
        name="force limit",
        minimum=0.0,
        maximum=10_000.0,
        minimum_inclusive=False,
    )
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(_MODEL)
    data = mujoco.MjData(model)
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    pusher_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pusher_geom")
    mujoco.mj_forward(model, data)
    initial_x = float(data.xpos[object_body][0])
    data.ctrl[0] = command_velocity_mps
    peak_force = 0.0
    contact_observed = False
    steps = math.ceil(duration_sec / float(model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            if {int(contact.geom1), int(contact.geom2)} != {object_geom, pusher_geom}:
                continue
            contact_observed = True
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_index, force)
            peak_force = max(peak_force, float(np.linalg.norm(force[:3])))
    displacement = float(data.xpos[object_body][0]) - initial_x
    object_velocity = float(data.cvel[object_body][3])
    success = bool(
        contact_observed
        and math.isfinite(displacement)
        and math.isfinite(peak_force)
        and displacement >= required_displacement_m
        and peak_force <= force_limit_n
        and math.isfinite(object_velocity)
    )
    return ContactPushEvidence(
        backend="mujoco_cpu",
        physics_executed=True,
        contact_observed=contact_observed,
        object_displacement_m=displacement,
        peak_contact_force_n=peak_force,
        final_object_speed_mps=object_velocity,
        steps=steps,
        success=success,
    )


def _bounded_number(
    value: float,
    *,
    name: str,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    normalized = float(value)
    lower_ok = normalized >= minimum if minimum_inclusive else normalized > minimum
    if not math.isfinite(normalized) or not lower_ok or normalized > maximum:
        interval = "[" if minimum_inclusive else "("
        raise ValueError(f"{name} must be in {interval}{minimum:g}, {maximum:g}]")
    return normalized


__all__ = ["ContactPushEvidence", "run_contact_push"]
