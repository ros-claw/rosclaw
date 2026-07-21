"""Physics-backed mobile-base lease/deadman verification scenario."""

from __future__ import annotations

from dataclasses import asdict, dataclass

_MODEL_XML = """
<mujoco model="rosclaw_mobile_deadman">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="mobile_base" pos="0 0 0.11">
      <joint name="base_x" type="slide" axis="1 0 0" damping="0.2"/>
      <geom name="chassis" type="box" size="0.18 0.12 0.10" mass="8"
            rgba="0.15 0.35 0.55 1"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="base_velocity" joint="base_x" kv="240" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


@dataclass(frozen=True)
class MobileDeadmanEvidence:
    backend: str
    has_physics: bool
    command_velocity_mps: float
    lease_ttl_ms: int
    renew_interval_ms: int
    client_loss_at_sec: float
    deadman_tripped_at_sec: float
    stopped_at_sec: float
    position_at_client_loss_m: float
    final_position_m: float
    distance_after_client_loss_m: float
    final_velocity_mps: float
    stopped: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_mobile_deadman_scenario(
    *,
    command_velocity_mps: float = 0.5,
    lease_ttl_ms: int = 60,
    renew_interval_ms: int = 20,
    client_loss_at_sec: float = 0.2,
    end_at_sec: float = 0.8,
) -> MobileDeadmanEvidence:
    """Drive in MuJoCo, lose renewals, and verify controller-level braking."""

    if not 0 < renew_interval_ms < lease_ttl_ms:
        raise ValueError("renew_interval_ms must be positive and less than lease_ttl_ms")
    if not 0 < client_loss_at_sec < end_at_sec:
        raise ValueError("client_loss_at_sec must be positive and less than end_at_sec")
    import mujoco

    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    data = mujoco.MjData(model)
    timestep = float(model.opt.timestep)
    lease_ttl_sec = lease_ttl_ms / 1000.0
    renew_interval_sec = renew_interval_ms / 1000.0
    lease_expires_at = lease_ttl_sec
    next_renewal = renew_interval_sec
    deadman_tripped_at: float | None = None
    stopped_at: float | None = None
    position_at_loss: float | None = None
    data.ctrl[0] = command_velocity_mps

    while data.time < end_at_sec:
        now = float(data.time)
        if now < client_loss_at_sec and now + timestep >= next_renewal:
            lease_expires_at = now + lease_ttl_sec
            next_renewal += renew_interval_sec
        if position_at_loss is None and now >= client_loss_at_sec:
            position_at_loss = float(data.qpos[0])
        if deadman_tripped_at is None and now >= lease_expires_at:
            data.ctrl[0] = 0.0
            deadman_tripped_at = now
        mujoco.mj_step(model, data)
        if (
            deadman_tripped_at is not None
            and stopped_at is None
            and abs(float(data.qvel[0])) <= 0.01
        ):
            stopped_at = float(data.time)

    if position_at_loss is None or deadman_tripped_at is None:
        raise RuntimeError("mobile deadman scenario did not reach client loss and lease expiry")
    final_position = float(data.qpos[0])
    final_velocity = float(data.qvel[0])
    return MobileDeadmanEvidence(
        backend="mujoco",
        has_physics=True,
        command_velocity_mps=command_velocity_mps,
        lease_ttl_ms=lease_ttl_ms,
        renew_interval_ms=renew_interval_ms,
        client_loss_at_sec=client_loss_at_sec,
        deadman_tripped_at_sec=deadman_tripped_at,
        stopped_at_sec=stopped_at if stopped_at is not None else float(data.time),
        position_at_client_loss_m=position_at_loss,
        final_position_m=final_position,
        distance_after_client_loss_m=final_position - position_at_loss,
        final_velocity_mps=final_velocity,
        stopped=stopped_at is not None and abs(final_velocity) <= 0.01,
    )


__all__ = ["MobileDeadmanEvidence", "run_mobile_deadman_scenario"]
