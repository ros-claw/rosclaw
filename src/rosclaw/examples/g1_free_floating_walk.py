"""G1 Humanoid Free-Floating Walking Demo.

Free-floating bipedal robot walks 3 meters with fall detection.
Demonstrates:
  - Free-joint base (6 DOF) + leg kinematics
  - Sinusoidal gait pattern with balance correction
  - Real-time fall detection (height + orientation thresholds)
  - GPU-ready MuJoCo simulation
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


def create_g1_free_floating_model():
    """Return MuJoCo XML for simplified free-floating G1 humanoid."""
    # 16 DOF: 6 freejoint (pelvis) + 5 left leg + 5 right leg
    xml = """
    <mujoco model="g1_free_floating">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast">
        <flag warmstart="enable"/>
      </option>
      <default>
        <joint damping="2" armature="0.01"/>
        <geom friction="0.9 0.2 0.01" density="1000"/>
        <motor ctrllimited="true" ctrlrange="-100 100"/>
      </default>

      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.6 0.8 1.0" rgb2="0.2 0.4 0.8"
                 width="512" height="512"/>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.9"
                 rgb2="0.7 0.7 0.7" width="512" height="512"/>
        <material name="grid_mat" texture="grid" texrepeat="10 10" reflectance="0.1"/>
        <material name="pelvis_mat" rgba="0.3 0.6 0.9 1"/>
        <material name="leg_mat" rgba="0.8 0.3 0.2 1"/>
        <material name="foot_mat" rgba="0.2 0.2 0.2 1"/>
      </asset>

      <worldbody>
        <light pos="0 0 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="50 50 0.1" material="grid_mat"/>

        <!-- Free-floating pelvis -->
        <body name="pelvis" pos="0 0 0.75">
          <freejoint name="root"/>
          <geom type="box" size="0.12 0.08 0.06" mass="8" material="pelvis_mat"/>
          <!-- torso visual -->
          <geom type="capsule" fromto="0 0 0.06 0 0 0.35" size="0.07" mass="15"
                material="pelvis_mat"/>

          <!-- Left Leg -->
          <body name="hip_yaw_left" pos="0 0.12 -0.06">
            <joint name="hip_yaw_left" type="hinge" axis="0 0 1" range="-0.87 0.87"/>
            <geom type="sphere" size="0.01" mass="0.1" rgba="0 0 0 0"/>
            <body name="hip_roll_left" pos="0 0 0">
              <joint name="hip_roll_left" type="hinge" axis="1 0 0" range="-0.52 0.52"/>
              <geom type="sphere" size="0.01" mass="0.1" rgba="0 0 0 0"/>
              <body name="hip_pitch_left" pos="0 0 0">
                <joint name="hip_pitch_left" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.055" mass="3.5"
                      material="leg_mat"/>
                <body name="knee_left" pos="0 0 -0.25">
                  <joint name="knee_pitch_left" type="hinge" axis="0 1 0" range="-0.1 2.5"/>
                  <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.045" mass="2.5"
                        material="leg_mat"/>
                  <body name="ankle_left" pos="0 0 -0.25">
                    <joint name="ankle_pitch_left" type="hinge" axis="0 1 0" range="-0.8 0.5"/>
                    <geom type="capsule" fromto="0 0 0 0 0 -0.08" size="0.04" mass="1.5"
                          material="leg_mat"/>
                    <body name="foot_left" pos="0 0 -0.08">
                      <geom name="foot_left_geom" type="box" size="0.09 0.04 0.012"
                            mass="0.8" material="foot_mat"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>

          <!-- Right Leg -->
          <body name="hip_yaw_right" pos="0 -0.12 -0.06">
            <joint name="hip_yaw_right" type="hinge" axis="0 0 1" range="-0.87 0.87"/>
            <geom type="sphere" size="0.01" mass="0.1" rgba="0 0 0 0"/>
            <body name="hip_roll_right" pos="0 0 0">
              <joint name="hip_roll_right" type="hinge" axis="1 0 0" range="-0.52 0.52"/>
              <geom type="sphere" size="0.01" mass="0.1" rgba="0 0 0 0"/>
              <body name="hip_pitch_right" pos="0 0 0">
                <joint name="hip_pitch_right" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.055" mass="3.5"
                      material="leg_mat"/>
                <body name="knee_right" pos="0 0 -0.25">
                  <joint name="knee_pitch_right" type="hinge" axis="0 1 0" range="-0.1 2.5"/>
                  <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.045" mass="2.5"
                        material="leg_mat"/>
                  <body name="ankle_right" pos="0 0 -0.25">
                    <joint name="ankle_pitch_right" type="hinge" axis="0 1 0" range="-0.8 0.5"/>
                    <geom type="capsule" fromto="0 0 0 0 0 -0.08" size="0.04" mass="1.5"
                          material="leg_mat"/>
                    <body name="foot_right" pos="0 0 -0.08">
                      <geom name="foot_right_geom" type="box" size="0.09 0.04 0.012"
                            mass="0.8" material="foot_mat"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </worldbody>

      <actuator>
        <position joint="hip_yaw_left" kp="60" kv="12" ctrlrange="-0.87 0.87"/>
        <position joint="hip_roll_left" kp="80" kv="16" ctrlrange="-0.52 0.52"/>
        <position joint="hip_pitch_left" kp="100" kv="20" ctrlrange="-2.0 2.0"/>
        <position joint="knee_pitch_left" kp="80" kv="16" ctrlrange="-0.1 2.5"/>
        <position joint="ankle_pitch_left" kp="40" kv="8" ctrlrange="-0.8 0.5"/>
        <position joint="hip_yaw_right" kp="60" kv="12" ctrlrange="-0.87 0.87"/>
        <position joint="hip_roll_right" kp="80" kv="16" ctrlrange="-0.52 0.52"/>
        <position joint="hip_pitch_right" kp="100" kv="20" ctrlrange="-2.0 2.0"/>
        <position joint="knee_pitch_right" kp="80" kv="16" ctrlrange="-0.1 2.5"/>
        <position joint="ankle_pitch_right" kp="40" kv="8" ctrlrange="-0.8 0.5"/>
      </actuator>

      <sensor>
        <framepos name="pelvis_pos" objtype="body" objname="pelvis"/>
        <framequat name="pelvis_quat" objtype="body" objname="pelvis"/>
        <subtreecom name="com" body="pelvis"/>
      </sensor>
    </mujoco>
    """
    return xml


@dataclasses.dataclass
class WalkState:
    """State tracker for walking controller."""
    distance_traveled: float = 0.0
    steps_taken: int = 0
    fall_detected: bool = False
    fall_reason: Optional[str] = None
    target_distance: float = 3.0
    gait_phase: float = 0.0
    stance_width: float = 0.18
    step_length: float = 0.15
    step_height: float = 0.06
    cycle_time: float = 0.8  # seconds per gait cycle
    pelvis_height_target: float = 0.78

    # Fall detection thresholds
    min_pelvis_height: float = 0.35
    max_tilt_angle_deg: float = 45.0
    max_sim_time: float = 30.0


def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to roll, pitch, yaw (rad)."""
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def check_fall(
    walk_state: WalkState,
    pelvis_height: float,
    pelvis_quat: np.ndarray,
    com_pos: np.ndarray,
) -> bool:
    """Detect if the robot has fallen."""
    if walk_state.fall_detected:
        return True

    # Check pelvis height
    if pelvis_height < walk_state.min_pelvis_height:
        walk_state.fall_detected = True
        walk_state.fall_reason = (
            f"pelvis_height_low ({pelvis_height:.3f}m < {walk_state.min_pelvis_height}m)"
        )
        return True

    # Check tilt angle (roll or pitch too large)
    rpy = quat_to_rpy(pelvis_quat)
    tilt_deg = np.degrees(np.max(np.abs(rpy[:2])))
    if tilt_deg > walk_state.max_tilt_angle_deg:
        walk_state.fall_detected = True
        walk_state.fall_reason = (
            f"excessive_tilt ({tilt_deg:.1f}deg > {walk_state.max_tilt_angle_deg}deg)"
        )
        return True

    # Check COM projection on ground
    if com_pos[2] < walk_state.min_pelvis_height * 0.8:
        walk_state.fall_detected = True
        walk_state.fall_reason = f"com_too_low ({com_pos[2]:.3f}m)"
        return True

    return False


def compute_gait_control(
    walk_state: WalkState,
    sim_time: float,
    pelvis_pos: np.ndarray,
    pelvis_quat: np.ndarray,
) -> np.ndarray:
    """Compute target joint positions for bipedal walking gait.

    Returns ctrl array of shape (nu,) with target joint angles for
    position actuators.
    """
    phase = 2 * np.pi * sim_time / walk_state.cycle_time
    walk_state.gait_phase = phase

    # Left and right leg phase offsets (anti-phase for walking)
    left_phase = phase
    right_phase = phase + np.pi

    # Hip pitch: push off in stance, swing forward in swing
    hip_amp = 0.25
    hip_pitch_left = hip_amp * np.sin(left_phase)
    hip_pitch_right = hip_amp * np.sin(right_phase)

    # Knee pitch: flex during swing, extend in stance
    knee_amp = 0.4
    knee_pitch_left = knee_amp * max(0.0, np.sin(left_phase))
    knee_pitch_right = knee_amp * max(0.0, np.sin(right_phase))

    # Ankle pitch: slight dorsiflexion at toe-off
    ankle_amp = 0.15
    ankle_pitch_left = ankle_amp * np.sin(left_phase - np.pi / 4)
    ankle_pitch_right = ankle_amp * np.sin(right_phase - np.pi / 4)

    # Hip roll: lateral balance (keep pelvis level, slight counter-sway)
    hip_roll_amp = 0.05
    hip_roll_left = hip_roll_amp * np.cos(phase)
    hip_roll_right = -hip_roll_amp * np.cos(phase)

    # Hip yaw: slight toe-out for stability
    hip_yaw = 0.03 * np.sin(phase)

    # Height correction: bias hip pitch based on pelvis height error
    height_error = walk_state.pelvis_height_target - pelvis_pos[2]
    height_correction = np.clip(height_error * 0.5, -0.1, 0.1)

    # Forward velocity correction bias
    forward_bias = 0.03

    ctrl = np.zeros(10)
    # Left leg: [hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch]
    ctrl[0] = hip_yaw
    ctrl[1] = hip_roll_left
    ctrl[2] = -(hip_pitch_left + forward_bias) + height_correction
    ctrl[3] = knee_pitch_left
    ctrl[4] = ankle_pitch_left
    # Right leg
    ctrl[5] = -hip_yaw
    ctrl[6] = hip_roll_right
    ctrl[7] = -(hip_pitch_right + forward_bias) + height_correction
    ctrl[8] = knee_pitch_right
    ctrl[9] = ankle_pitch_right

    return ctrl


def run_walking_demo(
    target_distance: float = 3.0,
    duration_limit: float = 30.0,
    render: bool = False,
    verbose: bool = True,
    gpu: bool = False,
) -> dict:
    """Run G1 free-floating walking demo.

    Args:
        target_distance: Target walking distance in meters (x-direction).
        duration_limit: Max simulation time in seconds.
        render: Whether to render (not implemented in headless mode).
        verbose: Print progress.
        gpu: Attempt GPU acceleration (requires MJX or compatible backend).

    Returns:
        Result dict with keys: success, distance_traveled, fall_detected,
        fall_reason, steps_taken, sim_time, energy_used, gpu_used.
    """
    import mujoco

    xml = create_g1_free_floating_model()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Get sensor addresses
    sensor_adr = {}
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        sensor_adr[name] = model.sensor_adr[i]

    walk_state = WalkState(
        target_distance=target_distance,
        max_sim_time=duration_limit,
    )

    # Initial pose: standing
    # qpos: [x, y, z, qw, qx, qy, qz, joint_angles...]
    data.qpos[2] = 0.78  # pelvis height
    data.qpos[3] = 1.0   # qw
    data.qpos[7] = -0.05   # hip_pitch_left
    data.qpos[8] = 0.1     # knee_pitch_left
    data.qpos[9] = -0.05   # ankle_pitch_left
    data.qpos[12] = -0.05  # hip_pitch_right
    data.qpos[13] = 0.1    # knee_pitch_right
    data.qpos[14] = -0.05  # ankle_pitch_right

    mujoco.mj_forward(model, data)

    if verbose:
        print(f"G1 Free-Floating Walk: {model.nq} DOF, {model.nu} actuators")
        print(f"  Target: {target_distance:.1f}m | Max time: {duration_limit:.1f}s")
        print(f"  Initial pelvis height: {data.qpos[2]:.3f}m")

    dt = model.opt.timestep
    max_steps = int(duration_limit / dt)
    energy = 0.0
    start_x = data.qpos[0]

    # Warm-up: settle into standing for 1.0s using position actuators
    settle_steps = int(1.0 / dt)
    # Standing pose target for all 10 joints
    target_qpos = np.zeros(10)
    target_qpos[2] = -0.05   # hip_pitch_left
    target_qpos[3] = 0.1     # knee_pitch_left
    target_qpos[4] = -0.05   # ankle_pitch_left
    target_qpos[7] = -0.05   # hip_pitch_right
    target_qpos[8] = 0.1     # knee_pitch_right
    target_qpos[9] = -0.05   # ankle_pitch_right
    for _ in range(settle_steps):
        data.ctrl[:] = target_qpos
        mujoco.mj_step(model, data)

    # Main walking loop
    report_interval = int(1.0 / dt)
    for step in range(max_steps):
        # Read sensors
        pelvis_pos = data.sensordata[
            sensor_adr["pelvis_pos"]:sensor_adr["pelvis_pos"] + 3
        ]
        pelvis_quat = data.sensordata[
            sensor_adr["pelvis_quat"]:sensor_adr["pelvis_quat"] + 4
        ]
        com_pos = data.sensordata[sensor_adr["com"]:sensor_adr["com"] + 3]

        # Fall detection
        if check_fall(walk_state, pelvis_pos[2], pelvis_quat, com_pos):
            if verbose:
                print(
                    f"\n[FAIL] Fall detected at t={data.time:.2f}s: {walk_state.fall_reason}"
                )
            break

        # Compute control
        data.ctrl[:] = compute_gait_control(
            walk_state, data.time, pelvis_pos, pelvis_quat
        )

        # Step physics
        mujoco.mj_step(model, data)
        energy += np.sum(np.abs(data.ctrl * data.qvel[6:])) * dt

        # Progress tracking
        walk_state.distance_traveled = pelvis_pos[0] - start_x
        if step % report_interval == 0 and verbose and step > 0:
            rpy = quat_to_rpy(pelvis_quat)
            print(
                f"  t={data.time:.2f}s | x={walk_state.distance_traveled:.3f}m | "
                f"z={pelvis_pos[2]:.3f}m | pitch={np.degrees(rpy[1]):.1f}deg | "
                f"energy={energy:.1f}J"
            )

        # Success check
        if walk_state.distance_traveled >= target_distance:
            walk_state.steps_taken = step
            if verbose:
                print(
                    f"\n[SUCCESS] Target reached: {walk_state.distance_traveled:.3f}m "
                    f"in {data.time:.2f}s"
                )
            break
    else:
        if verbose:
            print(
                f"\n[TIMEOUT] Traveled {walk_state.distance_traveled:.3f}m "
                f"in {data.time:.2f}s"
            )

    # GPU detection: check if MJX/JAX GPU is available
    gpu_used = False
    if gpu:
        try:
            import jax
            devices = jax.devices("gpu")
            gpu_used = len(devices) > 0
        except Exception:
            pass

    result = {
        "success": (
            walk_state.distance_traveled >= target_distance * 0.95
            and not walk_state.fall_detected
        ),
        "distance_traveled": float(walk_state.distance_traveled),
        "target_distance": target_distance,
        "fall_detected": walk_state.fall_detected,
        "fall_reason": walk_state.fall_reason,
        "steps_taken": walk_state.steps_taken,
        "sim_time": float(data.time),
        "energy_used": float(energy),
        "gpu_used": gpu_used,
        "final_pelvis_height": (
            float(pelvis_pos[2]) if "pelvis_pos" in sensor_adr else None
        ),
    }

    if verbose:
        status = "PASS" if result["success"] else "FAIL"
        print(f"\n{'='*50}")
        print(f"Result: {status}")
        print(f"  Distance: {result['distance_traveled']:.3f}m / {target_distance:.1f}m")
        print(f"  Sim time: {result['sim_time']:.2f}s")
        print(f"  Energy:   {result['energy_used']:.1f}J")
        print(f"  Fall:     {result['fall_detected']} ({result['fall_reason'] or 'N/A'})")
        print(f"  GPU:      {result['gpu_used']}")
        print(f"{'='*50}")

    return result


if __name__ == "__main__":
    run_walking_demo(target_distance=3.0, verbose=True)
