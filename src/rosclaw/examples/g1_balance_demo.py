"""G1 Humanoid Standing Balance Demo.

Demonstrates basic bipedal balance control using ankle torque
and hip/knee PD feedback. This is a foundational stepping stone
towards full walking gait generation.
"""

import numpy as np


def create_g1_model():
    """Create a simplified G1 MuJoCo model with ankles."""
    xml = """
    <mujoco model="g1_balance">
      <option timestep="0.001" gravity="0 0 -9.81"/>
      <worldbody>
        <geom type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="pelvis" pos="0 0 0.9">
          <freejoint/>
          <geom type="box" size="0.12 0.08 0.06" mass="10" rgba="0.3 0.6 0.9 1"/>
          <body name="thigh_left" pos="0 0.12 -0.06">
            <joint name="hip_left" type="hinge" axis="1 0 0" range="-1.5 0.5" damping="3"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.05" mass="3" rgba="0.8 0.3 0.2 1"/>
            <body name="shin_left" pos="0 0 -0.35">
              <joint name="knee_left" type="hinge" axis="1 0 0" range="0 2.5" damping="2"/>
              <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" mass="2" rgba="0.8 0.3 0.2 1"/>
              <body name="foot_left" pos="0 0 -0.3">
                <joint name="ankle_left" type="hinge" axis="1 0 0" range="-0.5 0.5" damping="1"/>
                <geom type="box" size="0.08 0.04 0.015" mass="0.8" rgba="0.2 0.2 0.2 1"/>
              </body>
            </body>
          </body>
          <body name="thigh_right" pos="0 -0.12 -0.06">
            <joint name="hip_right" type="hinge" axis="1 0 0" range="-1.5 0.5" damping="3"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.05" mass="3" rgba="0.8 0.3 0.2 1"/>
            <body name="shin_right" pos="0 0 -0.35">
              <joint name="knee_right" type="hinge" axis="1 0 0" range="0 2.5" damping="2"/>
              <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" mass="2" rgba="0.8 0.3 0.2 1"/>
              <body name="foot_right" pos="0 0 -0.3">
                <joint name="ankle_right" type="hinge" axis="1 0 0" range="-0.5 0.5" damping="1"/>
                <geom type="box" size="0.08 0.04 0.015" mass="0.8" rgba="0.2 0.2 0.2 1"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor joint="hip_left" gear="80" ctrlrange="-120 120"/>
        <motor joint="knee_left" gear="60" ctrlrange="-90 90"/>
        <motor joint="ankle_left" gear="40" ctrlrange="-60 60"/>
        <motor joint="hip_right" gear="80" ctrlrange="-120 120"/>
        <motor joint="knee_right" gear="60" ctrlrange="-90 90"/>
        <motor joint="ankle_right" gear="40" ctrlrange="-60 60"/>
      </actuator>
    </mujoco>
    """
    return xml


def balance_controller(model, data, target_height=0.9):
    """Simple inverted-pendulum-inspired balance controller."""
    height = data.qpos[2]
    pitch = data.qpos[4] if model.nq > 4 else 0.0
    
    Kp_pitch = 150.0
    Kd_pitch = 30.0
    Kp_height = 80.0
    Kd_height = 20.0
    
    h_error = target_height - height
    ankle_torque = Kp_pitch * (-pitch) - Kd_pitch * data.qvel[4]
    hip_torque = Kp_height * h_error - Kd_height * data.qvel[7]
    knee_torque = -Kp_height * 0.5 * h_error - Kd_height * 0.3 * data.qvel[8]
    
    data.ctrl[0] = hip_torque
    data.ctrl[1] = knee_torque
    data.ctrl[2] = ankle_torque
    data.ctrl[3] = hip_torque
    data.ctrl[4] = knee_torque
    data.ctrl[5] = ankle_torque


def run_demo(duration=5.0):
    """Run the G1 balance demo."""
    import mujoco
    
    xml = create_g1_model()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    print(f"G1 Balance Demo: {model.nq} DOF, {model.nu} actuators")
    
    steps = int(duration / 0.001)
    for i in range(steps):
        balance_controller(model, data)
        mujoco.mj_step(model, data)
        
        if i % 1000 == 0:
            print(f"  t={data.time:.2f}s: height={data.qpos[2]:.3f}m, "
                  f"pitch={data.qpos[4]:.3f}rad")
    
    print(f"\nFinal: height={data.qpos[2]:.3f}m, pitch={data.qpos[4]:.3f}rad")
    success = abs(data.qpos[2] - 0.9) < 0.1 and abs(data.qpos[4]) < 0.3
    print(f"Balance maintained: {success}")
    return success


if __name__ == "__main__":
    run_demo()
