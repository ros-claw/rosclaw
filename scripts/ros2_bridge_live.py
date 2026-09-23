#!/usr/bin/env python3
"""G43 ROS2 exact-step agreement live bring-up（MH25 后续，一次性 operator 脚本）。

协议（MH25 §42/§43）：真桥 pause → step N → compare（joint by name +
clock 对齐 + 分通道残差）。桥跑 MuJoCo 3.12.0（vendor），rosclaw 侧
3.13.0——版本差异诚实记录，不假装同版本。

用法：python scripts/ros2_bridge_live.py [--steps 500] [--out /tmp/ros2-g43]
产出：<out>/evidence.json（含逐通道残差与判定）。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: 单摆 MJCF——重力摆动 + 轻阻尼，动力学敏感（任何积分/参数差异都会
#: 显形）。damping=0.01：重阻尼会在 pause 前的服务就绪空隙内衰减到
#: 静止平衡点，把一致性测试掏空（run10 实证：暂停在 qpos≈4.9e-33
#: 静止处，500 步后双侧 bitwise 一致但真空）。
PENDULUM_MJCF = """<mujoco model="g43_pendulum">
  <option timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
      <geom name="bob" type="capsule" size="0.02 0.15" pos="0 0 -0.15" mass="1.0"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="swing" qpos="1.0" qvel="0"/>
  </keyframe>
</mujoco>
"""

#: URDF 包装——ros2_control 块指 MJCF；joint 名与 MJCF 一致（§43 按名）。
PENDULUM_URDF = """<?xml version="1.0"?>
<robot name="g43_pendulum">
  <link name="world"/>
  <link name="link"/>
  <joint name="hinge" type="revolute">
    <parent link="world"/>
    <child link="link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.2" upper="3.2" effort="100" velocity="100"/>
  </joint>
  <ros2_control name="MujocoSystem" type="system">
    <hardware>
      <plugin>mujoco_ros2_control/MujocoSystemInterface</plugin>
      <param name="mujoco_model">{mjcf}</param>
      <param name="headless">true</param>
      <!-- 实证：headless 无 UI 限速，sim 自由狂奔（pause 前已跑 232
           仿真秒，摆杆全衰减归零）。钉 1.0 实时，pause 前空隙只过
           ~15 仿真秒 -->
      <param name="sim_speed_factor">1.0</param>
      <param name="initial_keyframe">swing</param>
    </hardware>
    <joint name="hinge">
      <state_interface name="position">
        <!-- 实证：xacro initial_value 覆盖 initial_keyframe（桥日志
             "Loading initial positions from ros2_control xacro"）——
             不置初值摆杆停在平衡底点，一致性测试会被掏空 -->
        <param name="initial_value">1.0</param>
      </state_interface>
      <state_interface name="velocity"/>
    </joint>
  </ros2_control>
</robot>
"""

CONTROLLERS_YAML = """controller_manager:
  ros__parameters:
    update_rate: 1000
    use_sim_time: true
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
"""

#: 独立 DDS 域隔离（实证：本机共享 domain 0 上有他 session 的
#: controller_manager/simcam 节点，ros2 control CLI 会打到别人的
#: CM——"already loaded" 幽灵）
ROS_SETUP = "source /opt/ros/jazzy/setup.bash && export ROS_DOMAIN_ID=42"


def _ros_run(cmd: str, *, timeout: float = 60.0, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["bash", "-c", f"{ROS_SETUP} && {cmd}"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"ros cmd failed: {cmd}\nstdout={proc.stdout[-500:]}\nstderr={proc.stderr[-500:]}")
    return proc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out", default=None)
    parser.add_argument("--tolerance-rad", type=float, default=1e-3)
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="ros2-g43-"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "pendulum.xml").write_text(PENDULUM_MJCF, encoding="utf-8")
    urdf = PENDULUM_URDF.replace("{mjcf}", str(out / "pendulum.xml"))
    (out / "pendulum.urdf").write_text(urdf, encoding="utf-8")
    (out / "controllers.yaml").write_text(CONTROLLERS_YAML, encoding="utf-8")

    evidence: dict = {
        "gate": "G43",
        "bridge_mujoco_version": "3.12.0 (ros-jazzy-mujoco-vendor)",
        "rosclaw_mujoco_version": None,
        "steps": args.steps,
        "tolerance_rad": args.tolerance_rad,
        "verdict": "NOT_RUN",
        "notes": [],
    }

    # 0) 预检：无 ROS/桥包立即诚实 NOT_RUN（CI 无 ROS——不装死等待）。
    # 实证：ros2 不在默认 PATH（需 source setup.bash）——查安装路径。
    if not (
        Path("/opt/ros/jazzy/bin/ros2").is_file()
        and Path("/opt/ros/jazzy/lib/mujoco_ros2_control").is_dir()
    ):
        evidence["notes"].append("ros2 CLI 或 mujoco_ros2_control 包不在本机（CI/无桥环境）")
        (out / "evidence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        return 0

    # 1) 起桥（后台进程组，结束必清）。本包定制 ros2_control_node 从
    # robot_description **话题**（transient_local）读 URDF（不读参数——
    # live 实证：只传参数会永等 "Waiting for data on 'robot_description'
    # topic"），故先起 robot_state_publisher 发 URDF。
    rsp = subprocess.Popen(
        [
            "bash",
            "-c",
            f"{ROS_SETUP} && exec ros2 run robot_state_publisher robot_state_publisher"
            f" --ros-args -p robot_description:=\"$(cat {out}/pendulum.urdf)\"",
        ],
        stdout=(out / "rsp.log").open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    launch_cmd = (
        f"{ROS_SETUP} && exec ros2 run mujoco_ros2_control ros2_control_node"
        f" --ros-args --params-file {out}/controllers.yaml"
    )
    bridge = subprocess.Popen(
        ["bash", "-c", launch_cmd],
        stdout=(out / "bridge.log").open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    evidence["bridge_pid"] = bridge.pid
    try:
        # 2) 等服务面起来。实证：定制节点名是 /mujoco_ros2_control_node
        # （非文档的 /ros2_control_node）——按 substring 发现全名。
        deadline = time.monotonic() + 60
        step_srv = None
        while time.monotonic() < deadline:
            if bridge.poll() is not None:
                evidence["notes"].append(f"bridge exited early rc={bridge.returncode}")
                break
            proc = _ros_run("ros2 service list", check=False)
            for line in proc.stdout.splitlines():
                if line.strip().endswith("/step_simulation"):
                    step_srv = line.strip()
                    break
            if step_srv:
                break
            time.sleep(2.0)
        if not step_srv:
            evidence["notes"].append("step_simulation service never appeared")
            raise RuntimeError("BRIDGE_NOT_UP")
        node = step_srv.rsplit("/", 1)[0]
        evidence["bridge_node"] = node

        # 激活 joint_state_broadcaster（必须在 pause 前——实证 run4：
        # 暂停时 controller_manager 服务不再应答，load 超时）。
        # 实证 run7：Jazzy CLI 的 --set-state active 会二次 configure
        # 已 active 的控制器而报错退出——以 list_controllers 实测状态
        # 为准，不看 CLI 退出码。
        _ros_run(
            "ros2 control load_controller joint_state_broadcaster --set-state active",
            timeout=90, check=False,
        )
        deadline = time.monotonic() + 30
        active = False
        while time.monotonic() < deadline:
            lst = _ros_run("ros2 control list_controllers", check=False)
            # 实证行格式：joint_state_broadcaster joint_state_broadcaster/JointStateBroadcaster  active
            for line in lst.stdout.splitlines():
                parts = line.split()
                if parts and parts[0] == "joint_state_broadcaster" and "active" in parts:
                    active = True
                    break
            if active:
                break
            time.sleep(1.5)
        if not active:
            raise RuntimeError("BROADCASTER_NOT_ACTIVE")
        evidence["broadcaster"] = "active"

        # 3) §42 exact-step：pause 冻结物理（原子生效于循环边界），然后
        # 以桥的暂停态测量值为共享初态——彻底消除 reset/pause 竞态
        # （run3 实证：自由跑空隙导致起点相位漂移；run4 实证：暂停时
        # 无法 load controller/未知 reset 是否应答）。
        _ros_run(
            f"ros2 service call {node}/set_pause "
            "mujoco_ros2_control_msgs/srv/SetPause \"{paused: true}\"",
            timeout=30,
        )
        # 判别实验实证（2026-09-23）：headless 起步即暂停；暂停期
        # joint_states 是 latch 陈旧值（reset 后读仍旧）——初态由
        # reset_world(keyframe) 服务级确认钉死，不由暂停期读数定。
        # reset 是节点级服务（不经 CM 循环，与 load_controller 不同——
        # run4 实证只 CM 服务暂停时不应答）。
        rst = _ros_run(
            f"ros2 service call {node}/reset_world "
            "mujoco_ros2_control_msgs/srv/ResetWorld \"{keyframe: 'swing'}\"",
            timeout=30, check=False,
        )
        evidence["reset_response_tail"] = rst.stdout[-160:]
        if "success=True" not in rst.stdout:
            raise RuntimeError(f"RESET_FAILED: {rst.stdout[-200:]}")

        def _read_joint(field: str) -> float:
            out_js = _ros_run(
                f"ros2 topic echo --once --field {field} /joint_states",
                timeout=30, check=False,
            )
            # 实证格式：array('d', [0.00085...])\n---（非纯 float 行）
            m = re.search(r"\[\s*([-+0-9.eE]+)", out_js.stdout)
            if not m:
                raise ValueError(f"joint_states.{field} parse failed: {out_js.stdout[-120:]!r}")
            return float(m.group(1))

        step_resp = _ros_run(
            f"ros2 service call {node}/step_simulation "
            f"mujoco_ros2_control_msgs/srv/StepSimulation \"{{steps: {args.steps}}}\"",
            timeout=max(60.0, 0.01 * args.steps + 30),
        )
        evidence["step_response_tail"] = step_resp.stdout[-200:]
        if "success=True" not in step_resp.stdout:
            raise RuntimeError(f"STEP_FAILED: {step_resp.stdout[-200:]}")
        # 等一拍让 broadcaster 发布 step 后状态（step 推进时钟→CM 才发布）
        time.sleep(1.0)
        bridge_qpos = _read_joint("position")
        bridge_qvel = _read_joint("velocity")
        evidence["bridge_final"] = {"qpos": bridge_qpos, "qvel": bridge_qvel}
        # 非空虚门槛：动力学必须真跑（终态显著偏离初态 1.0）——
        # 诚实纪律：静止的"一致"不算证据（run10 实证 bitwise 一致但空）。
        if abs(bridge_qpos - 1.0) < 0.05:
            evidence["verdict"] = "VACUOUS_REST"
            evidence["notes"].append("final ≈ initial — dynamics did not run")
            raise RuntimeError("VACUOUS_REST")

        # 5) rosclaw 侧对照 rollout（同 MJCF、同 keyframe 初态 qpos=1.0/
        # qvel=0、同步数）——桥 3.12.0 vs 本地 3.13.0 版本差异诚实记录。
        import mujoco

        evidence["rosclaw_mujoco_version"] = mujoco.__version__
        model = mujoco.MjModel.from_xml_string(PENDULUM_MJCF)
        data = mujoco.MjData(model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "swing")
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        for _ in range(args.steps):
            mujoco.mj_step(model, data)
        native_qpos, native_qvel = float(data.qpos[0]), float(data.qvel[0])
        evidence["native_final"] = {"qpos": native_qpos, "qvel": native_qvel}

        if math.isfinite(bridge_qpos):
            dq = abs(bridge_qpos - native_qpos)
            dv = abs(bridge_qvel - native_qvel)
            evidence["final_abs_diff"] = {"qpos": dq, "qvel": dv}
            evidence["verdict"] = (
                "AGREEMENT" if dq <= args.tolerance_rad and dv <= args.tolerance_rad else "DIVERGED"
            )
        else:
            evidence["verdict"] = "NOT_COMPARABLE"
    except Exception as exc:  # noqa: BLE001
        # 已有更具体判定（如 VACUOUS_REST）不覆盖——诚实粒度保留。
        if evidence["verdict"] == "NOT_RUN":
            evidence["notes"].append(f"{type(exc).__name__}: {exc}")
        else:
            evidence["notes"].append(f"aborted: {type(exc).__name__}: {exc}")
    finally:
        # 精确 PID 组杀 + 本 bring-up 独有串兜底（ros2 run wrapper 的
        # 子进程可能逃出进程组；g43_pendulum/ros2-g43 只属本流程）。
        subprocess.run(["kill", "-TERM", f"-{bridge.pid}"], check=False)
        subprocess.run(["kill", "-TERM", f"-{rsp.pid}"], check=False)
        time.sleep(2)
        subprocess.run(["kill", "-KILL", f"-{bridge.pid}"], check=False)
        subprocess.run(["kill", "-KILL", f"-{rsp.pid}"], check=False)
        subprocess.run(["pkill", "-9", "-f", "g43_pendulum"], check=False)
        subprocess.run(["pkill", "-9", "-f", "params-file.*ros2-g43"], check=False)

    (out / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    print(f"evidence → {out / 'evidence.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
