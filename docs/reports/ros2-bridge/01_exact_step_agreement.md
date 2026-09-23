# G43 ROS2 Exact-step Agreement — live 证据（2026-09-23，GB10）

**判定：AGREEMENT（PASS）**——真桥（ros-jazzy-mujoco-ros2-control
0.1.1，vendor MuJoCo **3.12.0**）与 rosclaw 本地 MuJoCo **3.13.0**
对照：单摆 keyframe(qpos=1.0) 起步，pause → reset_world(swing) →
step_simulation(500)（500×2ms=1s 摆动），终态 qpos/qvel **逐位一致**
（0.8359684472717589 / -0.981779901850038，diff 0.0/0.0）。

证据：`01_exact_step_agreement.json`（本目录）。
复跑：`python scripts/ros2_bridge_live.py --steps 500`
测试：`tests/sim/test_ros2_bridge_live.py`（CI 无桥诚实 NOT_RUN）。

## 安装实况

- packages.ros.org 直连不可达（IPv6 unreachable + IPv4 超时）——
  清华 tuna 镜像（mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu）装通：
  ros-jazzy-mujoco-ros2-control 0.1.1 + msgs + plugins + mujoco-vendor。
- MH25 报告的 stale index 404 已解除（2026-06/09 构建入索引）。

## 协议实证坑（bring-up 六连）

1. **定制 ros2_control_node 从 robot_description 话题读 URDF**
   （transient_local latch），不读参数——只传参数永等 "Waiting for
   data on 'robot_description' topic"。需 robot_state_publisher 先行。
2. **节点实名 /mujoco_ros2_control_node**（非文档的
   /ros2_control_node）——服务发现按 substring 匹配。
3. **暂停时 controller_manager 服务不应答**（load_controller 超时）；
   reset_world/step_simulation/set_pause 是节点级服务，暂停时正常。
4. **headless 起步即暂停**（"Simulation is already paused"），且
   xacro 初值默认 0 覆盖 initial_keyframe/initial_value——物理起点
   须 pause + reset_world(keyframe) 钉死（服务级确认）。
5. **暂停期 /joint_states 是 latch 陈旧值**（reset 后读仍旧）——
   step 推进时钟后 CM 才发布新值；读数只在 step 后有效。
6. **Jazzy `ros2 control load_controller --set-state active` 会二次
   configure 已 active 控制器而报错退出**——以 list_controllers
   实测状态为准。
7. **headless 无 UI 时限速失效自由狂奔**（pause 前已跑 232 仿真秒，
   摆杆全衰减）——须 `sim_speed_factor` 钉 1.0。
8. **共享 DDS domain 0 有他 session 节点**（controller_manager 幽灵
   "already loaded"）+ 调试残留的僵尸桥会同域串台（同 stamp 双发布
   者污染 latch 读数）——独立 `ROS_DOMAIN_ID=42` + 精确 PID/特征串
   清杀是纪律。

## 边界与后续

- 模型覆盖：单摆单关节被动关节（被动关节注册路径实证）。多关节/
  执行器控制通道（position/velocity/effort 映射）、free joint、
  传感器（FT/IMU/camera）的一致性留后续矩阵。
- 跨版本（3.12 桥 vs 3.13 本地）在本模型零数值分歧；接触丰富模型
  的分歧面待测。
- G44（fault isolation）live 化现在是可及的（桥可控可停）。
