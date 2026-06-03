# ROSClaw v1.0 系统发现测试报告

## 测试时间
2026-06-03

## 系统整体状态
✅ HEALTHY - 所有 7 个核心模块健康

## 1. 当前有哪些机器人？

**8 个可用机器人：**

| Robot ID | 类型 |
|----------|------|
| crazyflie_2 | 纳米无人机 |
| fetch_robot | 移动操作 |
| franka_panda | 协作臂 |
| g1 | 人形 |
| mock_mobile_base | Mock 移动底盘 |
| skydio_x2 | 无人机 |
| unitree_go2 | 四足 |
| ur5e | 工业臂 |

**UR5e 详细检查：**
- ✅ 6 DOF, 9 Links, 6 Joints
- ✅ 4 Sensors (tcp_force_sensor, joint_torque_sensors, wrist_camera, base_imu)
- ✅ Safety Level 信息完整

## 2. 当前有哪些 provider？

**8 个内置 providers：**

| Name | Type | 描述 |
|------|------|------|
| llm | llm | 文本生成和任务规划 |
| vlm | vlm | 物体定位和感知 |
| vla | vla | 端到端控制 |
| vln | vln | 移动机器人路径规划 |
| world | world | 场景理解 |
| skill | skill | 机器人动作执行 |
| critic | critic | 成功/失败判断 |
| embedding | embedding | 语义搜索 |

## 3. 当前有哪些 skill？

**5 个内置 skills：**

| Skill ID | Type | 描述 |
|----------|------|------|
| pid_move | motion | PID 控制移动 |
| reach | manipulation | 到达目标位姿 |
| grasp | manipulation | 抓取物体 |
| navigate | navigation | 导航到航点 |
| inspect | perception | 传感器检查目标 |

## 4. 当前有哪些 sandbox backend？

**4 个 sandbox worlds：**

| Name | 描述 |
|------|------|
| mock | 测试用 mock（无物理） |
| mujoco | MuJoCo 物理仿真 |
| tabletop | 桌面操作场景 |
| empty | 空世界（单元测试用） |

## 5. ROS2 是否可用？

✅ **YES**
- ROS_DISTRO: humble
- RMW_IMPLEMENTATION: rmw_fastrtps_cpp
- /opt/ros/humble/bin/ros2

## 6. mock runtime 是否可用？

✅ **YES** - mock sandbox world 可用

## 7. sim runtime 是否可用？

✅ **YES** - MuJoCo 3.9.0 已安装

## 8. dashboard 是否可用？

✅ **可用**
- Providers: 8 registered
- Skills: 5 registered
- Episodes: 4374 recorded
- 7 模块全部 HEALTHY
- Config: found (rosclaw.yaml)

## 9. practice / memory 是否可用？

**Practice:**
✅ 可用，有 4374 个历史 episodes
- 子命令: list, show, replay, export

**Memory:**
⚠️ CLI 显示 0 experiences（SeekDB 内存隔离）
- 但查询 episode artifact 可找到数据（已修复）

## CLI 命令完整度

| 命令组 | 可用命令 | 状态 |
|--------|----------|------|
| robot | list, install, inspect, validate | ✅ |
| provider | list, invoke | ✅ |
| skill | list, invoke | ✅ |
| sandbox | list-worlds, validate, run, replay | ✅ |
| memory | status, query, explain | ✅ |
| practice | list, show, replay, export | ✅ |
| how | explain, recover | ✅ |
| firewall | check | ✅ |
| forge | validate, install | ✅ |
| events | tail, publish | ✅ |
| know | search, robot, recommend | ✅ |
| demo | mobile-pid, tabletop-grasp | ✅ |
