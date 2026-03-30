# ROSClaw 安全防火墙详细设计

> Digital Twin Firewall v2.0 - 从验证到预测性防护
> 融合来源: MJX, MuJoCo Warp, Pinocchio, awesome-robotics-libraries

---

## 一、架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ROSClaw Safety Firewall v2.0                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 4: Predictive Safety (预测性安全)                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Neural Twin (Cosmos/V-JEPA 2)                                        │ │
│  │  • 长程语义预测 (5-10秒)                                                │ │
│  │  • 物理常识推理 ("杯子会洒")                                            │ │
│  │  • 任务可行性评估                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓ 战略风险评估                                  │
│  Layer 3: Digital Twin (数字孪生)                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  MJX Firewall (GPU可微分仿真)                                          │ │
│  │  • 并行仿真: 1000+ 随机环境                                              │ │
│  │  • 域随机化: 摩擦/质量/质心                                              │ │
│  │  • 可微分验证: 梯度反向传播                                              │ │
│  │  • 延迟: < 10ms (CUDA Graph)                                            │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓ 战术安全验证                                  │
│  Layer 2: Analytical Check (解析验证)                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Pinocchio + Ruckig                                                   │ │
│  │  • 运动学验证: 工作空间/奇异点                                          │ │
│  │  • 动力学验证: 力矩/加速度限制                                          │ │
│  │  • 轨迹优化: 时间最优+急动度约束                                        │ │
│  │  • 延迟: < 1ms                                                          │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓ 硬约束检查                                    │
│  Layer 1: Hard Limits (硬限制)                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Joint Limits / Velocity Limits / Torque Limits                       │ │
│  │  • 关节限位: 位置/速度/加速度                                           │ │
│  │  • 紧急停止: 硬件级触发                                                 │ │
│  │  • 延迟: < 0.1ms                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、分层详细设计

### Layer 1: Hard Limits (硬限制层)

**目标**: 微秒级响应，硬件级安全

```python
# src/rosclaw/firewall/hard_limits.py
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass(frozen=True)
class RobotLimits:
    """机器人硬限制定义"""
    joint_position: Dict[str, Tuple[float, float]]  # 关节位置限制
    joint_velocity: Dict[str, Tuple[float, float]]  # 关节速度限制
    joint_acceleration: Dict[str, Tuple[float, float]]  # 加速度限制
    joint_torque: Dict[str, Tuple[float, float]]  # 力矩限制
    workspace: Dict[str, Tuple[float, float]]  # 工作空间限制 (x, y, z)
    collision_pairs: list  # 自碰撞检查对

class HardLimitChecker:
    """硬限制检查器 - < 0.1ms延迟"""

    def __init__(self, limits: RobotLimits):
        self.limits = limits
        self._cache_limits()  # 预计算加速

    def _cache_limits(self):
        """将限制转换为numpy数组加速计算"""
        self.pos_min = np.array([v[0] for v in self.limits.joint_position.values()])
        self.pos_max = np.array([v[1] for v in self.limits.joint_position.values()])
        self.vel_max = np.array([v[1] for v in self.limits.joint_velocity.values()])
        self.torque_max = np.array([v[1] for v in self.limits.joint_torque.values()])

    def check(self, q: np.ndarray, dq: np.ndarray, torque: np.ndarray) -> SafetyResult:
        """
        检查硬限制
        Args:
            q: 关节位置
            dq: 关节速度
            torque: 关节力矩
        Returns:
            SafetyResult: 是否安全，违规详情
        """
        violations = []

        # 位置限制检查
        if np.any(q < self.pos_min) or np.any(q > self.pos_max):
            violations.append("JOINT_POSITION_LIMIT")

        # 速度限制检查
        if np.any(np.abs(dq) > self.vel_max):
            violations.append("JOINT_VELOCITY_LIMIT")

        # 力矩限制检查
        if np.any(np.abs(torque) > self.torque_max):
            violations.append("JOINT_TORQUE_LIMIT")

        return SafetyResult(
            is_safe=len(violations) == 0,
            violations=violations,
            layer="HardLimits"
        )

    def emergency_stop(self, reason: str):
        """触发紧急停止"""
        # 发送硬件紧急停止信号
        pass
```

---

### Layer 2: Analytical Check (解析验证层)

**目标**: 毫秒级响应，运动学与动力学验证

```python
# src/rosclaw/firewall/analytical.py
import pinocchio as pin
from ruckig import InputParameter, OutputParameter, Ruckig

class AnalyticalSafetyChecker:
    """解析安全验证器 - Pinocchio + Ruckig"""

    def __init__(self, urdf_path: str):
        # 加载Pinocchio模型
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # 初始化Ruckig轨迹生成器
        self.ruckig = Ruckig(self.model.nq, 0.001)  # 1ms周期

    def check_kinematics(self, q: np.ndarray, target_q: np.ndarray) -> SafetyResult:
        """
        运动学安全检查
        - 工作空间检查
        - 奇异点检查
        - 可达性检查
        """
        # 正向运动学计算末端位置
        pin.forwardKinematics(self.model, self.data, q)
        tcp_pos = self.data.oMf[-1].translation

        # 检查工作空间
        workspace_violations = self._check_workspace(tcp_pos)

        # 检查奇异点 (雅可比矩阵条件数)
        J = pin.computeFrameJacobian(self.model, self.data, q, -1)
        cond = np.linalg.cond(J)
        is_singular = cond > 1e3

        return SafetyResult(
            is_safe=len(workspace_violations) == 0 and not is_singular,
            violations=workspace_violations + (["SINGULARITY"] if is_singular else []),
            layer="Analytical"
        )

    def check_dynamics(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> SafetyResult:
        """
        动力学安全检查
        - 计算所需力矩
        - 检查力矩限制
        - 检查功率限制
        """
        # 逆动力学计算所需力矩
        tau = pin.rnea(self.model, self.data, q, dq, ddq)

        # 检查力矩限制
        violations = []
        for i, (name, limit) in enumerate(self.torque_limits.items()):
            if abs(tau[i]) > limit:
                violations.append(f"TORQUE_EXCEEDED_{name}")

        return SafetyResult(
            is_safe=len(violations) == 0,
            violations=violations,
            predicted_torque=tau,
            layer="Analytical"
        )

    def optimize_trajectory(self, waypoints: list) -> Trajectory:
        """
        使用Ruckig进行轨迹优化
        - 时间最优
        - 急动度约束
        - 实时计算
        """
        inp = InputParameter(self.model.nq)
        inp.current_position = waypoints[0]
        inp.target_position = waypoints[-1]

        # 设置约束
        inp.max_velocity = self.vel_max
        inp.max_acceleration = self.acc_max
        inp.max_jerk = self.jerk_max

        out = OutputParameter(self.model.nq)
        result = self.ruckig.update(inp, out)

        if result == Ruckig.Result.Working:
            return Trajectory(out.trajectory)
        else:
            raise SafetyViolationError(f"Trajectory optimization failed: {result}")
```

---

### Layer 3: Digital Twin (MJX 可微分仿真层)

**目标**: 十毫秒级响应，GPU并行仿真验证

```python
# src/rosclaw/firewall/mjx_firewall.py
import jax
import jax.numpy as jnp
from mujoco import mjx

class MJXFirewall:
    """MJX GPU可微分防火墙"""

    def __init__(self, mjcf_path: str, batch_size: int = 1000):
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # 转换为MJX模型
        self.mjx_model = mjx.put_model(self.model)
        self.mjx_data = mjx.put_data(self.model, self.data)

        self.batch_size = batch_size

    @jax.jit
    def _batch_simulate(self, q_batch: jnp.ndarray, ctrl_batch: jnp.ndarray):
        """
        批量并行仿真
        Args:
            q_batch: [batch_size, nq] 初始位置
            ctrl_batch: [batch_size, nu] 控制序列
        Returns:
            仿真结果字典
        """
        def single_simulate(q, ctrl):
            data = self.mjx_data.replace(qpos=q, ctrl=ctrl)

            # 仿真100步 (100ms @ 1kHz)
            for _ in range(100):
                data = mjx.step(self.mjx_model, data)

            return {
                'qpos': data.qpos,
                'qvel': data.qvel,
                'qacc': data.qacc,
                'contact': data.contact,
                'sensordata': data.sensordata
            }

        # vmap批量处理
        return jax.vmap(single_simulate)(q_batch, ctrl_batch)

    def validate_with_domain_randomization(
        self,
        trajectory: np.ndarray,
        randomize_params: bool = True
    ) -> SafetyResult:
        """
        使用域随机化验证轨迹
        - 并行仿真1000个随机环境
        - 随机化: 摩擦、质量、质心、阻尼
        """
        # 生成随机环境参数
        if randomize_params:
            friction = np.random.uniform(0.5, 1.5, self.batch_size)
            mass_scale = np.random.uniform(0.9, 1.1, self.batch_size)
            # ... 更多随机化参数

        # 批量仿真
        results = self._batch_simulate(q_batch, ctrl_batch)

        # 分析结果
        collision_rate = jnp.mean(results['contact'] > 0)
        torque_violations = jnp.any(
            jnp.abs(results['sensordata']) > self.torque_limits,
            axis=1
        )

        return SafetyResult(
            is_safe=collision_rate < 0.01 and jnp.mean(torque_violations) < 0.05,
            collision_rate=float(collision_rate),
            torque_violation_rate=float(jnp.mean(torque_violations)),
            layer="MJX"
        )

    def gradient_based_optimization(self, initial_traj: np.ndarray) -> np.ndarray:
        """
        基于梯度的轨迹优化
        - 利用可微分仿真优化轨迹
        - 最小化能量消耗 + 最大化安全性
        """
        def loss_fn(traj):
            # 仿真轨迹
            result = self._batch_simulate(traj[:, :self.model.nq], traj[:, self.model.nq:])

            # 计算损失
            energy_cost = jnp.sum(jnp.square(result['qacc']))
            safety_cost = jnp.sum(result['contact'] > 0)

            return energy_cost + 1000 * safety_cost

        # 梯度下降优化
        grad_fn = jax.grad(loss_fn)
        optimized_traj = initial_traj - 0.01 * grad_fn(initial_traj)

        return optimized_traj
```

---

### Layer 4: Predictive Safety (预测性安全层)

**目标**: 百毫秒级响应，世界模型预测

```python
# src/rosclaw/firewall/predictive.py
class PredictiveSafetyChecker:
    """预测性安全检查器 - Neural Twin"""

    def __init__(self, world_model: str = "cosmos"):
        # 加载世界模型 (Cosmos Predict 2.5 / V-JEPA 2)
        self.world_model = load_world_model(world_model)

    def predict_long_term(self, current_state: State, action_sequence: list) -> Prediction:
        """
        长程语义预测 (5-10秒)
        - "如果我推倒这个杯子，水会洒出来"
        - "抓取这个物体需要什么样的姿势"
        """
        prediction = self.world_model.predict(
            current_state.image,
            current_state.physics,
            action_sequence
        )

        return Prediction(
            outcome=prediction.outcome,  # 预测结果
            confidence=prediction.confidence,
            risks=prediction.risks,  # 风险点
            time_horizon=prediction.time_horizon
        )

    def check_task_feasibility(self, task_description: str, current_state: State) -> FeasibilityResult:
        """
        任务可行性评估
        - "我能否在5秒内到达那个位置？"
        - "这个操作会成功吗？"
        """
        feasibility = self.world_model.evaluate(
            task_description,
            current_state
        )

        return FeasibilityResult(
            is_feasible=feasibility.score > 0.7,
            score=feasibility.score,
            alternative_plans=feasibility.alternatives,
            estimated_time=feasibility.time_estimate
        )
```

---

## 三、防火墙编排器

```python
# src/rosclaw/firewall/orchestrator.py
class SafetyFirewallOrchestrator:
    """安全防火墙编排器 - 四层协同"""

    def __init__(self, robot_config: RobotConfig):
        self.layer1 = HardLimitChecker(robot_config.limits)
        self.layer2 = AnalyticalSafetyChecker(robot_config.urdf_path)
        self.layer3 = MJXFirewall(robot_config.mjcf_path)
        self.layer4 = PredictiveSafetyChecker()

    def validate_command(self, command: RobotCommand) -> ValidationResult:
        """
        四层递进式验证
        """
        # Layer 1: 硬限制 (< 0.1ms)
        result1 = self.layer1.check(command.q, command.dq, command.torque)
        if not result1.is_safe:
            return ValidationResult(approved=False, reason=result1.violations)

        # Layer 2: 解析验证 (< 1ms)
        result2_kin = self.layer2.check_kinematics(command.q, command.target_q)
        result2_dyn = self.layer2.check_dynamics(command.q, command.dq, command.ddq)
        if not (result2_kin.is_safe and result2_dyn.is_safe):
            # 尝试优化轨迹
            try:
                optimized_traj = self.layer2.optimize_trajectory(command.waypoints)
                command.waypoints = optimized_traj
            except SafetyViolationError:
                return ValidationResult(approved=False, reason="Kinematics/Dynamics check failed")

        # Layer 3: MJX仿真 (< 10ms)
        result3 = self.layer3.validate_with_domain_randomization(command.waypoints)
        if not result3.is_safe:
            if result3.collision_rate > 0.5:
                return ValidationResult(approved=False, reason="High collision probability")
            # 尝试梯度优化
            command.waypoints = self.layer3.gradient_based_optimization(command.waypoints)

        # Layer 4: 预测性安全 (< 100ms)
        result4 = self.layer4.check_task_feasibility(command.description, command.current_state)
        if not result4.is_feasible:
            return ValidationResult(
                approved=False,
                reason=f"Task not feasible: {result4.alternative_plans}"
            )

        return ValidationResult(
            approved=True,
            optimized_command=command,
            confidence=min(result3.collision_rate, result4.score)
        )
```

---

## 四、性能指标

| 层级 | 延迟目标 | 技术实现 | 应用场景 |
|------|----------|----------|----------|
| Layer 1 | < 0.1ms | NumPy向量化 | 实时控制循环 |
| Layer 2 | < 1ms | Pinocchio C++ | 轨迹生成 |
| Layer 3 | < 10ms | JAX GPU | 复杂动作验证 |
| Layer 4 | < 100ms | World Model | 任务规划 |

---

## 五、安全策略配置

```yaml
# safety_config.yaml
firewall:
  layers:
    hard_limits:
      enabled: true
      emergency_stop_on_violation: true

    analytical:
      enabled: true
      check_workspace: true
      check_singularity: true
      auto_optimize_trajectory: true

    mjx:
      enabled: true
      batch_size: 1000
      domain_randomization:
        friction_range: [0.5, 1.5]
        mass_scale_range: [0.9, 1.1]
      gradient_optimization:
        enabled: true
        max_iterations: 10

    predictive:
      enabled: false  # 需要GPU资源，默认关闭
      world_model: "cosmos"
      prediction_horizon: 5.0  # 秒

  safety_levels:
    strict:  # 严格模式 - 所有检查必须通过
      - layer1
      - layer2
      - layer3

    moderate:  # 适中模式 - 允许小幅违规
      - layer1
      - layer2
      # layer3: 警告但不阻止

    minimal:  # 最小模式 - 仅硬限制
      - layer1
```
