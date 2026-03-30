# ROSClaw Skill 市场详细设计

> ClawHub for Robotics - 跨硬件技能共享生态
> 融合来源: OpenClaw Skill系统, ClawHub, Cross-Embodiment研究

---

## 一、核心概念

### 1.1 什么是 Skill？

Skill = 可复用的机器人能力单元，包含：
- **配置 (Config)**: 任务参数、约束条件
- **策略 (Policy)**: VLA模型权重
- **提示 (Prompt)**: LLM操作指令
- **安全 (Safety)**: 硬件限制、边界条件

### 1.2 Skill 分类

| 类型 | 描述 | 示例 |
|------|------|------|
| **原子Skill** | 单一基础动作 | `grasp`, `move_to`, `rotate` |
| **复合Skill** | 原子Skill组合 | `pick_and_place`, `screw_driving` |
| **自适应Skill** | 跨硬件可迁移 | `universal_grasp` (UR5/G1/Franka) |
| **交互Skill** | 人机协作 | `handover`, `collaborative_assembly` |

---

## 二、Skill 格式规范

### 2.1 文件结构

```
precision_screwdriving.rosclaw/
├── skill.yaml              # 技能元数据
├── config/
│   ├── task_config.yaml    # 任务参数
│   ├── safety_bounds.yaml  # 安全限制
│   └── motion_primitives.yaml  # 运动基元
├── policy/
│   ├── weights.bin         # VLA模型权重
│   ├── tokenizer.json      # 动作tokenizer
│   └── config.json         # 模型配置
├── prompt/
│   ├── system_prompt.txt   # 系统提示词
│   ├── examples.jsonl      # 示例对话
│   └── constraints.txt     # 约束说明
├── assets/
│   ├── tool_3d_model.stl   # 工具3D模型
│   ├── trajectory_template.pkl  # 轨迹模板
│   └── calibration_data/   # 标定数据
└── tests/
    ├── validation_scenarios.yaml
    └── test_cases/
```

### 2.2 skill.yaml 规范

```yaml
# skill.yaml - 技能元数据
schema_version: "2.0"

metadata:
  name: "precision_screwdriving"
  version: "1.2.0"
  author: "tokyo_dev"
  organization: "rosclaw-community"
  license: "Apache-2.0"
  description: "高精度螺丝拧紧技能，适用于M3-M6螺丝"
  tags: ["manipulation", "assembly", "screwdriving"]
  icon: "assets/icon.png"

  # 版本历史
  changelog:
    - version: "1.2.0"
      date: "2026-03-26"
      changes: ["添加G1支持", "改进力控"]
    - version: "1.1.0"
      date: "2026-02-15"
      changes: ["添加力矩反馈"]

# 硬件兼容性
compatibility:
  # 支持的机器人类型
  robots:
    - type: "ur5e"
      end_effector: "robotiq_2f85"
      tested_versions: ["5.13.0", "5.14.0"]

    - type: "unitree_g1"
      end_effector: "inspire_dexterous_hand"
      tested_versions: ["1.2.0"]
      requires_adaptation: true

    - type: "franka_panda"
      end_effector: "franka_hand"
      tested_versions: ["4.2.0"]

  # 最小硬件要求
  requirements:
    camera: "rgbd"  # rgb, rgbd, stereo
    force_sensor: true
    gripper: "parallel"  # parallel, dexterous, vacuum

# 依赖
dependencies:
  rosclaw_version: ">=0.2.0"
  python_packages:
    - "torch>=2.0.0"
    - "transformers>=4.30.0"
  ros_packages:
    - "moveit_ros_planning"
    - "force_torque_sensor"
  skills:  # 依赖的其他skill
    - name: "force_control"
      version: ">=1.0.0"

# 能力声明
capabilities:
  inputs:
    - name: "screw_location"
      type: "pose"
      description: "螺丝目标位置"

    - name: "screw_type"
      type: "enum"
      options: ["M3", "M4", "M5", "M6"]
      default: "M4"

    - name: "target_torque"
      type: "float"
      unit: "Nm"
      range: [0.5, 5.0]
      default: 2.0

  outputs:
    - name: "success"
      type: "bool"

    - name: "actual_torque"
      type: "float"
      unit: "Nm"

    - name: "execution_time"
      type: "float"
      unit: "seconds"

# 安全参数
safety:
  max_force: 50.0  # N
  max_torque: 5.0  # Nm
  timeout: 30.0  # seconds
  emergency_stop_on_over_torque: true

# 性能指标
performance:
  success_rate: 0.95
  avg_execution_time: 8.5
  sim_to_real_gap: 0.05  # 5%性能下降
```

---

## 三、Skill 运行时

### 3.1 Skill 加载器

```python
# src/rosclaw/skills/loader.py
from pathlib import Path
import yaml
import torch

class SkillLoader:
    """Skill加载器"""

    def load(self, skill_path: Path) -> Skill:
        """加载Skill包"""
        # 读取元数据
        with open(skill_path / "skill.yaml") as f:
            metadata = yaml.safe_load(f)

        # 验证兼容性
        self._check_compatibility(metadata)

        # 加载策略
        policy = self._load_policy(skill_path / "policy")

        # 加载配置
        config = self._load_config(skill_path / "config")

        # 加载提示词
        prompts = self._load_prompts(skill_path / "prompt")

        return Skill(
            metadata=metadata,
            policy=policy,
            config=config,
            prompts=prompts
        )

    def _check_compatibility(self, metadata: dict):
        """检查硬件兼容性"""
        robot_type = get_current_robot_type()
        compatible_robots = [r["type"] for r in metadata["compatibility"]["robots"]]

        if robot_type not in compatible_robots:
            # 检查是否支持自适应
            if metadata.get("adaptive", False):
                logger.warning(f"Skill requires cross-embodiment adaptation for {robot_type}")
            else:
                raise CompatibilityError(f"Skill not compatible with {robot_type}")
```

### 3.2 Skill 执行器

```python
# src/rosclaw/skills/executor.py
class SkillExecutor:
    """Skill执行器"""

    def __init__(self, skill: Skill, robot_interface: RobotInterface):
        self.skill = skill
        self.robot = robot_interface
        self.safety_checker = SafetyFirewall()

    async def execute(self, inputs: dict) -> ExecutionResult:
        """
        执行Skill
        """
        # 1. 输入验证
        validated_inputs = self._validate_inputs(inputs)

        # 2. 加载系统提示词
        system_prompt = self.skill.prompts["system_prompt"]
        system_prompt = system_prompt.format(**validated_inputs)

        # 3. VLA推理
        observation = self.robot.get_observation()

        with torch.no_grad():
            action_sequence = self.skill.policy.predict(
                image=observation.image,
                instruction=system_prompt
            )

        # 4. 安全检查
        for action in action_sequence:
            result = self.safety_checker.validate(action)
            if not result.is_safe:
                return ExecutionResult(
                    success=False,
                    error=f"Safety check failed: {result.violations}"
                )

        # 5. 执行动作序列
        execution_log = []
        for i, action in enumerate(action_sequence):
            try:
                result = await self.robot.execute(action)
                execution_log.append(result)

                # 实时反馈
                if i % 10 == 0:  # 每10步检查一次
                    progress = i / len(action_sequence)
                    logger.info(f"Skill execution progress: {progress:.1%}")

            except ExecutionError as e:
                # 尝试恢复
                recovery = await self._attempt_recovery(e)
                if not recovery.success:
                    return ExecutionResult(
                        success=False,
                        error=f"Execution failed at step {i}: {e}",
                        log=execution_log
                    )

        # 6. 返回结果
        return ExecutionResult(
            success=True,
            outputs=self._collect_outputs(execution_log),
            log=execution_log
        )

    async def _attempt_recovery(self, error: ExecutionError) -> RecoveryResult:
        """尝试错误恢复"""
        # 使用Skill中定义的恢复策略
        recovery_strategy = self.skill.config.get("recovery_strategy", "abort")

        if recovery_strategy == "retry":
            return await self._retry_action()
        elif recovery_strategy == "alternative":
            return await self._try_alternative()
        else:
            return RecoveryResult(success=False)
```

---

## 四、跨硬件迁移 (Cross-Embodiment)

### 4.1 动作空间统一

```python
# src/rosclaw/skills/cross_embodiment.py
class CrossEmbodimentAdapter:
    """跨硬件迁移适配器"""

    def __init__(self, source_robot: str, target_robot: str):
        self.source = source_robot
        self.target = target_robot

        # 加载迁移网络
        self.adapter_network = self._load_adapter()

    def adapt_action(self, source_action: np.ndarray, context: dict) -> np.ndarray:
        """
        将源机器人动作迁移到目标机器人

        方法1: 重定向 (Retargeting)
        - 基于运动学链映射
        - 保持末端执行器轨迹

        方法2: 学习适配器 (Learned Adapter)
        - 神经网络学习映射
        - 需要配对数据训练

        方法3: 动作Tokenization
        - 将动作离散化为tokens
        - 在token空间统一表示
        """

        # 获取当前状态
        source_state = context["source_state"]
        target_state = context["target_state"]

        # 方法1: 重定向
        if self.adaptation_method == "retargeting":
            # 源机器人末端位置
            source_ee_pos = self._forward_kinematics(
                self.source, source_action
            )

            # 目标机器人逆运动学
            target_action = self._inverse_kinematics(
                self.target, source_ee_pos
            )

        # 方法2: 学习适配器
        elif self.adaptation_method == "learned":
            target_action = self.adapter_network(
                source_action, source_state, target_state
            )

        # 方法3: Tokenization
        elif self.adaptation_method == "token":
            source_tokens = self._action_to_tokens(source_action)
            target_tokens = self.adapter_network(source_tokens)
            target_action = self._tokens_to_action(target_tokens)

        return target_action
```

### 4.2 Skill 迁移流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Skill Cross-Embodiment Migration                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: 分析源Skill                                                         │
│  ├── 提取运动基元 (Motion Primitives)                                        │
│ ├── 识别接触模式 (Contact Patterns)                                          │
│  └── 标记关键约束 (Key Constraints)                                          │
│                              ↓                                              │
│  Step 2: 硬件差异分析                                                        │
│  ├── 自由度对比 (DoF Comparison)                                             │
│  ├── 工作空间映射 (Workspace Mapping)                                        │
│  └── 动力学适配 (Dynamics Adaptation)                                        │
│                              ↓                                              │
│  Step 3: 动作重定向                                                          │
│  ├── 末端执行器轨迹保持                                                      │
│  ├── 关节空间映射                                                            │
│  └── 时间参数化调整                                                          │
│                              ↓                                              │
│  Step 4: 仿真验证 (MJX)                                                      │
│  ├── 目标机器人仿真                                                          │
│  ├── 域随机化测试                                                            │
│  └── 成功率评估                                                              │
│                              ↓                                              │
│  Step 5: 真实机器人微调                                                      │
│  ├── 少量真实数据收集                                                        │
│  ├── LoRA微调 (VLA-Adapter)                                                  │
│  └── 部署验证                                                                │
│                              ↓                                              │
│  Step 6: 发布迁移后Skill                                                     │
│  ├── 更新兼容性列表                                                          │
│  ├── 记录迁移参数                                                            │
│  └── 贡献到社区                                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 五、rosclaw-hub 市场

### 5.1 架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           rosclaw-hub                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Web Interface                    API Gateway                               │
│  ┌─────────────┐                  ┌─────────────────────────────────────┐  │
│  │  Skill浏览   │                  │  /api/v1/skills/search              │  │
│  │  分类筛选    │                  │  /api/v1/skills/download            │  │
│  │  评分评论    │                  │  /api/v1/skills/upload              │  │
│  │  在线测试    │                  │  /api/v1/skills/migrate             │  │
│  └─────────────┘                  └─────────────────────────────────────┘  │
│                                                                             │
│  Core Services                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Skill Registry│  │ Compatibility│  │ Migration    │  │ Analytics       │  │
│  │  - 元数据管理  │  │  - 硬件匹配   │  │  - 跨硬件迁移  │  │  - 下载统计     │  │
│  │  - 版本控制   │  │  - 依赖检查   │  │  - 自动适配   │  │  - 成功率追踪   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘  │
│                                                                             │
│  Storage                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Object Storage (S3/R2)                                              │   │
│  │  ├── skills/          # Skill包存储                                  │   │
│  │  ├── policies/        # 模型权重                                     │   │
│  │  └── datasets/        # 训练数据                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 CLI 工具

```bash
# 搜索Skill
rosclaw skills search "screwdriving" --robot ur5e

# 安装Skill
rosclaw skills install precision_screwdriving

# 列出已安装
rosclaw skills list

# 更新Skill
rosclaw skills update precision_screwdriving

# 卸载Skill
rosclaw skills uninstall precision_screwdriving

# 迁移Skill到新机器人
rosclaw skills migrate precision_screwdriving --from ur5e --to unitree_g1

# 发布Skill
rosclaw skills publish ./my_skill.rosclaw --tag "v1.0.0"

# 测试Skill (仿真)
rosclaw skills test precision_screwdriving --simulator mujoco
```

### 5.3 质量评级系统

```python
# Skill质量评分算法
class SkillQualityScore:
    """
    多维度Skill质量评分
    """

    def calculate(self, skill: Skill) -> Score:
        metrics = {
            # 技术质量 (40%)
            'sim_success_rate': self._test_in_simulation(skill),  # 仿真成功率
            'real_success_rate': self._get_reported_success(skill),  # 真实成功率
            'execution_time': self._benchmark_performance(skill),  # 执行效率

            # 兼容性 (30%)
            'hardware_coverage': len(skill.compatibility.robots),  # 支持硬件数
            'cross_embodiment': skill.metadata.get('adaptive', False),  # 是否自适应

            # 社区反馈 (20%)
            'user_rating': self._get_user_rating(skill),  # 用户评分
            'download_count': self._get_downloads(skill),  # 下载量
            'issue_resolution': self._get_issue_stats(skill),  # 问题解决率

            # 文档质量 (10%)
            'doc_completeness': self._check_documentation(skill),  # 文档完整度
            'example_count': len(skill.prompts.get('examples', [])),  # 示例数量
        }

        # 加权计算
        weights = {
            'sim_success_rate': 0.15,
            'real_success_rate': 0.15,
            'execution_time': 0.10,
            'hardware_coverage': 0.15,
            'cross_embodiment': 0.15,
            'user_rating': 0.10,
            'download_count': 0.05,
            'issue_resolution': 0.05,
            'doc_completeness': 0.05,
            'example_count': 0.05,
        }

        total_score = sum(metrics[k] * weights[k] for k in metrics)

        return Score(
            total=total_score,
            breakdown=metrics,
            tier=self._get_tier(total_score)  # gold/silver/bronze
        )
```

---

## 六、Skill Flywheel 集成

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Skill Flywheel Data Flow                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  数据采集层                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Teleoperation (OmniH2O/HumanPlus)                                     │ │
│  │  ├── Vision Pro 遥操作                                                  │ │
│  │  ├── 同构外骨骼数据                                                      │ │
│  │  └── 实时运动重定向                                                      │ │
│  │                                                                        │ │
│  │  Autonomous Execution                                                    │ │
│  │  ├── VLA策略执行                                                        │ │
│  │  ├── 成功/失败标记                                                      │ │
│  │  └── 用户反馈收集                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                              │
│  数据飞轮 (Event-Driven Ring Buffer)                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  触发条件                                                              │ │
│  │  ├── 任务成功 → 保存 [T-10s, T+5s]                                     │ │
│  │  ├── 任务失败 → 保存 [T-10s, T+10s]                                    │ │
│  │  ├── 紧急停止 → 保存完整序列                                           │ │
│  │  └── 用户标注 → 标记高质量数据                                         │ │
│  │                                                                        │ │
│  │  数据格式: LeRobot Dataset                                              │ │
│  │  ├── images/                                                            │ │
│  │  ├── actions/                                                           │ │
│  │  ├── states/                                                            │ │
│  │  └── metadata/                                                          │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                              │
│  训练层                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  VLA Fine-tuning                                                        │ │
│  │  ├── LoRA (轻量微调)                                                    │ │
│  │  ├── VLA-Adapter (适配器)                                               │ │
│  │  └── Full Fine-tune (全量微调)                                          │ │
│  │                                                                        │ │
│  │  RL Optimization (VLA-RL)                                               │ │
│  │  ├── Online RL                                                          │ │
│  │  ├── ConRFT (对比强化微调)                                              │ │
│  │  └── Human Feedback (RLHF)                                              │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                              │
│  验证层                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  MJX仿真验证                                                            │ │
│  │  ├── 域随机化测试                                                        │ │
│  │  ├── 成功率评估                                                          │ │
│  │  └── 安全性检查                                                          │ │
│  │                                                                        │ │
│  │  真实机器人验证                                                          │ │
│  │  ├── 小规模测试                                                          │ │
│  │  ├── A/B对比                                                             │ │
│  │  └── 性能监控                                                            │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                              │
│  发布层                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  版本更新                                                                │ │
│  │  ├── 更新版本号                                                          │ │
│  │  ├── 更新changelog                                                       │ │
│  │  └── 更新性能指标                                                        │ │
│  │                                                                        │ │
│  │  社区分享                                                                │ │
│  │  ├── 发布到rosclaw-hub                                                   │ │
│  │  ├── 跨硬件迁移                                                          │ │
│  │  └── 社区反馈收集                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 七、总结

| 组件 | 核心能力 | 技术亮点 |
|------|----------|----------|
| **Skill格式** | 标准化打包 | .rosclaw格式，包含policy+prompt+safety |
| **Skill运行时** | 加载+执行+恢复 | VLA推理，安全检查，错误恢复 |
| **跨硬件迁移** | 一键适配新机器人 | 重定向/学习适配器/tokenization |
| **rosclaw-hub** | 技能市场 | 搜索/安装/迁移/质量评级 |
| **Skill Flywheel** | 数据闭环 | 遥操作→数据→训练→验证→发布 |

**最终愿景**: 东京开发者教机械臂拧螺丝 → Skill Flywheel自动优化 → ClawHub共享 → 柏林工厂G1机器人下载执行
