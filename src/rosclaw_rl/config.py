"""
Configuration for ROSClaw RL training.

Defines training modes, hyperparameters, and system configurations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from pathlib import Path


class TrainingMode(Enum):
    """Supported training modes from OpenClaw-RL."""
    GRPO = "grpo"           # Group Relative Policy Optimization (binary RL)
    OPD = "opd"             # On-Policy Distillation
    COMBINE = "combine"     # Combined GRPO + OPD


class RewardMode(Enum):
    """Reward computation modes."""
    SCALAR = "scalar"           # Simple success/failure reward
    VLM_FEEDBACK = "vlm"        # VLM-based feedback
    MULTI_MODAL = "multimodal"  # Multi-modal PRM (Phase 2)


@dataclass
class DataConfig:
    """Data collection configuration."""
    # Trajectory settings
    max_trajectory_length: int = 1000
    min_trajectory_length: int = 10
    trajectory_buffer_size: int = 10000

    # Collection settings
    collection_frequency_hz: float = 10.0
    collection_batch_size: int = 32

    # Storage
    data_dir: Path = field(default_factory=lambda: Path("/tmp/rosclaw_rl/data"))
    trajectory_format: str = "parquet"  # parquet, jsonl, hdf5

    # Safety
    max_joint_velocity: float = 1.0  # rad/s
    max_end_effector_velocity: float = 0.5  # m/s
    emergency_stop_timeout: float = 0.1  # seconds


@dataclass
class RewardConfig:
    """Reward model configuration."""
    # Mode
    mode: RewardMode = RewardMode.SCALAR

    # Scalar rewards
    success_reward: float = 1.0
    failure_reward: float = -1.0
    step_penalty: float = -0.01

    # VLM feedback settings
    vlm_model: str = "qwen2.5-vl-7b"
    vlm_api_url: Optional[str] = None
    vlm_temperature: float = 0.2
    vlm_max_tokens: int = 256

    # Multi-modal settings (Phase 2)
    use_proprioception: bool = True
    use_vision: bool = True
    use_force: bool = False
    use_audio: bool = False

    # PRM settings
    prm_checkpoint: Optional[str] = None
    prm_vote_n: int = 3  # Number of votes for majority voting


@dataclass
class TrainingHyperparameters:
    """Training hyperparameters for different modes."""
    # Common
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # GRPO specific
    grpo_group_size: int = 8
    grpo_epsilon: float = 0.2
    grpo_kl_penalty: float = 0.01

    # OPD specific
    opd_teacher_temperature: float = 0.8
    opd_topk_threshold: float = 0.9
    opd_distillation_weight: float = 0.5

    # Combine specific
    combine_grpo_weight: float = 0.5
    combine_opd_weight: float = 0.5


@dataclass
class ModelConfig:
    """Model configuration for training."""
    # Base model
    base_model: str = "qwen2.5-vl-3b-instruct"
    checkpoint_dir: Optional[Path] = None

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Serving settings
    serving_port: int = 8000
    max_model_len: int = 4096
    tensor_parallel_size: int = 1


@dataclass
class SystemConfig:
    """System configuration."""
    # Compute
    num_gpus: int = 1
    num_cpu_workers: int = 4

    # Distributed
    distributed: bool = False
    world_size: int = 1
    rank: int = 0

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("/tmp/rosclaw_rl/output"))
    log_dir: Path = field(default_factory=lambda: Path("/tmp/rosclaw_rl/logs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("/tmp/rosclaw_rl/checkpoints"))

    # Logging
    log_level: str = "INFO"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


@dataclass
class ROSClawRLConfig:
    """Complete configuration for ROSClaw RL training."""
    # Mode
    training_mode: TrainingMode = TrainingMode.GRPO

    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    model: ModelConfig = field(default_factory=ModelConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    # Integration
    roboclaw_api_url: str = "http://localhost:8001"  # RoboClaw runtime API
    vla_serving_url: str = "http://localhost:8002"   # VLA model serving URL

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        def convert_value(v):
            if isinstance(v, Enum):
                return v.value
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, dataclass):
                return {k: convert_value(val) for k, val in v.__dict__.items()}
            return v

        return {k: convert_value(v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ROSClawRLConfig":
        """Create config from dictionary."""
        # Parse enums
        if "training_mode" in data:
            data["training_mode"] = TrainingMode(data["training_mode"])
        if "reward" in data and "mode" in data["reward"]:
            data["reward"]["mode"] = RewardMode(data["reward"]["mode"])

        # Parse paths
        for section in ["data", "model", "system"]:
            if section in data:
                for key, val in data[section].items():
                    if isinstance(val, str) and ("dir" in key or "path" in key):
                        data[section][key] = Path(val)

        return cls(
            training_mode=data.get("training_mode", TrainingMode.GRPO),
            data=DataConfig(**data.get("data", {})),
            reward=RewardConfig(**data.get("reward", {})),
            training=TrainingHyperparameters(**data.get("training", {})),
            model=ModelConfig(**data.get("model", {})),
            system=SystemConfig(**data.get("system", {})),
            roboclaw_api_url=data.get("roboclaw_api_url", "http://localhost:8001"),
            vla_serving_url=data.get("vla_serving_url", "http://localhost:8002"),
        )
