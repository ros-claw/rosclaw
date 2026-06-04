"""rosclaw-auto 配置."""
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class AutoConfig:
    enabled: bool = True
    trigger_repeated_failure_threshold: int = 3
    trigger_min_failure_severity: str = "medium"
    trigger_benchmark_regression_threshold: float = 0.05
    trigger_idle_window_enabled: bool = True
    patch_allow_config_patch: bool = True
    patch_allow_skill_param_patch: bool = True
    patch_allow_skill_graph_patch: bool = True
    patch_allow_code_patch: bool = False
    patch_require_human_approval_for_code: bool = True
    storage_backend: Literal["local", "seekdb", "hybrid"] = "local"
    local_store_path: str = "./.rosclaw.auto"
    max_concurrent_experiments: int = 3
    max_rounds: int = 10
    default_episodes: int = 50
    default_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    default_policy: str = "failure_guided"
    promotion_min_success_improvement: float = 0.05
    promotion_max_collision_increase: float = 0.0
    promotion_require_second_seed: bool = True
    promotion_require_regression_check: bool = True
    sandbox_required: bool = True
    human_approval_for_skill_graph: bool = False
    human_approval_for_code: bool = True
    human_approval_for_policy: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "AutoConfig":
        return cls(
            enabled=d.get("enabled", True),
            trigger_repeated_failure_threshold=d.get("trigger_repeated_failure_threshold", 3),
            trigger_min_failure_severity=d.get("trigger_min_failure_severity", "medium"),
            trigger_benchmark_regression_threshold=d.get("trigger_benchmark_regression_threshold", 0.05),
            patch_allow_config_patch=d.get("patch_allow_config_patch", True),
            patch_allow_skill_param_patch=d.get("patch_allow_skill_param_patch", True),
            patch_allow_skill_graph_patch=d.get("patch_allow_skill_graph_patch", True),
            patch_allow_code_patch=d.get("patch_allow_code_patch", False),
            storage_backend=d.get("storage_backend", "local"),
            local_store_path=d.get("local_store_path", "./.rosclaw.auto"),
            max_concurrent_experiments=d.get("max_concurrent_experiments", 3),
            max_rounds=d.get("max_rounds", 10),
            default_episodes=d.get("default_episodes", 50),
            default_policy=d.get("default_policy", "failure_guided"),
        )
