"""ROSClaw Provider - Capability taxonomy.

Canonical capability names used across the provider ecosystem.
Upper layers MUST use these names; providers declare which they support.
"""

from dataclasses import dataclass


class CapabilityDomain:
    """Capability domain constants."""

    LLM = "llm"
    VLM = "vlm"
    VLA = "vla"
    VLN = "vln"
    WORLD = "world"
    SKILL = "skill"
    CRITIC = "critic"
    EMBEDDING = "embedding"
    GEOMETRY = "geometry"
    SEGMENTATION = "segmentation"
    NAVIGATION = "navigation"
    REASONING = "reasoning"


@dataclass(frozen=True)
class Capability:
    """A canonical capability identifier.

    Format: domain.name, e.g., "vlm.object_grounding"
    """

    domain: str
    name: str

    def __str__(self) -> str:
        return f"{self.domain}.{self.name}"

    @classmethod
    def parse(cls, s: str) -> "Capability":
        if "." not in s:
            raise ValueError(f"Invalid capability string: {s!r}")
        domain, name = s.split(".", 1)
        return cls(domain=domain, name=name)


# Canonical capability catalog (non-exhaustive; providers may extend)
CAPABILITY_CATALOG: dict[str, list[str]] = {
    CapabilityDomain.LLM: [
        "chat",
        "plan",
        "tool_call",
        "state_summarize",
        "failure_reflect",
        "code_generate",
        "memory_compress",
    ],
    CapabilityDomain.VLM: [
        "scene_understanding",
        "scene",                     # GPU provider alias
        "visual_question_answering",
        "vqa",                       # GPU provider alias
        "object_grounding",
        "grounding",                 # GPU provider alias
        "object_detection",
        "segmentation",
        "affordance_estimation",
        "anomaly_detection",
        "ocr",
    ],
    CapabilityDomain.VLA: [
        "action_proposal",
        "action_chunk",
        "pose_delta",
        "grasp_intent",
        "manipulation_policy",
    ],
    CapabilityDomain.VLN: [
        "next_waypoint",
        "route_plan",
        "object_goal_navigation",
        "instruction_following",
        "frontier_selection",
        "exploration_policy",
    ],
    CapabilityDomain.WORLD: [
        "predict_next_state",
        "simulate_action_outcome",
        "generate_future_video",
        "counterfactual_rollout",
        "risk_estimate",
        "synthetic_episode_generation",
        "sim2real_translation",
    ],
    CapabilityDomain.SKILL: [
        "grasp",
        "place",
        "pick_and_place",
        "open_door",
        "push",
        "pull",
        "insert",
        "navigate",
        "inspect",
        "scan",
        "follow",
    ],
    CapabilityDomain.CRITIC: [
        "success_detection",
        "safety_check",
        "constraint_violation",
        "retry_advice",
        "failure_reasoning",
        "memory_write_filter",
        "trajectory_quality",
    ],
    CapabilityDomain.EMBEDDING: [
        "text",
        "image",
        "video",
        "state",
        "trajectory",
        "episode",
    ],
    CapabilityDomain.GEOMETRY: [
        "depth",
        "camera_pose",
        "pose",
        "point_cloud",
        "pointcloud",
        "bev",
    ],
    CapabilityDomain.SEGMENTATION: [
        "mask",
        "track",
    ],
    CapabilityDomain.NAVIGATION: [
        "traversability",
        "costmap",
    ],
    CapabilityDomain.REASONING: [
        "physical",
        "spatial_temporal",
        "spatial",
        "risk_explain",
        "risk",
    ],
}


def is_valid_capability(capability: str) -> bool:
    """Check if a capability string is in the canonical catalog."""
    try:
        cap = Capability.parse(capability)
    except ValueError:
        return False
    return cap.name in CAPABILITY_CATALOG.get(cap.domain, [])


def list_capabilities(domain: str | None = None) -> list[str]:
    """List all canonical capabilities, optionally filtered by domain."""
    if domain:
        return [f"{domain}.{n}" for n in CAPABILITY_CATALOG.get(domain, [])]
    result: list[str] = []
    for d, names in CAPABILITY_CATALOG.items():
        result.extend(f"{d}.{n}" for n in names)
    return result
