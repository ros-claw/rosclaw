"""ROSClaw Provider - ProviderManifest.

Python representation of provider.yaml manifest.
Loaded once at registration time; drives routing, guard, and runtime decisions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rosclaw.provider.core.errors import ManifestValidationError


@dataclass
class RuntimeSpec:
    """Runtime configuration from manifest."""

    backend: str = ""          # python, http, grpc, ros2, ollama, vllm, triton, onnx, tensorrt, isaac
    protocol: str = ""         # http, grpc, action, service, topic
    endpoint: str = ""         # e.g., http://localhost:11434/api/generate
    device: str = "cpu"        # cpu, cuda, cuda:0, auto
    min_vram_gb: float = 0.0
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RuntimeSpec":
        return cls(
            backend=d.get("backend", ""),
            protocol=d.get("protocol", ""),
            endpoint=d.get("endpoint", ""),
            device=d.get("device", "cpu"),
            min_vram_gb=d.get("min_vram_gb", 0.0),
            env=d.get("env", {}),
        )


@dataclass
class ModelSpec:
    """Model configuration from manifest."""

    name: str = ""
    source: str = ""           # hf, local, url
    model_id: str = ""         # e.g., openvla/openvla-7b
    precision: str = ""
    quantization: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        return cls(
            name=d.get("name", ""),
            source=d.get("source", ""),
            model_id=d.get("model_id", ""),
            precision=d.get("precision", ""),
            quantization=d.get("quantization", ""),
        )


@dataclass
class EmbodimentSpec:
    """Embodiment constraints from manifest."""

    supported_robots: list[str] = field(default_factory=list)
    camera_setup: list[str] = field(default_factory=list)
    action_space: list[str] = field(default_factory=list)
    control_frequency_hz: int = 0
    requires_calibration: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EmbodimentSpec":
        return cls(
            supported_robots=d.get("supported_robots", []),
            camera_setup=d.get("camera_setup", []),
            action_space=d.get("action_space", []),
            control_frequency_hz=d.get("control_frequency_hz", 0),
            requires_calibration=d.get("requires_calibration", False),
        )


@dataclass
class SafetySpec:
    """Safety declarations from manifest."""

    executable: bool = False           # If true, output can directly drive runtime
    requires_guard: bool = True        # Must pass guard before execution
    requires_collision_check: bool = False
    requires_workspace_check: bool = False
    max_action_norm: float = 0.0
    fallback_provider: str = ""        # Name of fallback provider if this fails

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SafetySpec":
        return cls(
            executable=d.get("executable", False),
            requires_guard=d.get("requires_guard", True),
            requires_collision_check=d.get("requires_collision_check", False),
            requires_workspace_check=d.get("requires_workspace_check", False),
            max_action_norm=d.get("max_action_norm", 0.0),
            fallback_provider=d.get("fallback_provider", ""),
        )


@dataclass
class ObservabilitySpec:
    """Observability settings from manifest."""

    log_inputs: bool = False
    log_outputs: bool = True
    trace_level: str = "standard"      # minimal, standard, detailed

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObservabilitySpec":
        return cls(
            log_inputs=d.get("log_inputs", False),
            log_outputs=d.get("log_outputs", True),
            trace_level=d.get("trace_level", "standard"),
        )


@dataclass
class ProviderManifest:
    """Complete provider manifest loaded from provider.yaml.

    This is the static contract that ROSClaw uses to decide:
    - Whether this provider can handle a request (capabilities, embodiment, modalities)
    - How to invoke it (runtime spec)
    - What safety checks to apply (safety spec)
    - How to observe it (observability spec)
    """

    name: str
    version: str
    type: str                          # llm, vlm, vla, vln, world, skill, critic, embedding
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    modalities: dict[str, list[str]] = field(default_factory=dict)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    embodiment: EmbodimentSpec = field(default_factory=EmbodimentSpec)
    safety: SafetySpec = field(default_factory=SafetySpec)
    observability: ObservabilitySpec = field(default_factory=ObservabilitySpec)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProviderManifest":
        p = Path(path)
        if not p.exists():
            raise ManifestValidationError(f"Manifest file not found: {p}")
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ManifestValidationError(f"Invalid YAML in {p}: expected dict, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProviderManifest":
        required = {"name", "version", "type"}
        missing = required - set(d.keys())
        if missing:
            raise ManifestValidationError(f"Missing required manifest fields: {missing}")

        return cls(
            name=d["name"],
            version=d["version"],
            type=d["type"],
            description=d.get("description", ""),
            capabilities=d.get("capabilities", []),
            modalities=d.get("modalities", {}),
            runtime=RuntimeSpec.from_dict(d.get("runtime", {})),
            model=ModelSpec.from_dict(d.get("model", {})),
            embodiment=EmbodimentSpec.from_dict(d.get("embodiment", {})),
            safety=SafetySpec.from_dict(d.get("safety", {})),
            observability=ObservabilitySpec.from_dict(d.get("observability", {})),
            extra={k: v for k, v in d.items() if k not in {
                "name", "version", "type", "description", "capabilities",
                "modalities", "runtime", "model", "embodiment", "safety", "observability",
            }},
        )

    def supports_capability(self, capability: str) -> bool:
        return capability in self.capabilities

    def supports_robot(self, robot_id: str) -> bool:
        return robot_id in self.embodiment.supported_robots or not self.embodiment.supported_robots

    def supports_input_modality(self, modality: str) -> bool:
        inputs = self.modalities.get("input", [])
        # Empty modalities declaration means universal support
        return not inputs or modality in inputs

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "type": self.type,
            "description": self.description,
            "capabilities": self.capabilities,
            "modalities": self.modalities,
            "runtime": {
                "backend": self.runtime.backend,
                "protocol": self.runtime.protocol,
                "endpoint": self.runtime.endpoint,
                "device": self.runtime.device,
                "min_vram_gb": self.runtime.min_vram_gb,
            },
            "model": {
                "name": self.model.name,
                "model_id": self.model.model_id,
            },
            "embodiment": {
                "supported_robots": self.embodiment.supported_robots,
                "action_space": self.embodiment.action_space,
            },
            "safety": {
                "executable": self.safety.executable,
                "requires_guard": self.safety.requires_guard,
                "fallback_provider": self.safety.fallback_provider,
            },
        }
