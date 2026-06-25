"""Pydantic v2 models and dataclasses for ROSClaw Skill Hub packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import semver
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_SKILL_SCHEMA = "rosclaw.skill.v1"
SUPPORTED_POLICY_SCHEMA = "rosclaw.policy.v1"
SUPPORTED_PROVIDERS_SCHEMA = "rosclaw.providers.v1"
SUPPORTED_EURDF_COMPAT_SCHEMA = "rosclaw.eurdf_compat.v1"
SUPPORTED_SAFETY_SCHEMA = "rosclaw.safety.v1"
SUPPORTED_DOJO_SCHEMA = "rosclaw.dojo.v1"
SUPPORTED_DARWIN_EVAL_SCHEMA = "rosclaw.darwin_eval.v1"
SUPPORTED_LINEAGE_SCHEMA = "rosclaw.lineage.v1"

STAGES = {"draft", "candidate", "validated", "deprecated", "revoked", "source_verified", "ci_passed", "official_verified", "installable"}
VALID_NAME_PATTERN = r"^[a-z0-9_\-]+$"
VALID_NAMESPACE_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping, got {type(data).__name__}: {path}")
    return data


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: ValidationReport) -> None:
        if not other.ok:
            self.ok = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.checks.update(other.checks)


@dataclass
class EvalReport:
    skill: str
    candidate_id: str | None
    version: str
    timestamp: str = field(default_factory=_now_iso)
    stage: str = "candidate"
    mode: str = "replay"
    metrics: dict[str, Any] = field(default_factory=dict)
    decision: str = "pending"
    artifacts: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rosclaw.eval_report.v1",
            "skill": self.skill,
            "candidate_id": self.candidate_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "mode": self.mode,
            "metrics": self.metrics,
            "decision": self.decision,
            "artifacts": self.artifacts,
            "checks": self.checks,
        }


@dataclass
class MiningReport:
    candidate_id: str
    source_episodes: list[str] = field(default_factory=list)
    score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    generated_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rosclaw.mining_report.v1",
            "candidate_id": self.candidate_id,
            "source_episodes": self.source_episodes,
            "score": self.score,
            "metrics": self.metrics,
            "generated_files": self.generated_files,
        }


# ---------------------------------------------------------------------------
# Skill package schemas
# ---------------------------------------------------------------------------


class Author(BaseModel):
    name: str
    url: str | None = None


class SkillMetadata(BaseModel):
    name: str = Field(pattern=VALID_NAME_PATTERN)
    display_name: str | None = None
    namespace: str = Field(default="ros-claw", pattern=VALID_NAMESPACE_PATTERN)
    version: str = Field(default="0.1.0")
    stage: str = Field(default="draft")
    candidate_id: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    license: str = "MIT"
    authors: list[Author] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        semver.VersionInfo.parse(value)
        return value

    @field_validator("stage")
    @classmethod
    def _validate_stage(cls, value: str) -> str:
        if value not in STAGES:
            raise ValueError(f"stage must be one of {sorted(STAGES)}, got {value!r}")
        return value


class SkillIdentity(BaseModel):
    skill_id: str | None = None
    package_name: str | None = None
    canonical_uri: str | None = None
    git_repo: str | None = None
    git_commit: str | None = None


class InputOutputContract(BaseModel):
    required: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)


class TaskSpec(BaseModel):
    intent: str = ""
    natural_language: dict[str, str] = Field(default_factory=dict)
    input_contract: InputOutputContract = Field(default_factory=InputOutputContract)
    output_contract: InputOutputContract = Field(default_factory=InputOutputContract)


class EntrypointSpec(BaseModel):
    type: str = "behavior_tree"
    file: str = "behavior_tree.xml"


class ExecutionSpec(BaseModel):
    entrypoint: EntrypointSpec = Field(default_factory=EntrypointSpec)
    runtime_adapter: str = "rosclaw.runtime.skill_executor"
    policy: dict[str, Any] = Field(default_factory=dict)
    prompts: dict[str, str] = Field(default_factory=dict)


class CompatibilitySpec(BaseModel):
    eurdf: str = "e-urdf-compat.yaml"
    providers: str = "providers.yaml"
    safety: str = "safety.yaml"


class EvaluationSpec(BaseModel):
    dojo: str = "dojo.yaml"
    darwin: str = "darwin_eval.yaml"
    tests: str = "tests/"


class LineageRef(BaseModel):
    file: str = "lineage.yaml"


class EvidenceSpec(BaseModel):
    directory: str = "evidence/"
    latest_eval_report: str | None = None


class StatusSpec(BaseModel):
    promotion_state: str = "draft"
    last_eval_passed: bool = False
    safe_to_run_on_real_robot: bool = False
    recommended_runtime_mode: str = "sandbox_first"


class SkillYaml(BaseModel):
    schema_version: Literal["rosclaw.skill.v1"] = SUPPORTED_SKILL_SCHEMA
    kind: Literal["Skill"] = "Skill"
    metadata: SkillMetadata = Field(default_factory=SkillMetadata)
    identity: SkillIdentity = Field(default_factory=SkillIdentity)
    task: TaskSpec = Field(default_factory=TaskSpec)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    compatibility: CompatibilitySpec = Field(default_factory=CompatibilitySpec)
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    lineage: LineageRef = Field(default_factory=LineageRef)
    evidence: EvidenceSpec = Field(default_factory=EvidenceSpec)
    status: StatusSpec = Field(default_factory=StatusSpec)


class CapabilityRoute(BaseModel):
    primary: str
    fallback: list[str] = Field(default_factory=list)
    timeout_ms: int = Field(default=1000, ge=1)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ProviderSpec(BaseModel):
    type: str
    endpoint: str | None = None
    endpoint_env: str | None = None
    api_key_env: str | None = None
    model: str | None = None
    entrypoint: str | None = None
    health_check: str | None = None


class RoutingPolicy(BaseModel):
    default: str = "capability_first"
    prefer_local: bool = True
    require_health_check: bool = False
    fallback_on: list[str] = Field(default_factory=list)


class ProvidersYaml(BaseModel):
    schema_version: Literal["rosclaw.providers.v1"] = SUPPORTED_PROVIDERS_SCHEMA
    required_capabilities: dict[str, CapabilityRoute] = Field(default_factory=dict)
    providers: dict[str, ProviderSpec] = Field(default_factory=dict)
    routing_policy: RoutingPolicy = Field(default_factory=RoutingPolicy)


class RobotCompatibility(BaseModel):
    robot: str
    eurdf_profile: str
    body_profile: dict[str, Any] = Field(default_factory=dict)
    required_limbs: list[str] = Field(default_factory=list)
    required_sensors: list[str] = Field(default_factory=list)
    optional_sensors: list[str] = Field(default_factory=list)
    required_frames: list[str] = Field(default_factory=list)
    action_interfaces: list[str] = Field(default_factory=list)
    physical_limits: dict[str, Any] = Field(default_factory=dict)
    environment_assumptions: dict[str, Any] = Field(default_factory=dict)


class IncompatibleRobot(BaseModel):
    robot: str
    reason: str


class EurdfCompatYaml(BaseModel):
    schema_version: Literal["rosclaw.eurdf_compat.v1"] = SUPPORTED_EURDF_COMPAT_SCHEMA
    compatible_robots: list[RobotCompatibility] = Field(default_factory=list)
    incompatible: list[IncompatibleRobot] = Field(default_factory=list)


class RuntimeMode(BaseModel):
    default: str = "sandbox_first"
    allowed: list[str] = Field(default_factory=lambda: ["dry_run", "replay", "sandbox", "real_robot_guarded"])


class RobotSafety(BaseModel):
    min_battery_percent: float | None = None
    max_joint_temp_c: float | None = None
    max_torso_pitch_deg: float | None = None
    max_torso_roll_deg: float | None = None
    require_estop_ready: bool = False


class ActionSafety(BaseModel):
    max_linear_velocity_mps: float | None = None
    max_angular_velocity_radps: float | None = None
    max_foot_swing_velocity_mps: float | None = None
    max_kick_strength: float | None = None


class EnvironmentSafety(BaseModel):
    require_clear_radius_m: float | None = None
    disallow_humans_within_m: float | None = None
    disallow_unknown_obstacles: bool = False


class SandboxSafety(BaseModel):
    required_checks: list[str] = Field(default_factory=list)
    block_on: list[str] = Field(default_factory=list)


class FailurePolicy(BaseModel):
    on_sandbox_block: str = "abort_and_explain"
    on_low_confidence: str = "verify_or_abort"
    on_repeated_failure: str = "write_memory_and_stop"
    max_runtime_retries: int = Field(default=2, ge=0, le=10)


class SafetyYaml(BaseModel):
    schema_version: Literal["rosclaw.safety.v1"] = SUPPORTED_SAFETY_SCHEMA
    runtime_mode: RuntimeMode = Field(default_factory=RuntimeMode)
    hard_constraints: dict[str, Any] = Field(default_factory=dict)
    robot: RobotSafety = Field(default_factory=RobotSafety)
    action: ActionSafety = Field(default_factory=ActionSafety)
    environment: EnvironmentSafety = Field(default_factory=EnvironmentSafety)
    sandbox: SandboxSafety = Field(default_factory=SandboxSafety)
    failure_policy: FailurePolicy = Field(default_factory=FailurePolicy)

    @model_validator(mode="after")
    def _ensure_constraints_present(self) -> SafetyYaml:
        if not self.hard_constraints and not any(
            [self.robot.model_dump(exclude_defaults=True), self.action.model_dump(exclude_defaults=True), self.environment.model_dump(exclude_defaults=True)]
        ):
            raise ValueError("SafetyYaml must define hard_constraints or concrete robot/action/environment constraints")
        return self


class PracticeSources(BaseModel):
    default_query: dict[str, Any] = Field(default_factory=dict)
    storage: dict[str, list[str]] = Field(default_factory=dict)


class MiningConfig(BaseModel):
    candidate_prefix: str = "candidate"
    min_episodes: int = 10
    min_success_episodes: int = 3
    include_failure_recovery: bool = True
    segmentation: dict[str, Any] = Field(default_factory=dict)
    clustering: dict[str, Any] = Field(default_factory=dict)


class ReplayConfig(BaseModel):
    required: bool = True
    sample_episodes: int = 5
    compare: list[str] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    enabled: bool = False
    output_dir: str = "policies/checkpoints"


class DojoYaml(BaseModel):
    schema_version: Literal["rosclaw.dojo.v1"] = SUPPORTED_DOJO_SCHEMA
    practice_sources: PracticeSources = Field(default_factory=PracticeSources)
    mining: MiningConfig = Field(default_factory=MiningConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


class DarwinTask(BaseModel):
    name: str
    episodes: int = Field(default=1, ge=1)


class MetricThreshold(BaseModel):
    required: bool = True
    promote_threshold: float | None = None
    max_allowed: float | None = None
    max_mean: float | None = None


class PromotionGates(BaseModel):
    candidate_to_validated: dict[str, Any] = Field(default_factory=dict)


class RegressionConfig(BaseModel):
    compare_against: list[str] = Field(default_factory=list)
    fail_if_success_drop_gt: float | None = None
    fail_if_safety_regression: bool = True


class DarwinEvalYaml(BaseModel):
    schema_version: Literal["rosclaw.darwin_eval.v1"] = SUPPORTED_DARWIN_EVAL_SCHEMA
    suite: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, MetricThreshold] = Field(default_factory=dict)
    promotion_gates: PromotionGates = Field(default_factory=PromotionGates)
    regression: RegressionConfig = Field(default_factory=RegressionConfig)


class LineageCandidate(BaseModel):
    id: str
    created_at: str = Field(default_factory=_now_iso)
    source: str = "practice_mining"
    status: str = "candidate"
    eval_report: str | None = None


class LineageVersion(BaseModel):
    version: str
    candidate_id: str | None = None
    git_commit: str | None = None
    package_hash: str | None = None
    promoted_at: str | None = None
    promoted_by: str | None = None


class LineageRollback(BaseModel):
    at: str = Field(default_factory=_now_iso)
    from_version: str
    to_version: str
    reason: str
    operator: str = "local"
    evidence: str | None = None


class LineageYaml(BaseModel):
    schema_version: Literal["rosclaw.lineage.v1"] = SUPPORTED_LINEAGE_SCHEMA
    skill: dict[str, Any] = Field(default_factory=dict)
    origin: dict[str, Any] = Field(default_factory=dict)
    parents: dict[str, Any] = Field(default_factory=dict)
    candidates: list[LineageCandidate] = Field(default_factory=list)
    versions: list[LineageVersion] = Field(default_factory=list)
    rollbacks: list[LineageRollback] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SkillPackage aggregate
# ---------------------------------------------------------------------------


class SkillPackage:
    """Loaded, typed representation of a local Skill Hub package."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.skill: SkillYaml | None = None
        self.providers: ProvidersYaml | None = None
        self.eurdf_compat: EurdfCompatYaml | None = None
        self.safety: SafetyYaml | None = None
        self.dojo: DojoYaml | None = None
        self.darwin_eval: DarwinEvalYaml | None = None
        self.lineage: LineageYaml | None = None

    @classmethod
    def load(cls, path: str | Path) -> SkillPackage:
        pkg = cls(path)
        pkg.skill = SkillYaml.model_validate(_load_yaml(pkg.root / "skill.yaml"))
        pkg.providers = ProvidersYaml.model_validate(_load_yaml(pkg.root / "providers.yaml"))
        pkg.eurdf_compat = EurdfCompatYaml.model_validate(_load_yaml(pkg.root / "e-urdf-compat.yaml"))
        pkg.safety = SafetyYaml.model_validate(_load_yaml(pkg.root / "safety.yaml"))
        pkg.dojo = DojoYaml.model_validate(_load_yaml(pkg.root / "dojo.yaml"))
        pkg.darwin_eval = DarwinEvalYaml.model_validate(_load_yaml(pkg.root / "darwin_eval.yaml"))
        pkg.lineage = LineageYaml.model_validate(_load_yaml(pkg.root / "lineage.yaml"))
        return pkg

    def try_load(self) -> SkillPackage:
        """Load whatever files exist; missing ones are left as None."""
        from pydantic import ValidationError

        for attr, path, model in [
            ("skill", "skill.yaml", SkillYaml),
            ("providers", "providers.yaml", ProvidersYaml),
            ("eurdf_compat", "e-urdf-compat.yaml", EurdfCompatYaml),
            ("safety", "safety.yaml", SafetyYaml),
            ("dojo", "dojo.yaml", DojoYaml),
            ("darwin_eval", "darwin_eval.yaml", DarwinEvalYaml),
            ("lineage", "lineage.yaml", LineageYaml),
        ]:
            try:
                setattr(self, attr, model.model_validate(_load_yaml(self.root / path)))
            except FileNotFoundError:
                setattr(self, attr, None)
            except ValidationError:
                setattr(self, attr, None)
        return self

    @property
    def skill_id(self) -> str:
        if self.skill is None:
            return self.root.name
        ns = self.skill.metadata.namespace
        name = self.skill.metadata.name
        return f"{ns}/{name}"

    @property
    def name(self) -> str:
        if self.skill is None:
            return self.root.name
        return self.skill.metadata.name

    @property
    def version(self) -> str:
        if self.skill is None:
            return "0.0.0"
        return self.skill.metadata.version

    @property
    def candidate_id(self) -> str | None:
        if self.skill is None:
            return None
        return self.skill.metadata.candidate_id

    def write_skill_yaml(self) -> None:
        if self.skill is None:
            raise RuntimeError("skill is not loaded")
        self._atomic_write(self.root / "skill.yaml", self._yaml_bytes(self.skill.model_dump(mode="json")))

    def write_lineage_yaml(self) -> None:
        if self.lineage is None:
            raise RuntimeError("lineage is not loaded")
        self._atomic_write(self.root / "lineage.yaml", self._yaml_bytes(self.lineage.model_dump(mode="json")))

    def write_lock_yaml(self, data: dict[str, Any]) -> None:
        self._atomic_write(self.root / ".rosclaw" / "lock.yaml", self._yaml_bytes(data))

    def write_manifest_json(self, data: dict[str, Any]) -> None:
        import json

        self._atomic_write(self.root / ".rosclaw" / "manifest.json", json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))

    def write_hashes_json(self, data: dict[str, Any]) -> None:
        import json

        self._atomic_write(self.root / ".rosclaw" / "hashes.json", json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))

    def write_upload_receipt(self, data: dict[str, Any]) -> None:
        import json

        self._atomic_write(self.root / ".rosclaw" / "upload_receipt.json", json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))

    @staticmethod
    def _yaml_bytes(data: dict[str, Any]) -> bytes:
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True).encode("utf-8")

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)


class SkillRef:
    """Parsed skill reference: name[@candidate|@version]."""

    def __init__(self, ref: str) -> None:
        self.raw = ref
        self.namespace: str | None = None
        rest = ref
        if "/" in ref:
            self.namespace, rest = ref.split("/", 1)
        if "@" in rest:
            self.name, self.ref = rest.split("@", 1)
        else:
            self.name = rest
            self.ref = None
        self.version: str | None = None
        self.candidate_id: str | None = None
        if self.ref:
            if self.ref.startswith("candidate_"):
                self.candidate_id = self.ref
            else:
                self.version = self.ref

    def __repr__(self) -> str:  # pragma: no cover
        return f"SkillRef(namespace={self.namespace!r}, name={self.name!r}, version={self.version!r}, candidate_id={self.candidate_id!r})"
