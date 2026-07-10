"""Orchestrate the P1.1 real-policy smoke gate.

``rosclaw lerobot smoke-policy`` runs a multi-stage acceptance workflow:

  0. runtime check
  1. policy materialization
  2. inspect
  3. load-test
  4. infer
  5. smoke report

The module stays free of torch/lerobot imports; it delegates heavy work to
``LeRobotPolicyProvider`` via the subprocess worker.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.core.async_utils import run_sync
from rosclaw.integrations.lerobot.config import get_configured_lerobot_runtime
from rosclaw.integrations.lerobot.observation_adapter import adapt_observation_for_worker
from rosclaw.integrations.lerobot.policy_cache import (
    MaterializationResult,
    PolicyMaterializationError,
    materialize_policy_path,
)
from rosclaw.integrations.lerobot.provider import LeRobotPolicyProvider
from rosclaw.integrations.lerobot.runtime import inspect_lerobot_runtime
from rosclaw.integrations.lerobot.schemas import LeRobotWorkerErrorCode
from rosclaw.integrations.lerobot.smoke_report import (
    SmokeReport,
    build_validation_block,
    compute_warnings,
    summarize_action_proposal,
    write_smoke_report,
)
from rosclaw.provider.core.manifest import (
    EmbodimentSpec,
    ModelSpec,
    ObservabilitySpec,
    ProviderManifest,
    RuntimeSpec,
    SafetySpec,
)
from rosclaw.provider.core.request import ProviderRequest

DEFAULT_SMOKE_POLICY = "lerobot/act_aloha_sim_transfer_cube_human"
DEFAULT_OBSERVATION_FILE = Path(__file__).with_name("resources") / "sample_observation_aloha_act.json"


@dataclass
class SmokePolicyOptions:
    """Options for ``run_smoke_policy``."""

    policy_path: str = DEFAULT_SMOKE_POLICY
    revision: str = "main"
    device: str = "cpu"
    dtype: str = "auto"
    allow_network: bool = False
    timeout_sec: int = 300
    output: Path | None = None
    keep_worker_files: bool = False
    force_download: bool = False
    skip_infer: bool = False
    observation_file: Path | None = None
    json_output: bool = False


@dataclass
class SmokeStageResult:
    """Result of a single smoke stage."""

    status: str = "ok"
    error: dict[str, Any] | None = None
    data: dict[str, Any] = field(default_factory=dict)
    latency_sec: float = 0.0


async def run_smoke_policy(options: SmokePolicyOptions) -> SmokeReport:
    """Run the full P1.1 smoke workflow and return a v1.1 report."""
    report = SmokeReport(status="error")
    stages: dict[str, Any] = {}
    timings: dict[str, float] = {}
    features: dict[str, Any] = {}
    runtime_info: dict[str, Any] = {}
    policy_info: dict[str, Any] = {
        "repo_id": _repo_id_or_none(options.policy_path),
        "local_path": None,
        "revision": options.revision,
        "policy_type": None,
        "artifact_files": {},
    }
    action_proposal: dict[str, Any] | None = None
    sample_obs: dict[str, Any] = {}

    # Stage 0: runtime check
    stage0 = _stage_runtime_check(options)
    stages["runtime_check"] = _stage_dict(stage0)
    runtime_info = stage0.data.get("runtime", {})
    runtime_info["device"] = options.device
    if stage0.status != "ok":
        _populate_error_report(report, stages, runtime_info, {}, {}, {}, error=stage0.error)
        return _maybe_write_report(report, options)

    # Stage 1: materialize policy
    stage1 = _stage_materialize(options)
    stages["materialize"] = _stage_dict(stage1)
    if stage1.status != "ok":
        _populate_error_report(report, stages, runtime_info, policy_info, features, sample_obs, error=stage1.error)
        return _maybe_write_report(report, options)

    materialized = stage1.data.get("materialization")
    local_policy_path = materialized.local_path if isinstance(materialized, MaterializationResult) else Path(materialized)
    policy_info["local_path"] = str(local_policy_path)
    policy_info["artifact_files"] = _artifact_files(local_policy_path)

    # Stage 2: inspect
    stage2 = await _stage_inspect(local_policy_path, options)
    stages["inspect"] = _stage_dict(stage2)
    timings["inspect_time_sec"] = stage2.latency_sec
    if stage2.status == "ok":
        metadata = stage2.data.get("policy_metadata", {})
        policy_info["policy_type"] = metadata.get("policy_type")
        features["input_features"] = _feature_shapes(metadata.get("input_features", {}))
        features["output_features"] = _feature_shapes(metadata.get("output_features", {}))

    if stage2.status != "ok":
        _populate_error_report(report, stages, runtime_info, policy_info, features, sample_obs, error=stage2.error)
        return _maybe_write_report(report, options)

    # Stage 3: load-test
    stage3 = await _stage_load_test(local_policy_path, options)
    stages["load_test"] = _stage_dict(stage3)
    timings["load_time_sec"] = stage3.latency_sec
    if stage3.status != "ok":
        _populate_error_report(report, stages, runtime_info, policy_info, features, sample_obs, error=stage3.error)
        return _maybe_write_report(report, options)

    # Stage 4: infer
    if not options.skip_infer:
        stage4 = await _stage_infer(local_policy_path, options)
        stages["infer"] = _stage_dict(stage4)
        timings["infer_time_sec"] = stage4.latency_sec
        if stage4.status == "ok":
            raw_proposal = stage4.data.get("action_proposal")
            action_proposal = summarize_action_proposal(raw_proposal)
            sample_obs = stage4.data.get("sample_observation", {})
        if stage4.status != "ok":
            _populate_error_report(report, stages, runtime_info, policy_info, features, sample_obs, error=stage4.error)
            return _maybe_write_report(report, options)
    else:
        stages["infer"] = {"status": "skipped"}

    total_time = sum(v for v in timings.values() if isinstance(v, (int, float)))
    timings["total_time_sec"] = round(total_time, 3)

    report.status = "ok"
    report.policy = policy_info
    report.runtime = runtime_info
    report.stages = stages
    report.features = features
    report.sample_observation = sample_obs
    report.action_proposal = action_proposal
    report.timing = timings
    report.warnings = compute_warnings(timings, action_proposal)
    report.validation = build_validation_block("ok")
    return _maybe_write_report(report, options)


def _stage_runtime_check(options: SmokePolicyOptions) -> SmokeStageResult:
    """Check that a usable LeRobot runtime exists."""
    configured = get_configured_lerobot_runtime()
    python_exe: str | None = None
    mode = "external"
    if configured and configured.get("subprocess_available"):
        python_exe = configured.get("python_executable")
        mode = configured.get("mode", "external")

    if not python_exe:
        import sys

        info = inspect_lerobot_runtime(sys.executable, mode="current-env")
        if info.state in ("ready", "degraded") and info.lerobot_version is not None:
            python_exe = str(info.python_executable)
            mode = "current-env"
        else:
            return SmokeStageResult(
                status="error",
                error={
                    "code": LeRobotWorkerErrorCode.RUNTIME_NOT_CONFIGURED.value,
                    "message": "No LeRobot runtime configured. Run `rosclaw setup lerobot --profile core` first.",
                },
            )

    runtime = inspect_lerobot_runtime(python_exe, mode=mode)
    if runtime.state not in ("ready", "degraded") or runtime.lerobot_version is None:
        return SmokeStageResult(
            status="error",
            error={
                "code": LeRobotWorkerErrorCode.RUNTIME_NOT_CONFIGURED.value,
                "message": f"LeRobot runtime is not ready: {runtime.error}",
            },
        )

    return SmokeStageResult(
        status="ok",
        data={
            "runtime": {
                "mode": runtime.mode,
                "python_executable": str(runtime.python_executable),
                "python_version": runtime.python_version,
                "lerobot_version": runtime.lerobot_version,
                "torch_version": runtime.torch_version,
                "cuda_available": runtime.cuda_available,
            }
        },
    )


def _stage_materialize(options: SmokePolicyOptions) -> SmokeStageResult:
    """Resolve the policy path to a local directory."""
    t0 = time.perf_counter()
    try:
        result = materialize_policy_path(
            options.policy_path,
            revision=options.revision,
            allow_network=options.allow_network,
            force_download=options.force_download,
        )
    except PolicyMaterializationError as exc:
        return SmokeStageResult(
            status="error",
            latency_sec=round(time.perf_counter() - t0, 3),
            error={"code": exc.code, "message": exc.message, "details": exc.details},
        )
    latency = round(time.perf_counter() - t0, 3)
    data: dict[str, Any] = {"materialization": result}
    if isinstance(result, MaterializationResult):
        data["network_used"] = result.network_used
        data["cache_hit"] = result.cache_hit
    return SmokeStageResult(status="ok", latency_sec=latency, data=data)


async def _stage_inspect(local_path: Path, options: SmokePolicyOptions) -> SmokeStageResult:
    """Run provider inspect."""
    provider, request = _build_provider_request(
        "lerobot.policy.inspect",
        local_path,
        options,
    )
    t0 = time.perf_counter()
    response = await provider.infer(request)
    latency = round(time.perf_counter() - t0, 3)
    if response.status != "ok":
        return SmokeStageResult(
            status="error",
            latency_sec=latency,
            error={
                "code": response.result.get("error_code", "policy_load_failed"),
                "message": response.result.get("message", "Inspect failed"),
            },
        )
    return SmokeStageResult(
        status="ok",
        latency_sec=latency,
        data={"policy_metadata": response.result.get("policy_metadata", {})},
    )


async def _stage_load_test(local_path: Path, options: SmokePolicyOptions) -> SmokeStageResult:
    """Run provider load-test."""
    provider, request = _build_provider_request(
        "lerobot.policy.load_test",
        local_path,
        options,
    )
    t0 = time.perf_counter()
    response = await provider.infer(request)
    latency = round(time.perf_counter() - t0, 3)
    if response.status != "ok":
        return SmokeStageResult(
            status="error",
            latency_sec=latency,
            error={
                "code": response.result.get("error_code", "policy_load_failed"),
                "message": response.result.get("message", "Load-test failed"),
            },
        )
    return SmokeStageResult(status="ok", latency_sec=latency, data={})


async def _stage_infer(local_path: Path, options: SmokePolicyOptions) -> SmokeStageResult:
    """Run provider infer with a sample observation."""
    observation_file = options.observation_file or DEFAULT_OBSERVATION_FILE
    try:
        observation = json.loads(observation_file.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return SmokeStageResult(
            status="error",
            error={
                "code": LeRobotWorkerErrorCode.OBSERVATION_SCHEMA_MISMATCH.value,
                "message": f"Failed to load observation file {observation_file}: {exc}",
            },
        )

    # Ensure image paths are absolute so the worker can find them.
    observation = _resolve_observation_paths(observation, observation_file.parent)

    try:
        observation = adapt_observation_for_worker(observation)
    except (ValueError, FileNotFoundError) as exc:
        return SmokeStageResult(
            status="error",
            error={
                "code": LeRobotWorkerErrorCode.OBSERVATION_SCHEMA_MISMATCH.value,
                "message": str(exc),
            },
        )

    provider, request = _build_provider_request(
        "lerobot.policy.infer",
        local_path,
        options,
        observation=observation,
    )
    t0 = time.perf_counter()
    response = await provider.infer(request)
    latency = round(time.perf_counter() - t0, 3)
    if response.status != "ok":
        return SmokeStageResult(
            status="error",
            latency_sec=latency,
            error={
                "code": response.result.get("error_code", "policy_infer_failed"),
                "message": response.result.get("message", "Inference failed"),
            },
        )

    proposal = response.result.get("action_proposal")
    if proposal is None:
        return SmokeStageResult(
            status="error",
            latency_sec=latency,
            error={
                "code": LeRobotWorkerErrorCode.POLICY_INFER_FAILED.value,
                "message": "Inference succeeded but no action_proposal was returned.",
            },
        )

    proposal["not_executed"] = True
    proposal["requires_sandbox"] = True
    proposal["executable"] = False
    proposal.setdefault("body_mapping_required", True)
    proposal.setdefault("body_compatible", False)

    return SmokeStageResult(
        status="ok",
        latency_sec=latency,
        data={
            "action_proposal": proposal,
            "sample_observation": _summarize_observation(observation),
        },
    )


def _build_provider_request(
    capability: str,
    local_path: Path,
    options: SmokePolicyOptions,
    observation: dict[str, Any] | None = None,
) -> tuple[LeRobotPolicyProvider, ProviderRequest]:
    """Build a provider and request for the given capability."""
    manifest = _minimal_manifest()
    provider = LeRobotPolicyProvider(manifest)
    inputs: dict[str, Any] = {
        "policy.path": str(local_path),
        "revision": options.revision,
        "device": options.device,
        "allow_network": options.allow_network,
        "timeout_sec": options.timeout_sec,
    }
    if observation is not None:
        inputs["observation"] = observation

    request = ProviderRequest(
        request_id=f"lerobot_smoke_{capability.split('.')[-1]}_{int(time.time()*1000)}",
        capability=capability,
        inputs=inputs,
        context={},
    )
    return provider, request


def _minimal_manifest() -> ProviderManifest:
    """Return a minimal manifest that satisfies the provider interface."""
    return ProviderManifest(
        name="lerobot_policy_smoke",
        version="0.2.0",
        type="skill",
        description="Minimal manifest for smoke-policy",
        capabilities=[
            "lerobot.policy.inspect",
            "lerobot.policy.load_test",
            "lerobot.policy.infer",
        ],
        modalities={"input": ["image", "state"], "output": ["action"]},
        runtime=RuntimeSpec(backend="python", protocol="subprocess_worker", device="cpu"),
        model=ModelSpec(name="lerobot_smoke", source="local"),
        embodiment=EmbodimentSpec(
            supported_robots=["aloha_sim"],
            camera_setup=["top"],
            action_space=[f"joint_{i}" for i in range(14)],
            control_frequency_hz=30,
        ),
        safety=SafetySpec(
            executable=False,
            requires_guard=True,
            requires_collision_check=True,
            requires_workspace_check=True,
            max_action_norm=0.5,
        ),
        observability=ObservabilitySpec(),
        extra={"action_shape": [14]},
    )


def _resolve_observation_paths(observation: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve relative image paths in an observation dict."""
    obs = observation.get("observation", observation)
    if not isinstance(obs, dict):
        return observation

    images = obs.get("images")
    if isinstance(images, dict):
        resolved = {}
        for name, path in images.items():
            p = Path(path)
            if not p.is_absolute():
                p = base_dir / p
            resolved[name] = str(p.resolve())
        obs["images"] = resolved

    return observation


def _feature_shapes(features: dict[str, Any]) -> dict[str, Any]:
    """Extract shapes from a LeRobot feature dict."""
    out: dict[str, Any] = {}
    for key, value in features.items():
        if isinstance(value, dict):
            out[key] = value.get("shape")
        else:
            out[key] = value
    return out


def _stage_dict(stage: SmokeStageResult) -> dict[str, Any]:
    """Convert a stage result into the v1.1 stage dict."""
    d: dict[str, Any] = {"status": stage.status}
    if stage.latency_sec:
        d["time_sec"] = stage.latency_sec
    if stage.error:
        d["error"] = stage.error
    d.update({k: v for k, v in stage.data.items() if k not in {"materialization"}})
    return d


def _populate_error_report(
    report: SmokeReport,
    stages: dict[str, Any],
    runtime_info: dict[str, Any],
    policy_info: dict[str, Any],
    features: dict[str, Any],
    sample_obs: dict[str, Any],
    error: dict[str, Any] | None,
) -> None:
    """Fill a report object for an early-exit failure."""
    report.stages = stages
    report.runtime = runtime_info
    report.policy = policy_info
    report.features = features
    report.sample_observation = sample_obs
    report.error = error
    report.warnings = compute_warnings(report.timing, None)
    report.validation = build_validation_block("error")


def _artifact_files(local_path: Path) -> dict[str, bool]:
    """Check for expected policy artifact files."""
    return {
        "config_json": (local_path / "config.json").exists(),
        "model_safetensors": (local_path / "model.safetensors").exists(),
        "train_config_json": (local_path / "train_config.json").exists(),
    }


def _summarize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Summarize the worker observation used for inference."""
    task = observation.get("task")
    obs = observation.get("observation", observation)
    if not isinstance(obs, dict):
        return {"task": task, "state_shape": None, "image_keys": [], "image_shapes": {}}

    if task is None:
        task = obs.get("task")

    state = obs.get("state")
    if state is None:
        state = obs.get("observation.state")
    state_shape = list(state) if isinstance(state, (list, tuple)) else None
    if state_shape and state_shape and isinstance(state_shape[0], (int, float)):
        state_shape = [len(state_shape)]

    image_shapes: dict[str, Any] = {}
    images = dict(obs.get("images", {})) if isinstance(obs.get("images"), dict) else {}
    for key, value in obs.items():
        if key.startswith("observation.images."):
            images[key.split(".", 2)[2]] = value
    if isinstance(images, dict):
        for name, path in images.items():
            try:
                from PIL import Image

                with Image.open(path) as img:
                    image_shapes[name] = list(img.size) + [len(img.getbands())]
            except Exception:  # noqa: BLE001
                image_shapes[name] = None

    return {
        "task": task,
        "state_shape": state_shape,
        "image_keys": list(images.keys()),
        "image_shapes": image_shapes,
    }


def _repo_id_or_none(policy_path: str) -> str | None:
    path = Path(policy_path).expanduser()
    if path.exists() or path.is_absolute() or policy_path.startswith("."):
        return None
    return policy_path if "/" in policy_path else None


def _maybe_write_report(report: SmokeReport, options: SmokePolicyOptions) -> SmokeReport:
    """Persist the report and optionally copy it to a user path."""
    path = write_smoke_report(report)
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    report.policy["report_path"] = str(path)
    return report


def run_smoke_policy_sync(options: SmokePolicyOptions) -> SmokeReport:
    """Synchronous entry point used by the CLI."""
    return run_sync(run_smoke_policy(options))


__all__ = [
    "DEFAULT_SMOKE_POLICY",
    "SmokePolicyOptions",
    "run_smoke_policy",
    "run_smoke_policy_sync",
]
