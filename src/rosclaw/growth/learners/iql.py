"""Conservative IQL candidate training for the G1 recovery transition schema."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.recovery_dataset import COST_NAMES, REWARD_NAMES, STATE_FEATURES


@dataclass(frozen=True)
class IQLTrainingConfig:
    steps: int = 2000
    batch_size: int = 256
    hidden_size: int = 256
    discount: float = 0.99
    expectile: float = 0.70
    advantage_temperature: float = 3.0
    maximum_advantage_weight: float = 100.0
    learning_rate: float = 3e-4
    seed: int = 20260805
    device: str = "cpu"
    schema_version: str = "rosclaw.growth.iql_training_config.v1"

    def __post_init__(self) -> None:
        if not 1 <= self.steps <= 1_000_000:
            raise ValueError("IQL steps must be in [1, 1000000]")
        if not 1 <= self.batch_size <= 65536:
            raise ValueError("IQL batch size must be in [1, 65536]")
        if not 16 <= self.hidden_size <= 4096:
            raise ValueError("IQL hidden size must be in [16, 4096]")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("IQL discount must be in (0, 1]")
        if not 0.5 < self.expectile < 1.0:
            raise ValueError("IQL expectile must be in (0.5, 1)")
        if self.advantage_temperature <= 0.0 or self.maximum_advantage_weight < 1.0:
            raise ValueError("IQL advantage weighting parameters are invalid")
        if self.learning_rate <= 0.0 or not math.isfinite(self.learning_rate):
            raise ValueError("IQL learning rate must be finite and positive")
        if not self.device or not (
            self.device == "cpu" or self.device == "cuda" or self.device.startswith("cuda:")
        ):
            raise ValueError("IQL device must be cpu, cuda, or cuda:<index>")


@dataclass(frozen=True)
class IQLTrainingReceipt:
    candidate_path: str
    candidate_hash: str
    weights_path: str
    weights_hash: str
    dataset_manifest_hash: str
    training_steps: int
    train_transition_count: int
    validation_transition_count: int
    reserved_transition_count: int
    validation_normalized_mse_before: float
    validation_normalized_mse_after: float
    device: str
    status: str
    schema_version: str = "rosclaw.growth.iql_training_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "promotion_authorized": False,
            "activation_authorized": False,
            "hardware_authorized": False,
        }


@dataclass(frozen=True)
class IQLResidualGuardConfig:
    """Bound an offline actor to a small correction around a proven controller.

    The standardized envelope is deliberately only a support heuristic, not a
    calibrated OOD probability. Rejected states fall back to the structured
    controller and every accepted correction remains amplitude bounded.
    """

    residual_fraction: float = 0.05
    maximum_residual_nm: float = 2.0
    maximum_standardized_rms: float = 4.0
    maximum_standardized_abs: float = 20.0
    joint_group: str = "legs"
    schema_version: str = "rosclaw.growth.iql_residual_guard_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.residual_fraction,
            self.maximum_residual_nm,
            self.maximum_standardized_rms,
            self.maximum_standardized_abs,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("IQL residual guard values must be finite")
        if not 0.0 < self.residual_fraction <= 0.50:
            raise ValueError("IQL residual fraction must be in (0, 0.50]")
        if not 0.10 <= self.maximum_residual_nm <= 20.0:
            raise ValueError("IQL maximum residual must be in [0.10, 20] Nm")
        if not 0.50 <= self.maximum_standardized_rms <= 20.0:
            raise ValueError("IQL standardized RMS bound must be in [0.50, 20]")
        if not 1.0 <= self.maximum_standardized_abs <= 100.0:
            raise ValueError("IQL standardized absolute bound must be in [1, 100]")
        if self.joint_group not in {"legs", "lower_body", "whole_body"}:
            raise ValueError("IQL residual joint group is invalid")


@dataclass(frozen=True)
class IQLResidualDecision:
    """One auditable residual action decision."""

    residual_torque: np.ndarray
    accepted: bool
    confidence: float
    standardized_rms: float
    standardized_abs: float
    peak_residual_nm: float
    reason: str


@dataclass(frozen=True)
class NumpyIQLActor:
    """No-pickle inference form of an unevaluated IQL recovery actor."""

    layer_weights: tuple[np.ndarray, ...]
    layer_biases: tuple[np.ndarray, ...]
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    candidate_hash: str

    @classmethod
    def load(cls, candidate_path: Path) -> NumpyIQLActor:
        metadata_path = candidate_path.expanduser().resolve()
        if not metadata_path.is_file() or metadata_path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("IQL candidate metadata is missing or oversized")
        candidate = json.loads(metadata_path.read_text(encoding="utf-8"))
        claimed = candidate.get("candidate_hash")
        unsigned = dict(candidate)
        unsigned.pop("candidate_hash", None)
        if claimed != canonical_hash(unsigned):
            raise ValueError("IQL candidate hash mismatch")
        if candidate.get("schema_version") != "rosclaw.growth.iql_candidate.v1":
            raise ValueError("unsupported IQL candidate schema")
        if candidate.get("status") != "CANDIDATE_UNEVALUATED":
            raise ValueError("only an unevaluated IQL candidate can enter SIM-only evaluation")
        for name, expected in (
            ("activation_ceiling", "SIM_ONLY"),
            ("promotion_authorized", False),
            ("hardware_command_sent", False),
        ):
            if candidate.get(name) != expected:
                raise ValueError(f"IQL candidate requires {name}={expected!r}")
        artifact = candidate.get("artifact", {})
        if artifact.get("format") != "numpy_npz_no_pickle":
            raise ValueError("IQL actor requires the safe NumPy artifact format")
        if (
            artifact.get("actor_output") != "executed_torque_nm"
            or artifact.get("learned_output_fraction") != 1.0
        ):
            raise ValueError("IQL actor output semantics are incompatible")
        weights_path = Path(str(artifact.get("weights_path", ""))).expanduser().resolve()
        if weights_path.parent != metadata_path.parent:
            raise ValueError("IQL actor weights must be adjacent to candidate metadata")
        if not weights_path.is_file() or weights_path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("IQL actor weights are missing or oversized")
        if _file_hash(weights_path) != artifact.get("weights_hash"):
            raise ValueError("IQL actor weight hash mismatch")
        with np.load(weights_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files}
        if _array_content_hash(arrays) != artifact.get("weights_content_hash"):
            raise ValueError("IQL actor weight content hash mismatch")
        required = {
            "net__0__weight",
            "net__0__bias",
            "net__2__weight",
            "net__2__bias",
            "net__4__weight",
            "net__4__bias",
            "state_mean",
            "state_std",
            "action_mean",
            "action_std",
        }
        missing = sorted(required.difference(arrays))
        if missing:
            raise ValueError(f"IQL actor artifact is missing arrays: {missing}")
        if not all(np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("IQL actor artifact contains non-finite arrays")
        weights = tuple(arrays[f"net__{index}__weight"] for index in (0, 2, 4))
        biases = tuple(arrays[f"net__{index}__bias"] for index in (0, 2, 4))
        state_size = len(STATE_FEATURES)
        if weights[0].shape[1] != state_size or weights[-1].shape[0] != 29:
            raise ValueError("IQL actor input/output contract mismatch")
        if any(
            weight.shape[0] != bias.shape[0] for weight, bias in zip(weights, biases, strict=True)
        ):
            raise ValueError("IQL actor layer bias contract mismatch")
        if weights[1].shape[1] != weights[0].shape[0] or weights[2].shape[1] != weights[1].shape[0]:
            raise ValueError("IQL actor hidden layer contract mismatch")
        if arrays["state_mean"].shape != (state_size,) or arrays["state_std"].shape != (
            state_size,
        ):
            raise ValueError("IQL actor state normalization contract mismatch")
        if arrays["action_mean"].shape != (29,) or arrays["action_std"].shape != (29,):
            raise ValueError("IQL actor action normalization contract mismatch")
        if np.any(arrays["state_std"] <= 0.0) or np.any(arrays["action_std"] <= 0.0):
            raise ValueError("IQL actor normalization scales must be positive")
        return cls(
            layer_weights=weights,
            layer_biases=biases,
            state_mean=arrays["state_mean"],
            state_std=arrays["state_std"],
            action_mean=arrays["action_mean"],
            action_std=arrays["action_std"],
            candidate_hash=str(claimed),
        )

    def action(self, state: np.ndarray) -> np.ndarray:
        value = np.asarray(state, dtype=np.float32)
        if value.shape != self.state_mean.shape or not np.all(np.isfinite(value)):
            raise ValueError("IQL actor state must be one finite state vector")
        value = (value - self.state_mean) / self.state_std
        for index, (weight, bias) in enumerate(
            zip(self.layer_weights, self.layer_biases, strict=True)
        ):
            value = weight @ value + bias
            if index < len(self.layer_weights) - 1:
                value = value / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))
        action = value * self.action_std + self.action_mean
        if action.shape != (29,) or not np.all(np.isfinite(action)):
            raise RuntimeError("IQL actor produced an invalid action")
        return action.astype(np.float64)

    def standardized_state(self, state: np.ndarray) -> np.ndarray:
        """Return the actor's frozen standardized state with strict validation."""

        value = np.asarray(state, dtype=np.float32)
        if value.shape != self.state_mean.shape or not np.all(np.isfinite(value)):
            raise ValueError("IQL actor state must be one finite state vector")
        standardized = (value - self.state_mean) / self.state_std
        if not np.all(np.isfinite(standardized)):
            raise RuntimeError("IQL actor standardized state is non-finite")
        return standardized.astype(np.float64)


@dataclass(frozen=True)
class SupportBoundIQLResidualActor:
    """Use an IQL actor only as an envelope-gated residual torque proposer."""

    actor: NumpyIQLActor
    config: IQLResidualGuardConfig

    @classmethod
    def load(
        cls,
        candidate_path: Path,
        config: IQLResidualGuardConfig | None = None,
    ) -> SupportBoundIQLResidualActor:
        return cls(
            actor=NumpyIQLActor.load(candidate_path),
            config=config or IQLResidualGuardConfig(),
        )

    @property
    def candidate_hash(self) -> str:
        return self.actor.candidate_hash

    def action(self, state: np.ndarray, baseline_torque: np.ndarray) -> IQLResidualDecision:
        baseline = np.asarray(baseline_torque, dtype=np.float64)
        if baseline.shape != (29,) or not np.all(np.isfinite(baseline)):
            raise ValueError("IQL residual baseline must be one finite 29-joint torque")
        standardized = self.actor.standardized_state(state)
        rms = float(np.sqrt(np.mean(np.square(np.clip(standardized, -1e3, 1e3)))))
        maximum = float(np.max(np.abs(standardized)))
        supported = bool(
            rms <= self.config.maximum_standardized_rms
            and maximum <= self.config.maximum_standardized_abs
        )
        if not supported:
            return IQLResidualDecision(
                residual_torque=np.zeros(29, dtype=np.float64),
                accepted=False,
                confidence=0.0,
                standardized_rms=rms,
                standardized_abs=maximum,
                peak_residual_nm=0.0,
                reason="outside_standardized_support_envelope",
            )
        learned = self.actor.action(state)
        residual = np.clip(
            learned - baseline,
            -self.config.maximum_residual_nm,
            self.config.maximum_residual_nm,
        )
        mask = np.zeros(29, dtype=np.float64)
        stop = {"legs": 12, "lower_body": 15, "whole_body": 29}[self.config.joint_group]
        mask[:stop] = 1.0
        # Confidence decays smoothly inside the admitted envelope. It never
        # enlarges residual_fraction and becomes zero at the RMS boundary.
        confidence = max(0.0, 1.0 - rms / self.config.maximum_standardized_rms)
        residual = residual * mask * self.config.residual_fraction * confidence
        peak = float(np.max(np.abs(residual)))
        return IQLResidualDecision(
            residual_torque=residual.astype(np.float64),
            accepted=True,
            confidence=confidence,
            standardized_rms=rms,
            standardized_abs=maximum,
            peak_residual_nm=peak,
            reason="accepted_bounded_residual",
        )


def train_recovery_iql(
    *,
    dataset_manifest_path: Path,
    output_dir: Path,
    source_checkout: Path,
    config: IQLTrainingConfig | None = None,
) -> IQLTrainingReceipt:
    """Train an unevaluated candidate; this function can never activate it."""

    config = config or IQLTrainingConfig()
    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("IQL artifacts must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    manifest_path = dataset_manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_dataset_manifest(manifest)
    array_path = Path(manifest["arrays"]["path"]).expanduser().resolve()
    if _file_hash(array_path) != manifest["arrays"]["file_hash"]:
        raise ValueError("IQL dataset file hash mismatch")
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if _array_content_hash(arrays) != manifest["arrays"]["content_hash"]:
        raise ValueError("IQL dataset content hash mismatch")
    _verify_arrays(arrays)
    episode_ids = sorted(int(item) for item in np.unique(arrays["episode_index"]))
    if len(episode_ids) < 3:
        raise ValueError(
            "IQL requires at least three episodes for train/validation/reserved splits"
        )
    train_episode_ids = tuple(episode_ids[:-2])
    validation_episode_id = episode_ids[-2]
    reserved_episode_id = episode_ids[-1]
    train_mask = np.isin(arrays["episode_index"], train_episode_ids)
    validation_mask = arrays["episode_index"] == validation_episode_id
    reserved_mask = arrays["episode_index"] == reserved_episode_id
    if not np.any(train_mask) or not np.any(validation_mask) or not np.any(reserved_mask):
        raise ValueError("IQL split contains an empty partition")

    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("IQL training requires the optional torch dependency") from exc
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("IQL requested CUDA but CUDA is unavailable")
    device = torch.device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)

    state = arrays["state"].astype(np.float32)
    next_state = arrays["next_state"].astype(np.float32)
    action = arrays["executed_action"].astype(np.float32)
    reward = _scalar_return(arrays["reward_vector"], arrays["cost_vector"])
    terminal = arrays["terminal"].astype(np.float32)
    state_mean = state[train_mask].mean(axis=0)
    state_std = np.maximum(state[train_mask].std(axis=0), 1e-4)
    action_mean = action[train_mask].mean(axis=0)
    action_std = np.maximum(action[train_mask].std(axis=0), 1e-3)
    normalized_state = (state - state_mean) / state_std
    normalized_next_state = (next_state - state_mean) / state_std
    normalized_action = (action - action_mean) / action_std

    class MLP(nn.Module):
        def __init__(self, input_size: int, output_size: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, output_size),
            )

        def forward(self, value: Any) -> Any:
            return self.net(value)

    state_size = state.shape[1]
    action_size = action.shape[1]
    actor = MLP(state_size, action_size).to(device)
    q1 = MLP(state_size + action_size, 1).to(device)
    q2 = MLP(state_size + action_size, 1).to(device)
    value = MLP(state_size, 1).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    q_optimizer = torch.optim.Adam((*q1.parameters(), *q2.parameters()), lr=config.learning_rate)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=config.learning_rate)

    tensor_state = torch.as_tensor(normalized_state, device=device)
    tensor_next = torch.as_tensor(normalized_next_state, device=device)
    tensor_action = torch.as_tensor(normalized_action, device=device)
    tensor_reward = torch.as_tensor(reward, device=device).unsqueeze(1)
    tensor_terminal = torch.as_tensor(terminal, device=device).unsqueeze(1)
    train_indices = torch.as_tensor(np.flatnonzero(train_mask), device=device)
    validation_indices = torch.as_tensor(np.flatnonzero(validation_mask), device=device)
    before = _actor_mse(actor, tensor_state, tensor_action, validation_indices)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    curve: list[dict[str, float | int]] = []
    for step in range(1, config.steps + 1):
        sampled = train_indices[
            torch.randint(
                len(train_indices),
                (min(config.batch_size, len(train_indices)),),
                generator=generator,
                device=device,
            )
        ]
        batch_state = tensor_state[sampled]
        batch_next = tensor_next[sampled]
        batch_action = tensor_action[sampled]
        with torch.no_grad():
            target = tensor_reward[sampled] + config.discount * (
                1.0 - tensor_terminal[sampled]
            ) * value(batch_next)
        q_input = torch.cat((batch_state, batch_action), dim=1)
        q1_value = q1(q_input)
        q2_value = q2(q_input)
        q_loss = torch.mean((q1_value - target) ** 2 + (q2_value - target) ** 2)
        q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        q_optimizer.step()

        with torch.no_grad():
            minimum_q = torch.minimum(q1(q_input), q2(q_input))
        value_prediction = value(batch_state)
        residual = minimum_q - value_prediction
        expectile_weight = torch.where(
            residual >= 0.0,
            config.expectile,
            1.0 - config.expectile,
        )
        value_loss = torch.mean(expectile_weight * residual**2)
        value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        value_optimizer.step()

        with torch.no_grad():
            advantage = minimum_q - value(batch_state)
            weights = torch.exp(config.advantage_temperature * advantage).clamp(
                max=config.maximum_advantage_weight
            )
        prediction = actor(batch_state)
        actor_loss = torch.mean(
            weights * torch.mean((prediction - batch_action) ** 2, dim=1, keepdim=True)
        )
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optimizer.step()
        if step == 1 or step == config.steps or step % max(1, config.steps // 20) == 0:
            curve.append(
                {
                    "step": step,
                    "q_loss": float(q_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "validation_normalized_mse": _actor_mse(
                        actor, tensor_state, tensor_action, validation_indices
                    ),
                }
            )
    after = _actor_mse(actor, tensor_state, tensor_action, validation_indices)
    weights = _safe_actor_arrays(actor, state_mean, state_std, action_mean, action_std)
    weights_path = root / "actor_weights.npz"
    np.savez_compressed(weights_path, **weights)
    curve_path = root / "learning_curve.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    status = "CANDIDATE_UNEVALUATED" if after < before else "REJECTED_NO_VALIDATION_GAIN"
    candidate = {
        "schema_version": "rosclaw.growth.iql_candidate.v1",
        "learner_id": "iql",
        "task_id": "g1_post_impact_recovery",
        "dataset_manifest_hash": manifest["manifest_hash"],
        "environment_hash": manifest["environment_hash"],
        "config": asdict(config),
        "partition": {
            "train_episode_ids": list(train_episode_ids),
            "validation_episode_id": validation_episode_id,
            "reserved_episode_id_commitment": canonical_hash(
                {"episode_id": reserved_episode_id, "dataset": manifest["manifest_hash"]}
            ),
            "reserved_metrics_accessed": False,
        },
        "artifact": {
            "weights_path": str(weights_path),
            "weights_hash": _file_hash(weights_path),
            "weights_content_hash": _array_content_hash(weights),
            "format": "numpy_npz_no_pickle",
            "actor_output": "executed_torque_nm",
            "learned_output_fraction": 1.0,
        },
        "metrics": {
            "validation_normalized_mse_before": before,
            "validation_normalized_mse_after": after,
            "validation_relative_mse_reduction": (before - after) / max(before, 1e-12),
        },
        "status": status,
        "darwin_evaluated": False,
        "promotion_authorized": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    candidate["candidate_hash"] = canonical_hash(candidate)
    candidate_path = root / "candidate.json"
    candidate_path.write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return IQLTrainingReceipt(
        candidate_path=str(candidate_path),
        candidate_hash=str(candidate["candidate_hash"]),
        weights_path=str(weights_path),
        weights_hash=str(candidate["artifact"]["weights_hash"]),
        dataset_manifest_hash=str(manifest["manifest_hash"]),
        training_steps=config.steps,
        train_transition_count=int(np.sum(train_mask)),
        validation_transition_count=int(np.sum(validation_mask)),
        reserved_transition_count=int(np.sum(reserved_mask)),
        validation_normalized_mse_before=before,
        validation_normalized_mse_after=after,
        device=str(device),
        status=status,
    )


def _scalar_return(reward: np.ndarray, cost: np.ndarray) -> np.ndarray:
    reward_weights = np.asarray((1.0, 1.0, 0.20, 0.002, 1.0, 2.0, 0.01), dtype=np.float32)
    cost_weights = np.asarray((100.0, 20.0, 0.10, 10.0, 100.0), dtype=np.float32)
    return (reward @ reward_weights - cost @ cost_weights).astype(np.float32)


def _actor_mse(actor: Any, state: Any, action: Any, indices: Any) -> float:
    import torch

    with torch.no_grad():
        return float(torch.mean((actor(state[indices]) - action[indices]) ** 2).cpu())


def _safe_actor_arrays(
    actor: Any,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = {
        name.replace(".", "__"): value.detach().cpu().numpy().astype(np.float32)
        for name, value in actor.state_dict().items()
    }
    arrays.update(
        {
            "state_mean": state_mean.astype(np.float32),
            "state_std": state_std.astype(np.float32),
            "action_mean": action_mean.astype(np.float32),
            "action_std": action_std.astype(np.float32),
        }
    )
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise RuntimeError("IQL actor contains non-finite parameters")
    return arrays


def _verify_dataset_manifest(manifest: dict[str, Any]) -> None:
    claimed = manifest.get("manifest_hash")
    unsigned = dict(manifest)
    unsigned.pop("manifest_hash", None)
    if claimed != canonical_hash(unsigned):
        raise ValueError("IQL dataset manifest hash mismatch")
    if manifest.get("training_eligible") is not True:
        raise ValueError("IQL dataset is not training eligible")
    if manifest.get("promotion_truth_allowed") is not False:
        raise ValueError("IQL discovery data must not be promotion truth")
    profile = manifest.get("data_profile", {})
    if profile.get("offline_rl_ready") is not True:
        raise ValueError("IQL dataset does not satisfy offline-RL semantics")


def _verify_arrays(arrays: dict[str, np.ndarray]) -> None:
    required = {
        "state",
        "next_state",
        "executed_action",
        "reward_vector",
        "cost_vector",
        "episode_index",
        "terminal",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"IQL dataset arrays are missing: {missing}")
    count = len(arrays["state"])
    expected = {
        "state": (count, len(STATE_FEATURES)),
        "next_state": (count, len(STATE_FEATURES)),
        "executed_action": (count, 29),
        "reward_vector": (count, len(REWARD_NAMES)),
        "cost_vector": (count, len(COST_NAMES)),
        "episode_index": (count,),
        "terminal": (count,),
    }
    invalid = [name for name, shape in expected.items() if arrays[name].shape != shape]
    if invalid or count < 3:
        raise ValueError(f"IQL dataset shapes are invalid: {invalid}")
    if not all(np.all(np.isfinite(arrays[name])) for name in required):
        raise ValueError("IQL dataset contains non-finite values")
    if np.any(arrays["cost_vector"] < 0.0):
        raise ValueError("IQL cost vector must be non-negative")


def _array_content_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return "sha256:" + digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "IQLResidualDecision",
    "IQLResidualGuardConfig",
    "IQLTrainingConfig",
    "IQLTrainingReceipt",
    "NumpyIQLActor",
    "SupportBoundIQLResidualActor",
    "train_recovery_iql",
]
