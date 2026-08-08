"""Content-addressed, SIM-only anticipatory balance memory for moving kicks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_json

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_ARTIFACT_BYTES = 64 * 1024


@dataclass(frozen=True)
class G1MovingBallBalanceArtifact:
    """A learned high-level balance parameter, never a joint command.

    The artifact is bound to the exact body, motion prior, and recovery
    controller used during bounded search.  Its only action is a lateral COM
    preload already constrained by :class:`ShotParameters`.
    """

    body_hash: str
    motion_hash: str
    recovery_config_hash: str
    training_dataset_hash: str
    com_shift_y_m: float
    lateral_com_shift_y_m: float
    minimum_lateral_offset_m: float
    development_case_count: int
    training_seed: int
    velocity_com_shift_y_m: float | None = None
    maximum_velocity_context_mps: float | None = None
    large_lateral_com_shift_y_m: float | None = None
    minimum_large_lateral_offset_m: float | None = None
    selection_method: str = "BOUNDED_CONTEXTUAL_GRID_SEARCH"
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.g1_goalforge.moving_ball_balance_artifact.v2"

    def __post_init__(self) -> None:
        for label, value in (
            ("body_hash", self.body_hash),
            ("motion_hash", self.motion_hash),
            ("recovery_config_hash", self.recovery_config_hash),
            ("training_dataset_hash", self.training_dataset_hash),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256 content hash")
        supported_schemas = {
            "rosclaw.g1_goalforge.moving_ball_balance_artifact.v2",
            "rosclaw.g1_goalforge.moving_ball_balance_artifact.v3",
            "rosclaw.g1_goalforge.moving_ball_balance_artifact.v4",
        }
        if self.schema_version not in supported_schemas:
            raise ValueError("unsupported moving-ball balance artifact schema")
        for value in (self.com_shift_y_m, self.lateral_com_shift_y_m):
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not -0.08 <= value <= 0.08
            ):
                raise ValueError("moving-ball COM shifts must be finite and in [-0.08, 0.08] m")
        if self.schema_version.endswith(".v2"):
            if (
                self.velocity_com_shift_y_m is not None
                or self.maximum_velocity_context_mps is not None
                or self.large_lateral_com_shift_y_m is not None
                or self.minimum_large_lateral_offset_m is not None
            ):
                raise ValueError("v2 moving-ball balance artifacts cannot enable velocity context")
        else:
            velocity_shift = self.velocity_com_shift_y_m
            velocity_limit = self.maximum_velocity_context_mps
            if (
                isinstance(velocity_shift, bool)
                or not isinstance(velocity_shift, (int, float))
                or not math.isfinite(velocity_shift)
                or not -0.08 <= velocity_shift <= 0.08
            ):
                raise ValueError("velocity-context COM shift must be in [-0.08, 0.08] m")
            if (
                isinstance(velocity_limit, bool)
                or not isinstance(velocity_limit, (int, float))
                or not math.isfinite(velocity_limit)
                or not 0.0 < velocity_limit <= 0.20
            ):
                raise ValueError("velocity-context threshold must be in (0, 0.20] m/s")
        if not self.schema_version.endswith(".v4"):
            if (
                self.large_lateral_com_shift_y_m is not None
                or self.minimum_large_lateral_offset_m is not None
            ):
                raise ValueError("only v4 balance artifacts can enable large-lateral context")
        else:
            large_shift = self.large_lateral_com_shift_y_m
            large_threshold = self.minimum_large_lateral_offset_m
            if (
                isinstance(large_shift, bool)
                or not isinstance(large_shift, (int, float))
                or not math.isfinite(large_shift)
                or not -0.08 <= large_shift <= 0.08
            ):
                raise ValueError("large-lateral COM shift must be in [-0.08, 0.08] m")
            if (
                isinstance(large_threshold, bool)
                or not isinstance(large_threshold, (int, float))
                or not math.isfinite(large_threshold)
                or not self.minimum_lateral_offset_m < large_threshold <= 0.16
            ):
                raise ValueError("large-lateral threshold must exceed the lateral threshold")
        if (
            isinstance(self.minimum_lateral_offset_m, bool)
            or not math.isfinite(self.minimum_lateral_offset_m)
            or not 0.0 < self.minimum_lateral_offset_m <= 0.16
        ):
            raise ValueError("moving-ball lateral threshold must be in (0, 0.16] m")
        if (
            isinstance(self.development_case_count, bool)
            or not isinstance(self.development_case_count, int)
            or self.development_case_count <= 0
        ):
            raise ValueError("moving-ball balance case count must be positive")
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed < 0
        ):
            raise ValueError("moving-ball balance seed must be non-negative")
        if self.selection_method != "BOUNDED_CONTEXTUAL_GRID_SEARCH":
            raise ValueError("unsupported moving-ball balance selection method")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("moving-ball balance artifact must remain SIM_ONLY")

    @property
    def artifact_hash(self) -> str:
        return hash_json(self.to_dict())

    def com_shift_for(
        self,
        *,
        predicted_ball_y_m: float,
        predicted_ball_speed_mps: float | None = None,
    ) -> float:
        if not math.isfinite(predicted_ball_y_m):
            raise ValueError("predicted lateral ball position must be finite")
        if self.schema_version.endswith(".v4"):
            large_threshold = self.minimum_large_lateral_offset_m
            large_shift = self.large_lateral_com_shift_y_m
            if large_threshold is None or large_shift is None:
                raise RuntimeError("validated v4 large-lateral context is incomplete")
            if abs(predicted_ball_y_m) >= large_threshold:
                return large_shift
        if abs(predicted_ball_y_m) >= self.minimum_lateral_offset_m:
            return self.lateral_com_shift_y_m
        if self.schema_version.endswith((".v3", ".v4")):
            if (
                predicted_ball_speed_mps is None
                or not math.isfinite(predicted_ball_speed_mps)
                or predicted_ball_speed_mps < 0.0
            ):
                raise ValueError("v3 balance memory requires a finite non-negative ball speed")
            velocity_limit = self.maximum_velocity_context_mps
            velocity_shift = self.velocity_com_shift_y_m
            if velocity_limit is None or velocity_shift is None:
                raise RuntimeError("validated v3 velocity context is incomplete")
            if predicted_ball_speed_mps <= velocity_limit:
                return velocity_shift
        return self.com_shift_y_m

    def require_compatible(
        self,
        *,
        body_hash: str,
        motion_hash: str,
        recovery_config_hash: str,
    ) -> None:
        if body_hash != self.body_hash:
            raise ValueError("moving-ball balance Body hash mismatch")
        if motion_hash != self.motion_hash:
            raise ValueError("moving-ball balance motion hash mismatch")
        if recovery_config_hash != self.recovery_config_hash:
            raise ValueError("moving-ball balance recovery config hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.schema_version.endswith(".v2"):
            # Preserve the exact v2 payload and all existing artifact hashes.
            value.pop("velocity_com_shift_y_m")
            value.pop("maximum_velocity_context_mps")
            value.pop("large_lateral_com_shift_y_m")
            value.pop("minimum_large_lateral_offset_m")
        elif self.schema_version.endswith(".v3"):
            value.pop("large_lateral_com_shift_y_m")
            value.pop("minimum_large_lateral_offset_m")
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> G1MovingBallBalanceArtifact:
        expected = {
            "schema_version",
            "body_hash",
            "motion_hash",
            "recovery_config_hash",
            "training_dataset_hash",
            "com_shift_y_m",
            "lateral_com_shift_y_m",
            "minimum_lateral_offset_m",
            "development_case_count",
            "training_seed",
            "selection_method",
            "activation_ceiling",
        }
        schema = value.get("schema_version")
        if schema in {
            "rosclaw.g1_goalforge.moving_ball_balance_artifact.v3",
            "rosclaw.g1_goalforge.moving_ball_balance_artifact.v4",
        }:
            expected.update(
                {
                    "velocity_com_shift_y_m",
                    "maximum_velocity_context_mps",
                }
            )
        if schema == "rosclaw.g1_goalforge.moving_ball_balance_artifact.v4":
            expected.update(
                {
                    "large_lateral_com_shift_y_m",
                    "minimum_large_lateral_offset_m",
                }
            )
        if set(value) != expected:
            raise ValueError("moving-ball balance artifact fields are invalid")
        count = value["development_case_count"]
        seed = value["training_seed"]
        shift = value["com_shift_y_m"]
        lateral_shift = value["lateral_com_shift_y_m"]
        lateral_threshold = value["minimum_lateral_offset_m"]
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("moving-ball balance case count must be an integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("moving-ball balance seed must be an integer")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in (shift, lateral_shift, lateral_threshold)
        ):
            raise ValueError("moving-ball balance parameters must be numeric")
        return cls(
            schema_version=str(value["schema_version"]),
            body_hash=str(value["body_hash"]),
            motion_hash=str(value["motion_hash"]),
            recovery_config_hash=str(value["recovery_config_hash"]),
            training_dataset_hash=str(value["training_dataset_hash"]),
            com_shift_y_m=float(shift),
            lateral_com_shift_y_m=float(lateral_shift),
            minimum_lateral_offset_m=float(lateral_threshold),
            development_case_count=count,
            training_seed=seed,
            velocity_com_shift_y_m=(
                _strict_optional_float(value["velocity_com_shift_y_m"])
                if schema
                in {
                    "rosclaw.g1_goalforge.moving_ball_balance_artifact.v3",
                    "rosclaw.g1_goalforge.moving_ball_balance_artifact.v4",
                }
                else None
            ),
            maximum_velocity_context_mps=(
                _strict_optional_float(value["maximum_velocity_context_mps"])
                if schema
                in {
                    "rosclaw.g1_goalforge.moving_ball_balance_artifact.v3",
                    "rosclaw.g1_goalforge.moving_ball_balance_artifact.v4",
                }
                else None
            ),
            large_lateral_com_shift_y_m=(
                _strict_optional_float(value["large_lateral_com_shift_y_m"])
                if schema == "rosclaw.g1_goalforge.moving_ball_balance_artifact.v4"
                else None
            ),
            minimum_large_lateral_offset_m=(
                _strict_optional_float(value["minimum_large_lateral_offset_m"])
                if schema == "rosclaw.g1_goalforge.moving_ball_balance_artifact.v4"
                else None
            ),
            selection_method=str(value["selection_method"]),
            activation_ceiling=str(value["activation_ceiling"]),
        )


def _strict_optional_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("moving-ball balance velocity context must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("moving-ball balance velocity context must be finite")
    return result


def serialize_g1_moving_ball_balance_artifact(
    artifact: G1MovingBallBalanceArtifact,
) -> bytes:
    return (json.dumps(artifact.to_dict(), sort_keys=True, indent=2) + "\n").encode()


def load_g1_moving_ball_balance_artifact(
    path: Path,
    *,
    expected_body_hash: str,
    expected_motion_hash: str,
    expected_recovery_config_hash: str,
) -> G1MovingBallBalanceArtifact:
    payload = path.expanduser().read_bytes()
    if not payload or len(payload) > _MAX_ARTIFACT_BYTES:
        raise ValueError("moving-ball balance artifact size is invalid")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("moving-ball balance artifact is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("moving-ball balance artifact root must be an object")
    artifact = G1MovingBallBalanceArtifact.from_dict(value)
    artifact.require_compatible(
        body_hash=expected_body_hash,
        motion_hash=expected_motion_hash,
        recovery_config_hash=expected_recovery_config_hash,
    )
    return artifact


def moving_ball_balance_payload_hash(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = [
    "G1MovingBallBalanceArtifact",
    "load_g1_moving_ball_balance_artifact",
    "moving_ball_balance_payload_hash",
    "serialize_g1_moving_ball_balance_artifact",
]
