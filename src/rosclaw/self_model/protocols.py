"""Generic Operational Self Model protocols (Physical Evolution Lab §8.2, PR-PE-4).

The v3 Self is NOT personality or consciousness — it is the
engineering-verifiable estimate of:

> 我当前是什么身体、处于什么状态、执行这个动作后会发生什么、
> 结果是否由我造成、我还具备哪些可靠能力。

The existing ``self_model`` (HybridForwardSelfModel, PredictionMonitor,
AgencyEstimator) is G1-shaped (pelvis/COM/foot contact/ball).  These
protocols are the abstraction layer: bodies provide ADAPTERS; the
residual/monitor/agency machinery above stays body-agnostic.  RH56 must
never pretend to be G1 (v3 §8.2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

PROTOCOLS_VERSION = "rosclaw.self_protocols.v1"


@dataclass(frozen=True)
class SelfPrediction:
    """One forward prediction in BODY-AGNOSTIC units (§8.4 targets).

    Every value is body-specific in meaning but shared in shape: the
    residual machinery compares predicted vs measured per channel
    without knowing what a pelvis is."""

    channels: dict[str, float]  # e.g. next joint positions, ttr, temp delta
    uncertainty: dict[str, float]  # per-channel std (calibrated separately)
    model_hash: str
    analytical_only: bool  # True when the bounded residual contributed nothing


@dataclass(frozen=True)
class SelfObservation:
    """The measured counterpart of SelfPrediction (same channel names)."""

    channels: dict[str, float]
    timestamp_ns: int
    source: str  # "telemetry" | "visual" | "fused"


class SelfStateAdapter(ABC):
    """Body → self-state channels (§8.3)."""

    @abstractmethod
    def body_id(self) -> str: ...

    @abstractmethod
    def body_hash(self) -> str:
        """Binds every prediction/snapshot to THIS body's measured config —
        a mutated body must fail to load models bound to the old hash."""

    @abstractmethod
    def current_state(self) -> dict[str, Any]:
        """The body's self-state blocks (identity/kinematics/interaction/
        health/perception/task — adapters name their own blocks)."""

    @abstractmethod
    def health_channels(self) -> dict[str, float]:
        """Channels used for regime/health reasoning (temperature, slope,
        error rates, latency)."""


class SelfActionAdapter(ABC):
    """Action → forward-model input channels."""

    @abstractmethod
    def action_channels(self, action: dict[str, Any]) -> dict[str, float]: ...


class ForwardModelProtocol(Protocol):
    """Analytical prior + bounded residual (§8.4)."""

    @property
    @abstractmethod
    def model_hash(self) -> str: ...

    @abstractmethod
    def predict(self, state: dict[str, float], action: dict[str, float]) -> SelfPrediction: ...

    @abstractmethod
    def expected_body_hash(self) -> str:
        """The body hash this model was built for — loading against a
        different body hash must raise (v3 §8.7/PR-PE-4 acceptance)."""


class ResidualEncoder(ABC):
    """Predicted × measured → named residual channels (§8.5)."""

    @abstractmethod
    def encode(
        self, prediction: SelfPrediction, observation: SelfObservation
    ) -> dict[str, float]: ...


@runtime_checkable
class AgencyEvidenceAdapter(Protocol):
    """Body → agency evidence (§8.6 four-class attribution)."""

    def agency_channels(self) -> dict[str, float]:
        """action_magnitude, prediction_error, external_force_evidence,
        sensor_inconsistency — the AgencyEstimator's four inputs."""
        ...
