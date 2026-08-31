"""Fail-closed numerical-runtime contracts for reproducible learning evidence.

The contract is intentionally independent of any simulator.  It can be bound
to MuJoCo, ONNX Runtime, a learner subprocess, or another numerical workload.
It does not mutate the current process: callers either validate the current
environment or use :meth:`NumericalRuntimeContract.subprocess_environment`
before importing a numerical backend in a child process.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class NumericalEnvironmentCheck:
    """Result of checking a process environment against a pinned contract."""

    expected: Mapping[str, str]
    observed: Mapping[str, str | None]
    mismatches: tuple[str, ...]
    schema_version: str = "rosclaw.continual.numerical_environment_check.v1"

    def __post_init__(self) -> None:
        expected = MappingProxyType(dict(sorted(self.expected.items())))
        observed = MappingProxyType(dict(sorted(self.observed.items())))
        if set(expected) != set(observed):
            raise ValueError("expected and observed environment keys must match")
        object.__setattr__(self, "expected", expected)
        object.__setattr__(self, "observed", observed)

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "expected": dict(self.expected),
            "observed": dict(self.observed),
            "mismatches": list(self.mismatches),
        }


@dataclass(frozen=True)
class NumericalRuntimeContract:
    """Pinned thread, floating-point, seed, and ONNX execution settings."""

    thread_counts: Mapping[str, int]
    random_seed: int
    onnx_execution_providers: tuple[str, ...] = ("CPUExecutionProvider",)
    onnx_intra_op_threads: int = 1
    onnx_inter_op_threads: int = 1
    onnx_execution_mode: str = "ORT_SEQUENTIAL"
    floating_point_mode: str = "IEEE754_STRICT"
    deterministic_compute: bool = True
    allow_tf32: bool = False
    schema_version: str = "rosclaw.continual.numerical_runtime_contract.v1"

    def __post_init__(self) -> None:
        normalized = {str(key): int(value) for key, value in self.thread_counts.items()}
        if set(normalized) != set(THREAD_ENVIRONMENT_KEYS):
            raise ValueError("thread_counts must pin every supported BLAS/OpenMP environment key")
        if any(value <= 0 for value in normalized.values()):
            raise ValueError("thread counts must be positive")
        if not 0 <= self.random_seed < 2**63:
            raise ValueError("random_seed must be in [0, 2**63)")
        if (
            not self.onnx_execution_providers
            or len(set(self.onnx_execution_providers)) != len(self.onnx_execution_providers)
            or any(not provider.strip() for provider in self.onnx_execution_providers)
        ):
            raise ValueError("ONNX execution providers must be non-empty and unique")
        if self.onnx_intra_op_threads <= 0 or self.onnx_inter_op_threads <= 0:
            raise ValueError("ONNX thread counts must be positive")
        if self.onnx_execution_mode not in {"ORT_SEQUENTIAL", "ORT_PARALLEL"}:
            raise ValueError("unsupported ONNX execution mode")
        if self.floating_point_mode not in {"IEEE754_STRICT", "BACKEND_DETERMINISTIC"}:
            raise ValueError("unsupported floating-point mode")
        object.__setattr__(
            self, "thread_counts", MappingProxyType(dict(sorted(normalized.items())))
        )

    @classmethod
    def single_threaded_cpu(cls, *, random_seed: int) -> NumericalRuntimeContract:
        """Return the strict CPU evidence profile used by promotion gates."""

        return cls(
            thread_counts=dict.fromkeys(THREAD_ENVIRONMENT_KEYS, 1),
            random_seed=random_seed,
        )

    @property
    def contract_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def required_environment(self) -> Mapping[str, str]:
        values = {key: str(value) for key, value in self.thread_counts.items()}
        values.update(
            {
                "PYTHONHASHSEED": str(self.random_seed),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
        )
        return MappingProxyType(dict(sorted(values.items())))

    def verify_environment(
        self,
        environment: Mapping[str, str] | None = None,
    ) -> NumericalEnvironmentCheck:
        """Fail closed on missing as well as mismatched numerical settings."""

        source = os.environ if environment is None else environment
        expected = self.required_environment
        observed = {key: source.get(key) for key in expected}
        mismatches = tuple(
            key for key, expected_value in expected.items() if observed[key] != expected_value
        )
        return NumericalEnvironmentCheck(
            expected=expected,
            observed=observed,
            mismatches=mismatches,
        )

    def subprocess_environment(
        self,
        base: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Build an environment that must be applied before backend imports."""

        result = dict(os.environ if base is None else base)
        result.update(self.required_environment)
        return result

    def onnx_session_settings(self) -> dict[str, Any]:
        """Return import-free settings for a caller-created ONNX session."""

        return {
            "providers": list(self.onnx_execution_providers),
            "intra_op_num_threads": self.onnx_intra_op_threads,
            "inter_op_num_threads": self.onnx_inter_op_threads,
            "execution_mode": self.onnx_execution_mode,
            "deterministic_compute": self.deterministic_compute,
            "allow_tf32": self.allow_tf32,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "thread_counts": dict(self.thread_counts),
            "random_seed": self.random_seed,
            "onnx_execution_providers": list(self.onnx_execution_providers),
            "onnx_intra_op_threads": self.onnx_intra_op_threads,
            "onnx_inter_op_threads": self.onnx_inter_op_threads,
            "onnx_execution_mode": self.onnx_execution_mode,
            "floating_point_mode": self.floating_point_mode,
            "deterministic_compute": self.deterministic_compute,
            "allow_tf32": self.allow_tf32,
        }


__all__ = [
    "NumericalEnvironmentCheck",
    "NumericalRuntimeContract",
    "THREAD_ENVIRONMENT_KEYS",
]
