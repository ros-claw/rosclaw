"""后端接口 phase-1（PR-MH2，ADR-0014，规格 §12）。

Maturity: experimental（ADR-0000 §4）。

只声明模型服务四方法；state/rollout/observe/render 属 MH3+，
此处不声明、不留 stub——避免假能力。
"""

from __future__ import annotations

from typing import Any, Protocol

from rosclaw.sim.contracts import (
    ModelInspection,
    ModelPatchResult,
    ModelReference,
    SimulationBackendCapabilities,
)


class SimBackend(Protocol):
    """物理仿真后端 phase-1 接口。"""

    name: str

    def capabilities(self) -> SimulationBackendCapabilities: ...

    def load_model(self, asset_ref: str) -> ModelReference: ...

    def inspect_model(self, model_ref: str) -> ModelInspection: ...

    def compile_model(self, model_ref: str) -> ModelInspection: ...

    def describe_model(self, model_ref: str) -> ModelReference: ...

    def patch_model(self, model_ref: str, patches: list[dict[str, Any]]) -> ModelPatchResult: ...
