"""InteractionRequestV1 (补充实施文档 §5.3)：select/confirm/input/editor 交互。

通用交互用于非安全场景（选择 mission、确认导出等）。物理/授权决定永远
走专用 /v1/approvals/{request_id}/decide，不与 generic confirm 混用——
generic confirm 不能伪造 approval（§12.5）。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from rosclaw.contracts.common import ContractModel


class InteractionRequestV1(ContractModel):
    SCHEMA = "rosclaw.ui.interaction_request.v1"

    schema_version: Literal["rosclaw.ui.interaction_request.v1"] = (
        "rosclaw.ui.interaction_request.v1"
    )
    interaction_id: str = Field(min_length=1)
    mission_id: str | None = None
    kind: Literal["select", "confirm", "input", "editor"]
    title: str = ""
    prompt: str = ""
    options: list[dict[str, Any]] = Field(default_factory=list)
    default: Any = None
    #: masked=True → client uses masked input; value never journaled
    masked: bool = False
    created_at: str
    expires_at: str | None = None
    status: Literal["pending", "responded", "expired", "cancelled"] = "pending"
