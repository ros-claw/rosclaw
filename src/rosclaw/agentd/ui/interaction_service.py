"""InteractionService (批次 B §5.3)：通用 select/confirm/input/editor。

硬边界：generic interaction 永远不能改变授权/物理状态——/approve、
/estop 走专用端点；secret 交互的值不落 journal，只存 redacted 标记。
"""

from __future__ import annotations

from datetime import UTC, datetime

from rosclaw.agentd.mission.store import _utcnow
from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.contracts.ui.interactions import InteractionRequestV1


class InteractionService:
    def __init__(self) -> None:
        self._pending: dict[str, InteractionRequestV1] = {}
        self._responses: dict[str, dict] = {}
        self._idempotency: dict[str, dict] = {}

    def create(
        self,
        kind: str,
        *,
        title: str = "",
        prompt: str = "",
        options: list[dict] | None = None,
        default=None,
        masked: bool = False,
        mission_id: str | None = None,
        expires_at: str | None = None,
    ) -> InteractionRequestV1:
        if kind not in ("select", "confirm", "input", "editor"):
            raise ValidationError(f"unknown interaction kind {kind!r}")
        request = InteractionRequestV1(
            interaction_id=new_id("uir"),
            mission_id=mission_id,
            kind=kind,  # type: ignore[arg-type]
            title=title,
            prompt=prompt,
            options=options or [],
            default=default,
            masked=masked,
            created_at=_utcnow(),
            expires_at=expires_at,
        )
        self._pending[request.interaction_id] = request
        return request

    def get(self, interaction_id: str) -> InteractionRequestV1 | None:
        return self._pending.get(interaction_id)

    def respond(self, interaction_id: str, *, value, idempotency_key: str = "") -> dict:
        if idempotency_key and idempotency_key in self._idempotency:
            return self._idempotency[idempotency_key]
        request = self._pending.get(interaction_id)
        if request is None:
            raise ValidationError(f"unknown interaction {interaction_id!r}")
        if request.status != "pending":
            raise ValidationError(f"interaction {interaction_id!r} already {request.status}")
        if request.expires_at and datetime.fromisoformat(request.expires_at) < datetime.now(UTC):
            self._pending[interaction_id] = request.model_copy(update={"status": "expired"})
            raise ValidationError(f"interaction {interaction_id!r} expired")
        if request.kind == "select" and request.options:
            valid = {str(o.get("value")) for o in request.options}
            if str(value) not in valid:
                raise ValidationError(f"value {value!r} not among select options")
        if request.kind == "confirm" and not isinstance(value, bool):
            raise ValidationError("confirm interactions require a boolean value")
        self._pending[interaction_id] = request.model_copy(update={"status": "responded"})
        # masked 值只回传给调用方内存路径，永不进入可持久化的响应记录。
        record = {
            "interaction_id": interaction_id,
            "status": "responded",
            "value": "<redacted-secret>" if request.masked else value,
            "masked": request.masked,
        }
        self._responses[interaction_id] = record
        if idempotency_key:
            self._idempotency[idempotency_key] = record
        return record
