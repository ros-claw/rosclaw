"""Public, versioned schemas used by ROSClaw interaction adapters."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ActionDisplay(BaseModel):
    """Human-readable description of one exact physical action."""

    model_config = ConfigDict(extra="allow")

    schema_version: Literal["rosclaw.action-display.v2"] = "rosclaw.action-display.v2"
    title: str = Field(min_length=1, max_length=160)
    summary: str = Field(min_length=1, max_length=1000)
    body: dict[str, Any] = Field(default_factory=dict)
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "HIGH"
    physical_effects: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    abort: list[str] = Field(default_factory=list)

    @classmethod
    def from_legacy(cls, value: dict[str, Any], *, risk_tier: str = "HIGH") -> ActionDisplay:
        """Normalize v1 robot-MCP display dictionaries into ActionDisplay v2."""

        data = dict(value)
        physical_effect = data.pop("physical_effect", None)
        effects = data.pop("physical_effects", None)
        if effects is None:
            effects = [str(physical_effect)] if physical_effect else []
        body = data.pop("body", None)
        if not isinstance(body, dict):
            body = {
                key: data.pop(key)
                for key in tuple(data)
                if key
                not in {"title", "summary", "risk_tier", "constraints", "verification", "abort"}
            }
        normalized_risk = str(data.pop("risk_tier", risk_tier)).upper()
        if normalized_risk not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            normalized_risk = "HIGH"
        return cls(
            title=str(data.pop("title", "Confirm physical action")),
            summary=str(data.pop("summary", "Run one exact action through ROSClaw.")),
            body=body,
            risk_tier=normalized_risk,  # type: ignore[arg-type]
            physical_effects=[str(item) for item in effects],
            constraints=[str(item) for item in data.pop("constraints", [])],
            verification=[str(item) for item in data.pop("verification", [])],
            abort=[str(item) for item in data.pop("abort", [])],
            **data,
        )

    def render_text(self) -> str:
        """Render a compact confirmation card for MCP clients."""

        lines = [self.title, self.summary, f"Risk: {self.risk_tier}"]
        if self.body:
            lines.append(f"Action: {self.body}")
        for label, values in (
            ("Physical effects", self.physical_effects),
            ("Constraints", self.constraints),
            ("Verification", self.verification),
            ("Abort", self.abort),
        ):
            if values:
                lines.append(f"{label}: {'; '.join(values)}")
        return "\n".join(lines)


class InteractionCapabilities(BaseModel):
    """Capabilities detected from the connected MCP client."""

    form_elicitation: bool = False
    url_elicitation: bool = False
    asynchronous_elicitation: bool = False
    progress: bool = False
    cancellation: bool = False
