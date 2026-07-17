"""RH56 transport profile schema ``rosclaw.rh56.transport_profile.v1``.

A transport profile binds a body instance to exactly one control protocol
(RS485/Modbus-RTU 0-1000 raw or CAN 2.0B 0-65535 raw).  P5 forbids guessing
the protocol from the device name: every execution path must load an explicit
profile and pass the binding gate in :func:`validate_transport_binding`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

TRANSPORT_PROFILE_SCHEMA_VERSION = "rosclaw.rh56.transport_profile.v1"

TRANSPORT_TYPES = ("serial_modbus_rtu", "can_2_0b")


class TransportBindingError(ValueError):
    """Raised when a body/policy/provider binding violates the transport profile.

    The message always starts with a machine-readable error code:

    - ``transport_profile_mismatch``
    - ``command_scale_mismatch``
    - ``actuator_count_mismatch``
    - ``provider_transport_mismatch``
    """


@dataclass
class TransportConfig:
    """Physical transport parameters."""

    type: str
    device: str = ""
    baudrate: int = 115200
    slave_id: int = 1
    # CAN-specific
    channel: str = ""
    extended_id: bool = True

    def __post_init__(self) -> None:
        if self.type not in TRANSPORT_TYPES:
            raise TransportBindingError(
                f"transport_profile_mismatch: unknown transport type {self.type!r}; "
                f"expected one of {TRANSPORT_TYPES}"
            )


@dataclass
class CommandConfig:
    """Command value conventions for the transport."""

    actuator_count: int
    position_range: list[int]
    position_convention: dict[str, int] = field(default_factory=dict)
    speed_range: list[int] = field(default_factory=lambda: [0, 1000])
    force_range: list[int] | None = None
    force_unit: str = "gram"
    current_unit: str = "milliampere"


@dataclass
class TransportProfile:
    """A ``rosclaw.rh56.transport_profile.v1`` document."""

    id: str
    transport: TransportConfig
    command: CommandConfig
    action_order: list[str]
    schema_version: str = TRANSPORT_PROFILE_SCHEMA_VERSION
    feedback: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransportProfile:
        schema = data.get("schema_version")
        if schema != TRANSPORT_PROFILE_SCHEMA_VERSION:
            raise TransportBindingError(
                f"transport_profile_mismatch: schema_version {schema!r} != "
                f"{TRANSPORT_PROFILE_SCHEMA_VERSION!r}"
            )
        transport = TransportConfig(**data.get("transport", {}))
        command = CommandConfig(**data.get("command", {}))
        action_order = [str(n) for n in data.get("action_order", [])]
        profile = cls(
            id=str(data.get("id", "")),
            transport=transport,
            command=command,
            action_order=action_order,
            schema_version=schema,
            feedback=dict(data.get("feedback", {})),
            metadata=dict(data.get("metadata", {})),
        )
        profile.validate_internal()
        return profile

    def validate_internal(self) -> None:
        """Fail-closed consistency checks inside the profile itself."""
        if not self.id:
            raise TransportBindingError("transport_profile_mismatch: profile id is empty")
        if len(self.action_order) != self.command.actuator_count:
            raise TransportBindingError(
                f"actuator_count_mismatch: action_order has {len(self.action_order)} entries "
                f"but command.actuator_count is {self.command.actuator_count}"
            )
        lo, hi = self.command.position_range
        if not (0 <= lo < hi):
            raise TransportBindingError(
                f"command_scale_mismatch: invalid position_range {self.command.position_range}"
            )
        convention = self.command.position_convention
        for key in ("closed", "open"):
            if key not in convention:
                raise TransportBindingError(
                    f"command_scale_mismatch: position_convention missing {key!r}"
                )
        if self.transport.type == "serial_modbus_rtu" and not self.transport.device:
            raise TransportBindingError(
                "transport_profile_mismatch: serial_modbus_rtu requires transport.device"
            )
        if self.transport.type == "can_2_0b" and not self.transport.channel:
            raise TransportBindingError(
                "transport_profile_mismatch: can_2_0b requires transport.channel"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------

    @property
    def actuator_count(self) -> int:
        return self.command.actuator_count

    @property
    def position_range(self) -> tuple[int, int]:
        lo, hi = self.command.position_range
        return int(lo), int(hi)

    def position_open(self) -> int:
        return int(self.command.position_convention["open"])

    def position_closed(self) -> int:
        return int(self.command.position_convention["closed"])

    def clamp_position(self, value: float) -> int:
        lo, hi = self.position_range
        return max(lo, min(hi, int(round(value))))

    def content_hash(self) -> str:
        """Stable hash over the semantic content (used for permits)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def load_transport_profile(path: str | Path) -> TransportProfile:
    """Load a transport profile from YAML or JSON, fail-closed on errors."""
    path = Path(path)
    if not path.exists():
        raise TransportBindingError(f"transport_profile_mismatch: profile not found: {path}")
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TransportBindingError(f"transport_profile_mismatch: {path} is not a mapping")
    return TransportProfile.from_dict(data)


# ---------------------------------------------------------------------------
# Binding gate
# ---------------------------------------------------------------------------


def validate_transport_binding(
    profile: TransportProfile,
    *,
    provider_ref: str | None = None,
    device_path: str | None = None,
    action_dim: int | None = None,
    action_names: list[str] | None = None,
    position_range: tuple[int, int] | None = None,
) -> None:
    """Validate that a concrete binding matches the profile.

    Raises :class:`TransportBindingError` with a machine-readable code on any
    mismatch.  Returns ``None`` on success.
    """
    if provider_ref is not None:
        expected = profile.metadata.get("provider_ref")
        if expected is not None and provider_ref != expected:
            raise TransportBindingError(
                f"provider_transport_mismatch: provider_ref {provider_ref!r} does not match "
                f"profile {profile.id!r} provider_ref {expected!r}"
            )
        # Guard against the classic RS485-vs-CAN mix-up.
        if profile.transport.type == "serial_modbus_rtu" and "can" in provider_ref.lower():
            raise TransportBindingError(
                f"provider_transport_mismatch: CAN provider {provider_ref!r} cannot bind "
                f"RS485 profile {profile.id!r}"
            )
        if profile.transport.type == "can_2_0b" and (
            "serial" in provider_ref.lower() or "rs485" in provider_ref.lower()
        ):
            raise TransportBindingError(
                f"provider_transport_mismatch: serial provider {provider_ref!r} cannot bind "
                f"CAN profile {profile.id!r}"
            )

    if (
        device_path is not None
        and profile.transport.type == "serial_modbus_rtu"
        and device_path != profile.transport.device
    ):
        raise TransportBindingError(
            f"transport_profile_mismatch: device {device_path!r} != profile device "
            f"{profile.transport.device!r}"
        )

    if action_dim is not None and action_dim != profile.command.actuator_count:
        raise TransportBindingError(
            f"actuator_count_mismatch: action dim {action_dim} != profile actuator_count "
            f"{profile.command.actuator_count}"
        )

    if action_names is not None and list(action_names) != list(profile.action_order):
        raise TransportBindingError(
            f"actuator_count_mismatch: action names {list(action_names)!r} do not match "
            f"profile action_order {list(profile.action_order)!r}"
        )

    if position_range is not None and tuple(position_range) != profile.position_range:
        raise TransportBindingError(
            f"command_scale_mismatch: position range {tuple(position_range)} != profile "
            f"range {profile.position_range}"
        )
