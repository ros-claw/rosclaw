"""Execution permit issuance, validation and revocation (plan §7.3).

A permit is valid **only** for the exact hashes it was issued against.  Any of
the following invalidates it immediately:

- body state/profile change
- calibration change
- policy contract change
- mapping change
- transport profile change
- worker restart (worker_generation change)
- serial reconnect
- expiry
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.execution.schema import ExecutionPermit


class PermitError(ValueError):
    """Permit issuance/validation failure; message starts with a code."""


class PermitManager:
    """Issue and validate execution permits."""

    def __init__(self, store_dir: str | Path | None = None) -> None:
        self._permits: dict[str, ExecutionPermit] = {}
        self._revoked: dict[str, str] = {}
        self._store_dir = Path(store_dir).expanduser() if store_dir else None

    def issue(
        self,
        *,
        body_id: str,
        policy_contract_hash: str,
        body_hash: str,
        calibration_hash: str,
        mapping_hash: str,
        transport_profile_hash: str,
        allowed_representation: str = "joint_position",
        allowed_unit: str = "raw_device_unit",
        max_step_delta_raw: float = 30.0,
        max_speed: int = 100,
        max_force_g: float = 100.0,
        expires_in_sec: float = 120.0,
        operator_armed: bool = False,
        physical_estop_confirmed: bool = False,
        task: str = "",
        calibration_status: str | None = None,
    ) -> ExecutionPermit:
        """Issue a new permit.  Requires operator arming + estop confirmation."""
        if not operator_armed:
            raise PermitError("permit_not_armed: operator_armed is required")
        if not physical_estop_confirmed:
            raise PermitError("permit_estop_unconfirmed: physical estop must be confirmed")
        if calibration_status is not None and calibration_status != "validated":
            raise PermitError(
                f"calibration_not_validated: status {calibration_status!r}; permit denied"
            )
        for name, value in (
            ("policy_contract_hash", policy_contract_hash),
            ("body_hash", body_hash),
            ("calibration_hash", calibration_hash),
            ("mapping_hash", mapping_hash),
            ("transport_profile_hash", transport_profile_hash),
        ):
            if not value:
                raise PermitError(f"permit_missing_hash: {name} is required")
        if max_step_delta_raw <= 0:
            raise PermitError("permit_invalid_limits: max_step_delta_raw must be positive")

        now = time.monotonic_ns()
        expires_mono = now + int(expires_in_sec * 1e9)
        expires_at = (datetime.now(UTC) + timedelta(seconds=expires_in_sec)).isoformat().replace(
            "+00:00", "Z"
        )
        permit = ExecutionPermit(
            permit_id=f"permit_{uuid.uuid4().hex[:12]}",
            body_id=body_id,
            policy_contract_hash=policy_contract_hash,
            body_hash=body_hash,
            calibration_hash=calibration_hash,
            mapping_hash=mapping_hash,
            transport_profile_hash=transport_profile_hash,
            allowed_representation=allowed_representation,
            allowed_unit=allowed_unit,
            max_step_delta_raw=max_step_delta_raw,
            max_speed=int(max_speed),
            max_force_g=float(max_force_g),
            expires_at=expires_at,
            expires_at_monotonic_ns=expires_mono,
            operator_armed=True,
            physical_estop_confirmed=True,
            task=task,
        )
        self._permits[permit.permit_id] = permit
        return permit

    def get(self, permit_id: str) -> ExecutionPermit | None:
        return self._permits.get(permit_id)

    def revoke(self, permit_id: str, reason: str) -> None:
        self._revoked[permit_id] = reason
        self._permits.pop(permit_id, None)
        if self._store_dir is not None:
            _write_revoked_marker(self._store_dir, permit_id, reason)

    def revoke_all(self, reason: str) -> int:
        count = len(self._permits)
        for permit_id in list(self._permits):
            self.revoke(permit_id, reason)
        return count

    def is_revoked(self, permit_id: str) -> bool:
        return permit_id in self._revoked

    # ------------------------------------------------------------------

    def validate(
        self,
        permit_id: str,
        *,
        body_id: str,
        policy_contract_hash: str,
        body_hash: str,
        calibration_hash: str,
        mapping_hash: str,
        transport_profile_hash: str,
        representation: str,
        units: str,
    ) -> ExecutionPermit:
        """Fail-closed validation of a permit for one execution request."""
        permit = self._permits.get(permit_id)
        if permit is None:
            reason = self._revoked.get(permit_id, "unknown_or_expired")
            raise PermitError(f"permit_revoked: {permit_id} ({reason})")
        if time.monotonic_ns() > permit.expires_at_monotonic_ns:
            self.revoke(permit_id, "expired")
            raise PermitError(f"permit_expired: {permit_id}")
        # Wall-clock check as well: a persisted permit's monotonic deadline is
        # meaningless after a reboot, so expiry is enforced in both domains.
        wall_expiry = _parse_iso(permit.expires_at)
        if wall_expiry is not None and datetime.now(UTC) > wall_expiry:
            self.revoke(permit_id, "expired_wall_clock")
            raise PermitError(f"permit_expired: {permit_id} (wall clock)")
        if permit.body_id != body_id:
            raise PermitError(
                f"permit_body_mismatch: {body_id} != {permit.body_id}"
            )
        checks = (
            ("policy_contract_hash", policy_contract_hash, permit.policy_contract_hash),
            ("body_hash", body_hash, permit.body_hash),
            ("calibration_hash", calibration_hash, permit.calibration_hash),
            ("mapping_hash", mapping_hash, permit.mapping_hash),
            ("transport_profile_hash", transport_profile_hash, permit.transport_profile_hash),
        )
        for name, current, expected in checks:
            if current != expected:
                self.revoke(permit_id, f"{name}_changed")
                raise PermitError(
                    f"permit_hash_mismatch: {name} changed; permit revoked"
                )
        if representation != permit.allowed_representation:
            raise PermitError(
                f"permit_representation_mismatch: {representation} not allowed"
            )
        if units != permit.allowed_unit:
            raise PermitError(f"permit_unit_mismatch: {units} not allowed")
        return permit

    # ------------------------------------------------------------------

    def on_worker_restart(self) -> int:
        """Revoke every permit when the policy worker restarts."""
        return self.revoke_all("worker_restart")

    def on_serial_reconnect(self) -> int:
        """Revoke every permit when the serial link reconnects."""
        return self.revoke_all("serial_reconnect")

    def status(self) -> dict[str, Any]:
        return {
            "active": sorted(self._permits.keys()),
            "revoked": dict(self._revoked),
        }


# ---------------------------------------------------------------------------
# Permit persistence (CLI arm → execute flow spans processes)
# ---------------------------------------------------------------------------


def save_permit(permit: ExecutionPermit, directory: str | Path) -> Path:
    """Persist an issued permit so a later CLI process can validate it."""
    import json

    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{permit.permit_id}.json"
    path.write_text(json.dumps(permit.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_permit(permit_id: str, directory: str | Path) -> ExecutionPermit | None:
    """Load a persisted permit back into a manager-compatible object."""
    import json

    path = Path(directory).expanduser() / f"{permit_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("schema_version", None)
    return ExecutionPermit(**data)


def load_permit_into_manager(
    permit_id: str,
    directory: str | Path,
    manager: PermitManager,
) -> ExecutionPermit | None:
    """Load a persisted permit and register it as active in ``manager``.

    Returns ``None`` when the permit file is missing **or a revoked marker
    exists** — revocation survives process restarts.
    """
    directory = Path(directory).expanduser()
    if (directory / f"{permit_id}.revoked.json").exists():
        manager._revoked[permit_id] = "revoked(persisted)"
        return None
    permit = load_permit(permit_id, directory)
    if permit is not None:
        manager._permits[permit.permit_id] = permit
    return permit


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _write_revoked_marker(store_dir: Path, permit_id: str, reason: str) -> None:
    import json

    store_dir.mkdir(parents=True, exist_ok=True)
    marker = store_dir / f"{permit_id}.revoked.json"
    payload = {
        "permit_id": permit_id,
        "reason": reason,
        "revoked_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    marker.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
