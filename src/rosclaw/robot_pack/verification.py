"""Evidence-producing Robot Pack contract and read-only verification."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from rosclaw.body.resolver import BodyResolver
from rosclaw.firstboot.workspace import resolve_home
from rosclaw.robot_pack.discovery import (
    DiscoveryReport,
    discover_realsense_devices,
    match_device_variant,
)
from rosclaw.robot_pack.instance import (
    RobotInstanceConfig,
    RobotInstanceError,
    load_robot_instance,
    resolve_adapter_binding,
)
from rosclaw.robot_pack.schema import RobotPackManifest, SupportTier
from rosclaw.robot_pack.store import InstalledRobotPack, RobotPackStore, RobotPackStoreError
from rosclaw.robot_pack.verifier import PackVerificationResult, verify_robot_pack

CheckStatus = Literal["pass", "fail", "warn", "skip"]
AdapterProbe = Callable[[str, Path], list[dict[str, Any]]]


@dataclass(frozen=True)
class VerificationCheckResult:
    id: str
    status: CheckStatus
    message: str
    evidence: dict[str, Any] | None = None


@dataclass(frozen=True)
class RobotPackVerificationReport:
    schema_version: str
    evidence_id: str
    generated_at: str
    pack_ref: str
    manifest_digest: str
    stage: str
    passed: bool
    support_tier: str
    observed_candidate_tier: str | None
    canonical_promotion: bool
    promotion_blockers: tuple[str, ...]
    instance_id: str | None
    checks: tuple[VerificationCheckResult, ...]
    environment: dict[str, Any]
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "generated_at": self.generated_at,
            "pack_ref": self.pack_ref,
            "manifest_digest": self.manifest_digest,
            "stage": self.stage,
            "passed": self.passed,
            "support_tier": self.support_tier,
            "observed_candidate_tier": self.observed_candidate_tier,
            "canonical_promotion": self.canonical_promotion,
            "promotion_blockers": list(self.promotion_blockers),
            "instance_id": self.instance_id,
            "checks": [asdict(check) for check in self.checks],
            "environment": self.environment,
            "report_path": self.report_path,
        }


def verify_installed_robot_pack(
    identifier: str,
    *,
    stage: Literal["contract", "read-only"] = "contract",
    instance_id: str | None = None,
    home: str | Path | None = None,
    discovery_report: DiscoveryReport | None = None,
    adapter_probe: AdapterProbe | None = None,
    receipt_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> RobotPackVerificationReport:
    """Verify one installed Pack and persist a non-promoting local evidence report."""

    resolved_home = resolve_home(str(home) if home is not None else None)
    store = RobotPackStore(resolved_home)
    record_identifier = identifier
    resolved_instance_id = instance_id
    if stage == "read-only":
        try:
            preloaded_instance, _preloaded_path = load_robot_instance(
                instance_id or identifier,
                home=resolved_home,
            )
            record_identifier = preloaded_instance.pack.ref
            resolved_instance_id = preloaded_instance.instance_id
        except RobotInstanceError:
            pass
    record = _resolve_record(store, record_identifier)
    integrity = verify_robot_pack(record.path)
    manifest = integrity.manifest
    if manifest is None:
        raise RobotPackStoreError(
            f"Installed Robot Pack manifest is invalid: {'; '.join(integrity.errors)}"
        )
    checks = _contract_checks(record, manifest, integrity)
    configured: RobotInstanceConfig | None = None
    if stage == "read-only":
        configured, read_checks = _read_only_checks(
            identifier=resolved_instance_id or identifier,
            record=record,
            manifest=manifest,
            home=resolved_home,
            discovery_report=discovery_report,
            adapter_probe=adapter_probe,
            receipt_path=Path(receipt_path) if receipt_path is not None else None,
        )
        checks.extend(read_checks)

    passed = all(check.status != "fail" for check in checks)
    contract_passed = all(
        check.status != "fail" for check in checks if check.id.startswith("contract.")
    )
    support_tier = SupportTier.H1_CONTRACT_VERIFIED if contract_passed else SupportTier.H0_INDEXED
    observed_candidate = (
        SupportTier.H3_HARDWARE_READ_VERIFIED.value if stage == "read-only" and passed else None
    )
    blockers: list[str] = []
    if stage == "read-only" and passed:
        blockers.extend(
            [
                "independent physical observer attestation is required for H3",
                "canonical product status is never promoted by a local verification run",
            ]
        )
    elif stage == "read-only":
        blockers.append("one or more read-only verification checks failed")

    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    evidence_seed = json.dumps(
        {
            "pack": record.ref,
            "digest": record.manifest_digest,
            "stage": stage,
            "instance": configured.instance_id if configured else None,
            "generated_at": generated_at,
            "checks": [asdict(check) for check in checks],
        },
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    evidence_id = f"rpe_{hashlib.sha256(evidence_seed).hexdigest()[:24]}"
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else resolved_home / "robots" / "evidence" / f"{evidence_id}.json"
    )
    report = RobotPackVerificationReport(
        schema_version="rosclaw.robot_pack.evidence.v1",
        evidence_id=evidence_id,
        generated_at=generated_at,
        pack_ref=record.ref,
        manifest_digest=record.manifest_digest,
        stage=stage,
        passed=passed,
        support_tier=support_tier.value,
        observed_candidate_tier=observed_candidate,
        canonical_promotion=False,
        promotion_blockers=tuple(blockers),
        instance_id=configured.instance_id if configured else None,
        checks=tuple(checks),
        environment=_environment(),
        report_path=str(destination),
    )
    _write_report(destination, report)
    if contract_passed:
        store.record_verification(
            record.ref,
            evidence_id=evidence_id,
            support_tier=SupportTier.H1_CONTRACT_VERIFIED,
        )
    return report


def _contract_checks(
    record: InstalledRobotPack,
    manifest: RobotPackManifest,
    integrity: PackVerificationResult,
) -> list[VerificationCheckResult]:
    checks: list[VerificationCheckResult] = []
    integrity_ok = (
        integrity.ok
        and integrity.trusted
        and integrity.signature_status == "valid"
        and integrity.manifest_digest == record.manifest_digest
    )
    checks.append(
        VerificationCheckResult(
            id="contract.signed-pack-integrity",
            status="pass" if integrity_ok else "fail",
            message=(
                "schema, complete payload hashes, lock digest, and trusted Ed25519 signature match"
                if integrity_ok
                else "; ".join(integrity.errors)
                or "installed Pack digest or trust state does not match its lock"
            ),
            evidence=integrity.to_dict(),
        )
    )

    current_os = platform.system().lower()
    current_arch = _normalize_arch(platform.machine())
    python_version = Version(
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    compatible = (
        current_os in manifest.compatibility.os
        and current_arch in manifest.compatibility.arch
        and python_version in SpecifierSet(manifest.compatibility.python)
    )
    checks.append(
        VerificationCheckResult(
            id="contract.platform-compatibility",
            status="pass" if compatible else "fail",
            message=(
                f"host {current_os}/{current_arch} Python {python_version} is compatible"
                if compatible
                else f"host {current_os}/{current_arch} Python {python_version} is outside Pack compatibility"
            ),
        )
    )

    from rosclaw.runtime import RobotRegistry

    registry = RobotRegistry()
    missing_profiles = [
        variant.body_profile
        for variant in manifest.device.variants
        if registry.get(variant.body_profile) is None
    ]
    checks.append(
        VerificationCheckResult(
            id="contract.body-profile-availability",
            status="fail" if missing_profiles else "pass",
            message=(
                f"missing e-URDF profiles: {', '.join(missing_profiles)}"
                if missing_profiles
                else "all declared e-URDF body profiles are loadable"
            ),
        )
    )
    policy_safe = (
        manifest.safety.perception_only
        and manifest.safety.actuation == "forbidden"
        and manifest.safety.direct_driver_access == "forbidden"
        and manifest.safety.agent_southbound_access == "daemon_only"
        and all(capability.safety_class == "read_only" for capability in manifest.capabilities)
    )
    checks.append(
        VerificationCheckResult(
            id="contract.perception-only-policy",
            status="pass" if policy_safe else "fail",
            message=(
                "Pack is perception-only, forbids actuation/direct driver access, and requires rosclawd"
                if policy_safe
                else "Pack safety declarations do not preserve the read-only boundary"
            ),
        )
    )
    try:
        from rosclaw.robot_pack.runtime_loader import RealSenseCaptureExecutor

        loader_available = callable(RealSenseCaptureExecutor)
    except Exception:
        loader_available = False
    checks.append(
        VerificationCheckResult(
            id="contract.daemon-loader-contract",
            status="pass" if loader_available else "fail",
            message=(
                "daemon-side camera.capture_rgbd executor is available"
                if loader_available
                else "daemon-side Pack executor is unavailable"
            ),
        )
    )
    return checks


def _read_only_checks(
    *,
    identifier: str,
    record: InstalledRobotPack,
    manifest: RobotPackManifest,
    home: Path,
    discovery_report: DiscoveryReport | None,
    adapter_probe: AdapterProbe | None,
    receipt_path: Path | None,
) -> tuple[RobotInstanceConfig | None, list[VerificationCheckResult]]:
    checks: list[VerificationCheckResult] = []
    try:
        instance, config_path = load_robot_instance(identifier, home=home)
    except RobotInstanceError as exc:
        checks.append(
            VerificationCheckResult(
                id="read-only.configured-device-binding",
                status="fail",
                message=str(exc),
            )
        )
        return None, checks

    binding_ok = (
        instance.pack.ref == record.ref
        and instance.pack.manifest_digest == record.manifest_digest
        and instance.device.type == manifest.device.type
        and instance.device.serial != ""
    )
    try:
        effective = BodyResolver(workspace=home, body_id=instance.instance_id).get_effective_body()
        snapshot_ok = effective.effective_body_hash == instance.body_snapshot_hash
    except Exception:
        snapshot_ok = False
    checks.append(
        VerificationCheckResult(
            id="read-only.configured-device-binding",
            status="pass" if binding_ok and snapshot_ok else "fail",
            message=(
                "instance, Pack lock, serial identity, and immutable Body snapshot match"
                if binding_ok and snapshot_ok
                else "configured instance, Pack lock, or Body snapshot does not match"
            ),
            evidence={"config_path": str(config_path), "body_snapshot_match": snapshot_ok},
        )
    )

    tools: list[dict[str, Any]] = []
    probe_error = ""
    current_adapter = resolve_adapter_binding(manifest, home)
    adapter_binding_matches = bool(
        instance.adapter.server_name
        and instance.adapter.status == "installed"
        and current_adapter.status == "installed"
        and current_adapter.server_name == instance.adapter.server_name
    )
    if not adapter_binding_matches:
        probe_error = "required RealSense MCP adapter is not installed in this ROSCLAW_HOME"
    else:
        try:
            probe = adapter_probe or _probe_adapter_tools
            server_name = instance.adapter.server_name
            assert server_name is not None
            tools = probe(server_name, home)
        except Exception as exc:  # noqa: BLE001 - MCP process boundary
            probe_error = str(exc)
    tool_names = {
        str(tool.get("name")) for tool in tools if isinstance(tool, dict) and tool.get("name")
    }
    missing_groups = [
        requirement.id
        for requirement in manifest.adapter.tools
        if not any(candidate in tool_names for candidate in requirement.any_of)
    ]
    adapter_ok = not probe_error and not missing_groups
    checks.append(
        VerificationCheckResult(
            id="read-only.mcp-adapter-tool-contract",
            status="pass" if adapter_ok else "fail",
            message=(
                f"MCP adapter exposes required tool groups: {', '.join(sorted(tool_names))}"
                if adapter_ok
                else probe_error
                or f"MCP adapter is missing tool groups: {', '.join(missing_groups)}"
            ),
            evidence={"server_name": instance.adapter.server_name, "tools": sorted(tool_names)},
        )
    )

    discovery = discovery_report or discover_realsense_devices(manifest)
    matching = [device for device in discovery.devices if device.serial == instance.device.serial]
    live_variant = (
        match_device_variant(
            manifest,
            product_id=matching[0].product_id,
            model=matching[0].model,
        )
        if len(matching) == 1
        else None
    )
    identity_ok = bool(
        len(matching) == 1
        and matching[0].identity_complete
        and matching[0].vendor_id == instance.device.vendor_id
        and matching[0].product_id == instance.device.product_id
        and matching[0].stable_uri == instance.device.stable_uri
        and live_variant is not None
        and live_variant.body_profile == instance.body_profile
    )
    checks.append(
        VerificationCheckResult(
            id="read-only.live-device-identity",
            status="pass" if identity_ok else "fail",
            message=(
                "live device model, serial, firmware, USB speed, backend, and stable URI are complete"
                if identity_ok
                else "configured serial was not discovered with complete SDK identity"
            ),
            evidence=matching[0].to_dict() if matching else discovery.to_dict(),
        )
    )
    streams = matching[0].stream_profiles if matching else ()
    stream_names = {profile.stream.casefold() for profile in streams}
    streams_ok = "depth" in stream_names and bool(stream_names & {"color", "rgb"})
    checks.append(
        VerificationCheckResult(
            id="read-only.stream-profile-enumeration",
            status="pass" if streams_ok else "fail",
            message=(
                "live depth and color stream profiles were enumerated"
                if streams_ok
                else "live discovery did not report both depth and color stream profiles"
            ),
            evidence={"stream_profiles": [asdict(profile) for profile in streams]},
        )
    )

    receipt_ok, receipt_message, receipt_evidence = _validate_receipt(
        receipt_path,
        instance,
        home,
    )
    checks.append(
        VerificationCheckResult(
            id="read-only.rgbd-artifact-receipt",
            status="pass" if receipt_ok else "fail",
            message=receipt_message,
            evidence=receipt_evidence,
        )
    )
    return instance, checks


def _validate_receipt(
    path: Path | None,
    instance: RobotInstanceConfig,
    home: Path,
) -> tuple[bool, str, dict[str, Any] | None]:
    if path is None:
        return (
            False,
            "a canonical rosclawd RGB-D ExecutionReceipt is required; pass --receipt",
            None,
        )
    if path.is_symlink():
        return False, "receipt cannot be a symbolic link", {"path": str(path)}
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"receipt is unreadable: {exc}", {"path": str(path)}
    if isinstance(receipt, dict) and isinstance(receipt.get("receipt"), dict):
        receipt = receipt["receipt"]
    if not isinstance(receipt, dict):
        return False, "receipt must be a JSON object", {"path": str(path)}
    action_id = receipt.get("action_id")
    trace_id = receipt.get("trace_id")
    started_at = _parse_timestamp(receipt.get("started_at"))
    finished_at = _parse_timestamp(receipt.get("finished_at"))
    identity_contract_ok = (
        receipt.get("schema_version") == "rosclaw.receipt.v1"
        and isinstance(action_id, str)
        and bool(action_id)
        and isinstance(trace_id, str)
        and bool(trace_id)
        and receipt.get("body_id") == instance.instance_id
        and receipt.get("body_snapshot_hash") == instance.body_snapshot_hash
        and receipt.get("capability_id") == "camera.capture_rgbd"
        and receipt.get("execution_mode") == "REAL"
        and receipt.get("final_state") == "COMPLETED"
        and receipt.get("evidence_level") in {"PHYSICALLY_OBSERVED", "TASK_VERIFIED"}
        and receipt.get("verified") is True
        and receipt.get("trust_level") == "VERIFIED"
        and receipt.get("usable_for_real_execution") is True
    )
    lease = receipt.get("resource_lease")
    lease_acquired_at = (
        _parse_timestamp(lease.get("acquired_at")) if isinstance(lease, dict) else None
    )
    canonical_timing_ok = bool(
        started_at is not None
        and finished_at is not None
        and started_at <= finished_at
        and isinstance(lease, dict)
        and isinstance(lease.get("lease_id"), str)
        and bool(lease["lease_id"])
        and lease.get("resource_id") == instance.instance_id
        and lease.get("action_id") == action_id
        and lease.get("exclusive") is True
        and lease_acquired_at is not None
        and started_at - timedelta(seconds=5)
        <= lease_acquired_at
        <= finished_at + timedelta(seconds=5)
    )
    control_plane_ok = bool(
        isinstance(receipt.get("policy_decision"), dict)
        and receipt["policy_decision"].get("allowed") is True
        and isinstance(receipt.get("authorization_decision"), dict)
        and receipt["authorization_decision"].get("authorized") is True
        and isinstance(receipt.get("dispatch_result"), dict)
        and receipt["dispatch_result"].get("accepted") is True
        and isinstance(receipt.get("driver_ack"), dict)
        and receipt["driver_ack"].get("acknowledged") is True
        and isinstance(receipt.get("verification_result"), dict)
        and receipt["verification_result"].get("success") is True
    )
    transitions = receipt.get("transitions")
    transition_items = transitions if isinstance(transitions, list) else []
    transition_states = [
        item.get("state") for item in transition_items if isinstance(item, dict)
    ]
    transition_times = [
        _parse_timestamp(item.get("at")) for item in transition_items if isinstance(item, dict)
    ]
    required_transitions = [
        "DISPATCHED",
        "DRIVER_ACKNOWLEDGED",
        "PHYSICALLY_OBSERVED",
        "COMPLETED",
    ]
    transitions_ok = bool(
        started_at is not None
        and finished_at is not None
        and len(transition_times) == len(transition_items)
        and all(item is not None for item in transition_times)
        and _ordered_subsequence(required_transitions, transition_states)
        and _timestamps_are_ordered(transition_times)
        and all(
            started_at - timedelta(seconds=5)
            <= item
            <= finished_at + timedelta(seconds=5)
            for item in transition_times
            if item is not None
        )
    )
    observations = receipt.get("observations")
    capture = next(
        (
            item
            for item in observations or []
            if isinstance(item, dict) and item.get("kind") == "rgbd_capture"
        ),
        None,
    )
    if not identity_contract_ok or not isinstance(capture, dict):
        return (
            False,
            "receipt identity, mode, state, evidence, or RGB-D observation is invalid",
            {
                "path": str(path),
                "action_id": receipt.get("action_id"),
            },
        )
    if not canonical_timing_ok:
        return (
            False,
            "receipt action identity, exclusive resource lease, or timing is non-canonical",
            {
                "path": str(path),
                "action_id": action_id,
            },
        )
    if not control_plane_ok or not transitions_ok:
        return (
            False,
            "receipt control-plane decisions, ACK, verification, or transitions are incomplete",
            {
                "path": str(path),
                "action_id": receipt.get("action_id"),
            },
        )
    device = capture.get("device_identity", {})
    captured_at = _parse_timestamp(capture.get("captured_at"))
    if (
        not isinstance(device, dict)
        or device.get("serial") != instance.device.serial
        or str(device.get("model", "")).casefold() != instance.device.model.casefold()
        or device.get("stable_uri") != instance.device.stable_uri
        or captured_at is None
        or started_at is None
        or finished_at is None
        or captured_at < started_at - timedelta(seconds=5)
        or captured_at > finished_at + timedelta(seconds=5)
    ):
        return (
            False,
            "receipt device identity or capture timestamp does not match the instance",
            {
                "path": str(path),
                "action_id": receipt.get("action_id"),
            },
        )
    metrics = capture.get("metrics")
    if not (
        isinstance(metrics, dict)
        and _positive_int(metrics.get("width"))
        and _positive_int(metrics.get("height"))
        and metrics.get("aligned") is True
    ):
        return (
            False,
            "receipt does not contain aligned RGB-D dimensions",
            {
                "path": str(path),
                "action_id": receipt.get("action_id"),
            },
        )
    artifacts = capture.get("artifacts", {})
    hashes = capture.get("artifact_hashes", {})
    receipt_artifacts = receipt.get("artifacts", [])
    managed_parent = home / "artifacts"
    managed_path = managed_parent / "robot-packs"
    if managed_parent.is_symlink() or managed_path.is_symlink():
        return (
            False,
            "managed Robot Pack artifact root cannot contain a symbolic link",
            {
                "path": str(path),
                "action_id": receipt.get("action_id"),
            },
        )
    managed_root = managed_path.resolve()
    if managed_root != managed_path:
        return (
            False,
            "managed Robot Pack artifact root cannot contain a symbolic link",
            {
                "path": str(path),
                "action_id": receipt.get("action_id"),
            },
        )
    receipt_uri = path.resolve().as_uri()
    if (
        not isinstance(receipt_artifacts, list)
        or not all(isinstance(item, str) for item in receipt_artifacts)
        or receipt_uri not in receipt_artifacts
    ):
        return (
            False,
            "canonical receipt artifact URI is missing from the receipt",
            {
                "path": str(path),
                "action_id": action_id,
            },
        )
    for name in ("color", "depth"):
        uri = artifacts.get(name)
        expected = hashes.get(name)
        artifact_path = _file_uri_path(uri)
        if (
            artifact_path is None
            or not artifact_path.is_file()
            or not isinstance(expected, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", expected) is None
            or uri not in receipt_artifacts
        ):
            return (
                False,
                f"receipt {name} artifact or SHA-256 evidence is missing",
                {
                    "path": str(path),
                    "action_id": receipt.get("action_id"),
                },
            )
        try:
            artifact_path.relative_to(managed_root)
        except ValueError:
            return (
                False,
                f"receipt {name} artifact is outside the managed Robot Pack artifact root",
                {
                    "path": str(path),
                    "action_id": receipt.get("action_id"),
                },
            )
        modified_at = datetime.fromtimestamp(artifact_path.stat().st_mtime, UTC)
        if (
            started_at is None
            or finished_at is None
            or modified_at < started_at - timedelta(seconds=5)
            or modified_at > finished_at + timedelta(seconds=5)
        ):
            return (
                False,
                f"receipt {name} artifact timestamp is outside the execution window",
                {
                    "path": str(path),
                    "action_id": action_id,
                },
            )
        actual = f"sha256:{_hash_file(artifact_path)}"
        if actual != expected:
            return (
                False,
                f"receipt {name} artifact hash does not match",
                {
                    "path": str(path),
                    "action_id": receipt.get("action_id"),
                },
            )
    return (
        True,
        "real RGB-D artifacts, hashes, device identity, and rosclawd receipt match",
        {
            "path": str(path),
            "action_id": receipt.get("action_id"),
            "evidence_level": receipt.get("evidence_level"),
        },
    )


def _probe_adapter_tools(server_name: str, home: Path) -> list[dict[str, Any]]:
    from rosclaw.mcp.onboarding.stdio_client import list_server_tools

    return list_server_tools(server_name, home=home, timeout=20.0)


def _resolve_record(store: RobotPackStore, identifier: str) -> InstalledRobotPack:
    matches = [
        record
        for record in store.list_installed()
        if identifier
        in {
            record.ref,
            record.name,
            f"{record.namespace}/{record.name}",
            f"{record.namespace}/{record.name}@{record.version}",
        }
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RobotPackStoreError(f"Installed Robot Pack reference is ambiguous: {identifier}")
    try:
        return store.resolve_installed(identifier)[0]
    except RobotPackStoreError:
        raise RobotPackStoreError(f"Robot Pack is not installed: {identifier}") from None


def _environment() -> dict[str, Any]:
    from rosclaw import __version__

    return {
        "os": platform.system().lower(),
        "arch": _normalize_arch(platform.machine()),
        "python": platform.python_version(),
        "rosclaw_version": __version__,
        "commit_sha": os.environ.get("GITHUB_SHA"),
    }


def _normalize_arch(value: str) -> str:
    normalized = value.casefold()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(normalized, normalized)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _timestamps_are_ordered(values: list[datetime | None]) -> bool:
    timestamps = [value for value in values if value is not None]
    return len(timestamps) == len(values) and all(
        earlier <= later for earlier, later in zip(timestamps, timestamps[1:], strict=False)
    )


def _ordered_subsequence(required: list[str], observed: list[Any]) -> bool:
    position = 0
    for state in observed:
        if position < len(required) and state == required[position]:
            position += 1
    return position == len(required)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_uri_path(value: Any) -> Path | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return None
    unresolved = Path(unquote(parsed.path))
    if unresolved.is_symlink():
        return None
    return unresolved.resolve()


def _write_report(path: Path, report: RobotPackVerificationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "RobotPackVerificationReport",
    "VerificationCheckResult",
    "verify_installed_robot_pack",
]
