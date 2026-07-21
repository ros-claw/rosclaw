"""Load and validate the canonical product capability status."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, cast

import yaml

SCHEMA_VERSION = "rosclaw.product_status.v1"

ALLOWED_STATES = {
    "developer_observed",
    "fixture_available",
    "fixture_verified",
    "not_applicable",
    "not_run",
    "not_verified",
    "verified",
}

ALLOWED_CLAIMS = {
    "component_system_verified",
    "component_verified",
    "developer_observed_revalidation_pending",
    "experimental",
    "fixture_only",
    "not_verified",
    "revalidation_pending",
    "simulation_verified",
}

VERIFIED_CLAIMS = {
    "component_system_verified",
    "component_verified",
    "simulation_verified",
}

TIER_REQUIREMENTS = {
    "H2_SIMULATION_VERIFIED": ("simulation", "verified"),
    "H3_HARDWARE_READ_VERIFIED": ("hardware_read", "verified"),
    "H4_HARDWARE_ACTUATION_VERIFIED": ("hardware_actuation", "verified"),
    "H5_AGENT_BLACKBOX_VERIFIED": ("agent_blackbox", "verified"),
}


class ProductStatusError(ValueError):
    """The product status source is missing or overclaims available evidence."""


def product_status_path() -> Path:
    """Return the source-tree path when available."""

    return Path(__file__).with_name("status.yaml")


def load_product_status(path: Path | None = None) -> dict[str, Any]:
    """Load and validate product status from a path or package resource."""

    if path is None:
        text = (
            resources.files("rosclaw.product").joinpath("status.yaml").read_text(encoding="utf-8")
        )
    else:
        text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ProductStatusError("Product status must be a YAML mapping.")
    status = cast(dict[str, Any], raw)
    errors = validate_product_status(status)
    if errors:
        raise ProductStatusError("\n".join(errors))
    return status


def validate_product_status(
    status: dict[str, Any],
    *,
    repository_root: Path | None = None,
    package_version: str | None = None,
) -> list[str]:
    """Return truthfulness and schema errors without mutating ``status``."""

    errors: list[str] = []
    if status.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    release = _mapping(status.get("release"), "release", errors)
    version = str(release.get("version", ""))
    if package_version is None:
        from rosclaw import __version__

        package_version = __version__
    if version != package_version:
        errors.append(
            f"release.version {version!r} does not match package version {package_version!r}"
        )

    golden_paths = _mapping(status.get("golden_paths"), "golden_paths", errors)
    components = _mapping(status.get("components"), "components", errors)
    if not golden_paths:
        errors.append("golden_paths must not be empty")

    for key, raw_entry in golden_paths.items():
        name = f"golden_paths.{key}"
        entry = _mapping(raw_entry, name, errors)
        _validate_entry(name, entry, errors, repository_root=repository_root, is_golden=True)

    for key, raw_entry in components.items():
        name = f"components.{key}"
        entry = _mapping(raw_entry, name, errors)
        _validate_entry(name, entry, errors, repository_root=repository_root, is_golden=False)

    matrix = status.get("readme_matrix")
    if not isinstance(matrix, list) or not matrix:
        errors.append("readme_matrix must be a non-empty list")
    else:
        for reference in matrix:
            if not isinstance(reference, str) or _resolve_reference(status, reference) is None:
                errors.append(f"readme_matrix contains unknown reference {reference!r}")

    return errors


def iter_matrix_entries(status: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return capability-matrix entries in the canonical presentation order."""

    entries: list[tuple[str, dict[str, Any]]] = []
    for reference in cast(list[Any], status.get("readme_matrix", [])):
        if not isinstance(reference, str):
            continue
        value = _resolve_reference(status, reference)
        if isinstance(value, dict):
            entries.append((reference, cast(dict[str, Any], value)))
    return entries


def _validate_entry(
    name: str,
    entry: dict[str, Any],
    errors: list[str],
    *,
    repository_root: Path | None,
    is_golden: bool,
) -> None:
    display = _mapping(entry.get("display"), f"{name}.display", errors)
    for language in ("en", "zh"):
        if not str(display.get(language, "")).strip():
            errors.append(f"{name}.display.{language} is required")

    claim = str(entry.get("claim", ""))
    if claim not in ALLOWED_CLAIMS:
        errors.append(f"{name}.claim has unsupported value {claim!r}")

    evidence = entry.get("evidence", [])
    if not isinstance(evidence, list):
        errors.append(f"{name}.evidence must be a list")
        evidence = []
    evidence_records = [item for item in evidence if isinstance(item, dict)]
    if len(evidence_records) != len(evidence):
        errors.append(f"{name}.evidence entries must be mappings")

    if claim in VERIFIED_CLAIMS and not evidence_records:
        errors.append(f"{name} claims {claim} without an Evidence ID")

    evidence_ids: set[str] = set()
    for index, item in enumerate(evidence_records):
        prefix = f"{name}.evidence[{index}]"
        evidence_id = str(item.get("id", "")).strip()
        if not evidence_id:
            errors.append(f"{prefix}.id is required")
        elif evidence_id in evidence_ids:
            errors.append(f"{name} contains duplicate Evidence ID {evidence_id!r}")
        evidence_ids.add(evidence_id)
        if item.get("fixture") is True and item.get("observation_scope") == "physical_hardware":
            errors.append(f"{prefix} cannot be both fixture and physical_hardware")
        raw_path = item.get("path")
        if (
            repository_root is not None
            and isinstance(raw_path, str)
            and not (repository_root / raw_path).is_file()
        ):
            errors.append(f"{prefix}.path does not exist: {raw_path}")

    if not is_golden:
        return

    modes = _mapping(entry.get("modes"), f"{name}.modes", errors)
    dimensions = _mapping(entry.get("dimensions"), f"{name}.dimensions", errors)
    for section_name, values in (("modes", modes), ("dimensions", dimensions)):
        for key, raw_state in values.items():
            state = str(raw_state)
            if state not in ALLOWED_STATES:
                errors.append(f"{name}.{section_name}.{key} has unsupported state {state!r}")

    if modes.get("fixture") == "verified":
        errors.append(f"{name}.modes.fixture must use fixture_verified, never verified")
    if modes.get("simulation") == "verified" or dimensions.get("simulation") == "verified":
        _require_simulation_evidence(name, evidence_records, errors)
    if modes.get("real") == "verified" or any(
        dimensions.get(field) == "verified" for field in ("hardware_read", "hardware_actuation")
    ):
        _require_physical_evidence(name, evidence_records, errors)
    if dimensions.get("agent_blackbox") == "verified":
        _require_agent_blackbox_evidence(name, evidence_records, errors)

    if entry.get("agent_ready") is True and dimensions.get("agent_blackbox") != "verified":
        errors.append(f"{name} cannot be agent_ready without verified agent_blackbox evidence")

    tier = str(entry.get("support_tier", ""))
    if tier not in {
        "H0_INDEXED",
        "H1_CONTRACT_VERIFIED",
        "H6_REFERENCE_SUPPORTED",
        *TIER_REQUIREMENTS,
    }:
        errors.append(f"{name}.support_tier has unsupported value {tier!r}")
    requirement = TIER_REQUIREMENTS.get(tier)
    if requirement is not None:
        dimension, required_state = requirement
        if dimensions.get(dimension) != required_state:
            errors.append(
                f"{name}.support_tier {tier} requires dimensions.{dimension}={required_state}"
            )
    if tier == "H6_REFERENCE_SUPPORTED" and (
        dimensions.get("agent_blackbox") != "verified"
        or entry.get("reference_supported") is not True
        or not any(
            dimensions.get(dimension) == "verified"
            for dimension in ("simulation", "hardware_read", "hardware_actuation")
        )
    ):
        errors.append(
            f"{name}.support_tier H6_REFERENCE_SUPPORTED requires verified Agent black-box "
            "evidence, a verified execution dimension, and reference_supported=true"
        )

    verified_states = [
        state for state in [*modes.values(), *dimensions.values()] if str(state) == "verified"
    ]
    if verified_states and not any(
        str(item.get("id", "")).strip() and item.get("fixture") is not True
        for item in evidence_records
    ):
        errors.append(f"{name} uses verified without non-fixture evidence")


def _require_physical_evidence(
    name: str,
    evidence: list[dict[str, Any]],
    errors: list[str],
) -> None:
    matching = [
        item
        for item in evidence
        if item.get("observation_scope") == "physical_hardware"
        and item.get("fixture") is not True
        and str(item.get("id", "")).strip()
    ]
    if not matching:
        errors.append(f"{name} claims hardware verified without physical observation evidence")
    elif not any(item.get("independent") is True for item in matching):
        errors.append(
            f"{name} claims hardware verified without independent physical observation evidence"
        )


def _require_simulation_evidence(
    name: str,
    evidence: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if not any(
        item.get("observation_scope") == "physics_simulation"
        and item.get("fixture") is not True
        and str(item.get("id", "")).strip()
        for item in evidence
    ):
        errors.append(f"{name} claims simulation verified without physics simulation evidence")


def _require_agent_blackbox_evidence(
    name: str,
    evidence: list[dict[str, Any]],
    errors: list[str],
) -> None:
    matching = [
        item
        for item in evidence
        if item.get("observation_scope") in {"external_agent_simulation", "external_agent_hardware"}
        and item.get("fixture") is not True
        and str(item.get("id", "")).strip()
    ]
    if not matching:
        errors.append(f"{name} claims agent blackbox verified without external Agent evidence")
    elif not any(item.get("independent") is True for item in matching):
        errors.append(
            f"{name} claims agent blackbox verified without independent external Agent evidence"
        )


def _mapping(value: Any, name: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{name} must be a mapping")
        return {}
    return cast(dict[str, Any], value)


def _resolve_reference(status: dict[str, Any], reference: str) -> Any:
    value: Any = status
    for part in reference.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


__all__ = [
    "ALLOWED_CLAIMS",
    "ALLOWED_STATES",
    "ProductStatusError",
    "SCHEMA_VERSION",
    "iter_matrix_entries",
    "load_product_status",
    "product_status_path",
    "validate_product_status",
]
