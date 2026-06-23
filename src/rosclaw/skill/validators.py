"""Validation logic for ROSClaw skill packages."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

from rosclaw.skill.hash import sha256_file
from rosclaw.skill.models import (
    DarwinEvalYaml,
    DojoYaml,
    EurdfCompatYaml,
    LineageYaml,
    ProvidersYaml,
    SafetyYaml,
    SkillPackage,
    SkillYaml,
    ValidationReport,
)

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_skill_yaml(path: Path) -> ValidationReport:
    report = ValidationReport()
    try:
        raw = _load_yaml(path)
        SkillYaml.model_validate(raw)
        report.checks["skill_schema"] = True
    except Exception as exc:  # noqa: BLE001
        report.add_error(f"skill.yaml: {exc}")
    return report


def validate_providers_yaml(path: Path) -> ValidationReport:
    report = ValidationReport()
    try:
        raw = _load_yaml(path)
        model = ProvidersYaml.model_validate(raw)
        report.checks["providers_schema"] = True
        # Cross-check required capabilities map to providers.
        for cap, route in model.required_capabilities.items():
            if route.primary not in model.providers:
                report.add_warning(f"Capability {cap} primary provider {route.primary!r} not declared")
            for fb in route.fallback:
                if fb not in model.providers:
                    report.add_warning(f"Capability {cap} fallback provider {fb!r} not declared")
    except Exception as exc:  # noqa: BLE001
        report.add_error(f"providers.yaml: {exc}")
    return report


def validate_eurdf_compat_yaml(path: Path) -> ValidationReport:
    report = ValidationReport()
    try:
        raw = _load_yaml(path)
        model = EurdfCompatYaml.model_validate(raw)
        report.checks["eurdf_compat_schema"] = True
        if not model.compatible_robots:
            report.add_error("e-urdf-compat.yaml must declare at least one compatible robot")
        for robot in model.compatible_robots:
            if not robot.eurdf_profile:
                report.add_error(f"compatible robot {robot.robot!r} must declare eurdf_profile")
            if not robot.required_sensors:
                report.add_warning(f"compatible robot {robot.robot!r} has no required_sensors")
            if not robot.action_interfaces:
                report.add_error(f"compatible robot {robot.robot!r} must declare action_interfaces")
    except Exception as exc:  # noqa: BLE001
        report.add_error(f"e-urdf-compat.yaml: {exc}")
    return report


def validate_safety_yaml(path: Path) -> ValidationReport:
    report = ValidationReport()
    try:
        raw = _load_yaml(path)
        model = SafetyYaml.model_validate(raw)
        report.checks["safety_schema"] = True
        raw_flat = yaml.safe_dump(raw).lower()
        if "disable_sandbox" in raw_flat:
            report.add_error("safety.yaml must not contain disable_sandbox")
        if model.robot.require_estop_ready is False and "real_robot_guarded" in model.runtime_mode.allowed:
            report.add_warning("real_robot_guarded allowed but robot.require_estop_ready is false")
        if not model.sandbox.required_checks:
            report.add_warning("safety.yaml sandbox.required_checks is empty")
    except Exception as exc:  # noqa: BLE001
        report.add_error(f"safety.yaml: {exc}")
    return report


def validate_dojo_yaml(path: Path) -> ValidationReport:
    report = ValidationReport()
    try:
        raw = _load_yaml(path)
        DojoYaml.model_validate(raw)
        report.checks["dojo_schema"] = True
    except Exception as exc:  # noqa: BLE001
        report.add_error(f"dojo.yaml: {exc}")
    return report


def validate_darwin_eval_yaml(path: Path) -> ValidationReport:
    report = ValidationReport()
    try:
        raw = _load_yaml(path)
        model = DarwinEvalYaml.model_validate(raw)
        report.checks["darwin_eval_schema"] = True
        if not model.metrics:
            report.add_error("darwin_eval.yaml must define metrics")
        if "success_rate" not in model.metrics:
            report.add_warning("darwin_eval.yaml missing success_rate metric")
        if "no_fall_rate" not in model.metrics:
            report.add_warning("darwin_eval.yaml missing no_fall_rate metric")
    except Exception as exc:  # noqa: BLE001
        report.add_error(f"darwin_eval.yaml: {exc}")
    return report


def validate_lineage_yaml(path: Path) -> ValidationReport:
    report = ValidationReport()
    try:
        raw = _load_yaml(path)
        LineageYaml.model_validate(raw)
        report.checks["lineage_schema"] = True
    except Exception as exc:  # noqa: BLE001
        report.add_error(f"lineage.yaml: {exc}")
    return report


# ---------------------------------------------------------------------------
# File and content validators
# ---------------------------------------------------------------------------


REQUIRED_FILES = [
    "skill.yaml",
    "README.md",
    "behavior_tree.xml",
    "providers.yaml",
    "e-urdf-compat.yaml",
    "safety.yaml",
    "dojo.yaml",
    "darwin_eval.yaml",
    "tests/",
    "evidence/",
    "lineage.yaml",
    "CHANGELOG.md",
    ".rosclaw/lock.yaml",
]

README_REQUIRED_SECTIONS = [
    "what it does",
    "supported robots",
    "required sensors",
    "required providers",
    "safety constraints",
    "how to run",
    "evaluation evidence",
    "version history",
    "known limitations",
]


def validate_file_existence(root: Path) -> ValidationReport:
    report = ValidationReport()
    for req in REQUIRED_FILES:
        path = root / req
        if not path.exists():
            report.add_error(f"Missing required file/dir: {req}")
    report.checks["required_files"] = report.ok
    return report


def validate_readme(root: Path) -> ValidationReport:
    report = ValidationReport()
    readme = root / "README.md"
    if not readme.exists():
        report.add_error("README.md is missing")
        return report
    text = readme.read_text(encoding="utf-8").lower()
    missing = [s for s in README_REQUIRED_SECTIONS if s not in text]
    for m in missing:
        report.add_warning(f"README.md missing section: {m}")
    report.checks["readme_sections"] = not missing
    return report


def validate_behavior_tree(path: Path) -> ValidationReport:
    report = ValidationReport()
    if not path.exists():
        report.add_error(f"behavior tree missing: {path}")
        return report
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        report.add_error(f"behavior_tree.xml parse error: {exc}")
        return report

    root = tree.getroot()
    tags = {elem.tag for elem in root.iter()}
    required_nodes = {"SandboxValidate", "VerifyOutcome", "Fallback"}
    missing = required_nodes - tags
    if missing:
        report.add_error(f"behavior_tree.xml missing required nodes: {sorted(missing)}")
    report.checks["behavior_tree_lint"] = not missing
    return report


def validate_package_integrity(pkg: SkillPackage) -> ValidationReport:
    report = ValidationReport()
    hashes_path = pkg.root / ".rosclaw" / "hashes.json"
    if not hashes_path.exists():
        report.add_warning(".rosclaw/hashes.json missing; run `rosclaw skill package` first")
        report.checks["package_integrity"] = False
        return report
    try:
        import json

        stored = json.loads(hashes_path.read_text(encoding="utf-8"))
        mismatches = []
        for rel, expected in stored.get("files", {}).items():
            path = pkg.root / rel
            if not path.exists():
                mismatches.append(f"{rel}: missing")
                continue
            actual = f"sha256:{sha256_file(path)}"
            if actual != expected:
                mismatches.append(f"{rel}: hash mismatch")
        if mismatches:
            for m in mismatches:
                report.add_error(f"package integrity: {m}")
        else:
            report.checks["package_integrity"] = True
    except Exception as exc:  # noqa: BLE001
        report.add_error(f"package integrity check failed: {exc}")
    return report


# ---------------------------------------------------------------------------
# Aggregate validation
# ---------------------------------------------------------------------------


def validate_package(pkg: SkillPackage) -> ValidationReport:
    report = validate_file_existence(pkg.root)
    report.merge(validate_readme(pkg.root))
    report.merge(validate_behavior_tree(pkg.root / "behavior_tree.xml"))
    report.merge(validate_skill_yaml(pkg.root / "skill.yaml"))
    report.merge(validate_providers_yaml(pkg.root / "providers.yaml"))
    report.merge(validate_eurdf_compat_yaml(pkg.root / "e-urdf-compat.yaml"))
    report.merge(validate_safety_yaml(pkg.root / "safety.yaml"))
    report.merge(validate_dojo_yaml(pkg.root / "dojo.yaml"))
    report.merge(validate_darwin_eval_yaml(pkg.root / "darwin_eval.yaml"))
    report.merge(validate_lineage_yaml(pkg.root / "lineage.yaml"))
    report.merge(validate_package_integrity(pkg))
    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping, got {type(data).__name__}")
    return data
