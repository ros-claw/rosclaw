"""Truthfulness contracts for product capability claims."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from rosclaw.product.readme import extract_readme_matrix, render_readme_matrix
from rosclaw.product.status import load_product_status, validate_product_status

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_status_is_valid_and_references_existing_evidence() -> None:
    status = load_product_status()
    assert validate_product_status(status, repository_root=REPOSITORY_ROOT) == []


def test_verified_claim_requires_evidence_id() -> None:
    status = deepcopy(load_product_status())
    status["golden_paths"]["ur5e_reach"]["evidence"] = []

    errors = validate_product_status(status)

    assert any("without an Evidence ID" in error for error in errors)
    assert any("without non-fixture evidence" in error for error in errors)


def test_hardware_verified_requires_physical_observation() -> None:
    status = deepcopy(load_product_status())
    path = status["golden_paths"]["realsense_inspect"]
    path["dimensions"]["hardware_read"] = "verified"
    path["support_tier"] = "H3_HARDWARE_READ_VERIFIED"

    errors = validate_product_status(status)

    assert any("without physical observation evidence" in error for error in errors)


def test_hardware_verified_requires_independent_physical_observation() -> None:
    status = deepcopy(load_product_status())
    path = status["golden_paths"]["rh56_single_step"]
    path["dimensions"]["hardware_actuation"] = "verified"
    path["support_tier"] = "H4_HARDWARE_ACTUATION_VERIFIED"

    errors = validate_product_status(status)

    assert any(
        "without independent physical observation evidence" in error for error in errors
    )


def test_simulation_verified_requires_physics_evidence() -> None:
    status = deepcopy(load_product_status())
    status["golden_paths"]["ur5e_reach"]["evidence"][0][
        "observation_scope"
    ] = "component"

    errors = validate_product_status(status)

    assert any("without physics simulation evidence" in error for error in errors)


def test_agent_blackbox_verified_requires_independent_external_agent() -> None:
    status = deepcopy(load_product_status())
    path = status["golden_paths"]["ur5e_reach"]
    path["dimensions"]["agent_blackbox"] = "verified"
    path["support_tier"] = "H5_AGENT_BLACKBOX_VERIFIED"
    path["agent_ready"] = True

    errors = validate_product_status(status)

    assert any(
        "without independent external Agent evidence" in error for error in errors
    )


def test_agent_ready_requires_blackbox_verification() -> None:
    status = deepcopy(load_product_status())
    status["golden_paths"]["rh56_single_step"]["agent_ready"] = True

    errors = validate_product_status(status)

    assert any("cannot be agent_ready" in error for error in errors)


def test_fixture_cannot_be_marked_verified() -> None:
    status = deepcopy(load_product_status())
    status["golden_paths"]["ur5e_reach"]["modes"]["fixture"] = "verified"

    errors = validate_product_status(status)

    assert any("must use fixture_verified" in error for error in errors)


def test_readme_capability_matrices_match_status_source() -> None:
    status = load_product_status()
    for filename, language in (("README.md", "en"), ("README.zh.md", "zh")):
        text = (REPOSITORY_ROOT / filename).read_text(encoding="utf-8")
        assert extract_readme_matrix(text) == render_readme_matrix(status, language)
