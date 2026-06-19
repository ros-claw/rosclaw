"""License and data-rights policy checks for ROSClaw Hub assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rosclaw.hub.schema import AssetManifest, load_manifest

# SPDX identifiers considered pre-approved for automatic acceptance.
# This is intentionally conservative and can be expanded via policy config.
APPROVED_SPDX_LICENSES: set[str] = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
    "ISC",
    "Zlib",
    "MPL-2.0",
    "Unlicense",
    "CC0-1.0",
}


@dataclass
class LicenseCheckResult:
    """Result of checking an asset's license and data rights."""

    accepted: bool = True
    requires_acceptance: bool = False
    issues: list[str] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        self.accepted = False
        self.issues.append(reason)


def check_license(
    manifest: AssetManifest | str | Path,
    *,
    accept_license: bool = False,
    approved_spdx: set[str] | None = None,
    asset_dir: Path | None = None,
) -> LicenseCheckResult:
    """Check whether an asset's license and data rights are acceptable.

    Args:
        manifest: Loaded manifest, path, or path string.
        accept_license: Whether the caller has explicitly accepted a license
            that requires manual acceptance.
        approved_spdx: Override set of pre-approved SPDX identifiers.
        asset_dir: Directory containing the manifest and license file. If
            omitted, license file existence is not checked.

    Returns:
        :class:`LicenseCheckResult` with acceptance status and issues.
    """
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        manifest = load_manifest(manifest_path)
        if asset_dir is None:
            asset_dir = manifest_path.parent

    result = LicenseCheckResult()
    license_section = manifest.license
    data_rights = manifest.data_rights
    approved = APPROVED_SPDX_LICENSES if approved_spdx is None else approved_spdx

    spdx = license_section.get("spdx")
    if not spdx:
        result.reject("License section missing SPDX identifier")
    elif spdx not in approved:
        result.requires_acceptance = True
        result.reject(
            f"License {spdx} is not on the automatic-approval list and requires explicit acceptance"
        )

    # License file must exist next to the manifest.
    license_file = license_section.get("license_file")
    if license_file and asset_dir is not None and not (asset_dir / license_file).exists():
        result.reject(f"License file missing: {license_file}")

    # Commercial use restrictions.
    if license_section.get("commercial_use") is False and not result.requires_acceptance:
        result.reject("License prohibits commercial use")

    # Redistribution restrictions.
    if license_section.get("redistribution") is False and not result.requires_acceptance:
        result.reject("License prohibits redistribution")

    # Export control flags.
    export_control = license_section.get("export_control", "none")
    if export_control not in (None, "none"):
        result.requires_acceptance = True
        if not accept_license:
            result.reject(
                f"License has export control classification '{export_control}' and requires explicit acceptance"
            )

    # Data rights restrictions.
    allowed_usage = set(data_rights.get("allowed_usage", []))
    if allowed_usage and "commercial" not in allowed_usage:
        result.requires_acceptance = True
        if not accept_license:
            result.reject("Data rights restrict usage categories; explicit acceptance required")

    # Personal data requires acceptance.
    if data_rights.get("contains_personal_data", False):
        result.requires_acceptance = True
        if not accept_license:
            result.reject("Asset contains personal data; explicit acceptance required")

    # If explicit acceptance was given, clear rejection issues caused only by
    # the acceptance requirement, but keep other policy violations.
    if accept_license and result.requires_acceptance:
        result.accepted = True
        # Re-add any non-acceptance issues.
        non_acceptance_issues = [
            issue
            for issue in result.issues
            if "requires explicit acceptance" not in issue
            and "requires manual acceptance" not in issue
        ]
        result.issues = non_acceptance_issues

    return result
