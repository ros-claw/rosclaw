"""Hardware MCP body/e-URDF binding.

Ensures the required e-URDF profile is registered and writes the binding
fragment into ``body.yaml`` using only the paths declared in the manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rosclaw.body.resolver import BodyNotLinkedError, BodyResolver
from rosclaw.body.validators import apply_nested_update
from rosclaw.mcp.onboarding.errors import (
    BindingError,
    BodyNotLinkedError as OnboardingBodyNotLinkedError,
    EurdfHashMismatchError,
    EurdfProfileMissingError,
)
from rosclaw.mcp.onboarding.schema import BodyBindingTemplate, EurdfBinding, McpManifest
from rosclaw.runtime.eurdf_loader import RobotRegistry


@dataclass
class BindingResult:
    """Result of applying a body binding."""

    binding_key: str
    body_yaml_path: Path
    eurdf_profile: str | None = None
    eurdf_hash: str | None = None
    patched_paths: list[str] = None  # type: ignore[assignment]
    missing_required_fields: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.patched_paths is None:
            self.patched_paths = []
        if self.missing_required_fields is None:
            self.missing_required_fields = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_key": self.binding_key,
            "body_yaml_path": str(self.body_yaml_path),
            "eurdf_profile": self.eurdf_profile,
            "eurdf_hash": self.eurdf_hash,
            "patched_paths": list(self.patched_paths),
            "missing_required_fields": list(self.missing_required_fields),
        }


def _get_nested(data: dict[str, Any], path: str) -> Any:
    """Return a nested value or None if the path does not exist."""
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _build_body_patch(binding: BodyBindingTemplate) -> dict[str, Any]:
    """Translate a binding template into body.yaml patch entries.

    ``write_paths`` maps a logical alias to a dotted target path. The value
    is taken from the template by following the target path first, then the
    alias, so simple manifests do not need to duplicate keys.
    """
    patch: dict[str, Any] = {}
    for alias, target_path in (binding.write_paths or {}).items():
        value = _get_nested(binding.template or {}, target_path)
        if value is None:
            value = _get_nested(binding.template or {}, alias)
        if value is None and binding.template:
            value = binding.template.get(alias)
        if value is not None:
            patch[target_path] = value
    # If the template has top-level keys not covered by write_paths, merge them
    # directly so long as they are not already a prefix of a targeted dotted path.
    targeted = set((binding.write_paths or {}).values())
    targeted_prefixes = {t.split(".", 1)[0] for t in targeted}
    for key, value in (binding.template or {}).items():
        if key in targeted or key in (binding.write_paths or {}):
            continue
        if key in targeted_prefixes:
            continue
        patch[key] = value
    return patch


def _compute_profile_hash(profile: Any) -> str:
    """Compute a stable SHA-256 hash for an e-URDF profile."""
    if hasattr(profile, "to_dict"):
        data = profile.to_dict()
    else:
        data = dict(profile)
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class BodyBindingManager:
    """Bind a Hardware MCP manifest to the current body instance."""

    def __init__(self, workspace: Path | None = None) -> None:
        self.resolver = BodyResolver(workspace=workspace)
        self.registry = RobotRegistry()

    def ensure_eurdf(self, eurdf: EurdfBinding | None, dry_run: bool = False) -> tuple[str | None, str | None]:
        """Ensure the required e-URDF profile is installed.

        Returns:
            (profile_id, profile_hash) or (None, None) if no profile is required.

        Raises:
            EurdfProfileMissingError: if a required profile cannot be installed.
        """
        if not eurdf or not eurdf.profiles:
            return None, None

        default_profile = eurdf.default_profile
        for profile_ref in eurdf.profiles:
            profile_id = profile_ref.id
            required = profile_ref.required
            profile = self.registry.get(profile_id)
            if profile is None:
                if dry_run:
                    # In dry-run we cannot install, but we can report it.
                    if required:
                        raise EurdfProfileMissingError(
                            f"e-URDF profile '{profile_id}' is required but not installed "
                            "(dry-run cannot install new profiles)"
                        )
                    continue
                if not required:
                    # Optional profiles are not auto-installed.
                    continue
                try:
                    self.registry.install(profile_id)
                    profile = self.registry.get(profile_id)
                except FileNotFoundError as exc:
                    raise EurdfProfileMissingError(
                        f"Required e-URDF profile '{profile_id}' not found: {exc}"
                    ) from exc

            if profile_id == default_profile and profile is not None:
                return profile_id, _compute_profile_hash(profile)

        # If we reach here, no default profile was matched; use the first available.
        for profile_ref in eurdf.profiles:
            profile = self.registry.get(profile_ref.id)
            if profile is not None:
                return profile_ref.id, _compute_profile_hash(profile)

        return None, None

    def check_required_fields(self, required_fields: list[str]) -> list[str]:
        """Return required fields that are missing from body.yaml."""
        if not self.resolver.body_yaml_path.exists():
            raise OnboardingBodyNotLinkedError(
                f"No body linked at {self.resolver.body_yaml_path}"
            )
        with open(self.resolver.body_yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        missing: list[str] = []
        for path in required_fields:
            value = _get_nested(data, path)
            if value is None or value == "":
                missing.append(path)
        return missing

    def apply_binding(
        self,
        manifest: McpManifest,
        dry_run: bool = False,
    ) -> BindingResult:
        """Apply the body binding declared by ``manifest``.

        Raises:
            BodyNotLinkedError: if body.yaml is not present.
            EurdfProfileMissingError: if a required e-URDF profile is missing.
        """
        binding = manifest.body_binding
        if binding is None:
            return BindingResult(
                binding_key=manifest.server_name,
                body_yaml_path=self.resolver.body_yaml_path,
            )

        eurdf_profile, eurdf_hash = self.ensure_eurdf(manifest.eurdf, dry_run=dry_run)

        patch = _build_body_patch(binding)
        patched_paths = sorted(patch.keys())

        if dry_run:
            return BindingResult(
                binding_key=binding.binding_key,
                body_yaml_path=self.resolver.body_yaml_path,
                eurdf_profile=eurdf_profile,
                eurdf_hash=eurdf_hash,
                patched_paths=patched_paths,
                missing_required_fields=[],
            )

        if not self.resolver.is_linked():
            raise OnboardingBodyNotLinkedError(
                f"No body linked at {self.resolver.body_yaml_path}"
            )

        missing = self.check_required_fields(binding.required_fields or [])
        if missing:
            # Do not block installation; record missing fields for health/reporting.
            pass

        # Backup and mutate body.yaml atomically via BodyResolver.
        try:
            self.resolver.update_body_yaml(patch)
        except BodyNotLinkedError as exc:
            raise OnboardingBodyNotLinkedError(str(exc)) from exc

        return BindingResult(
            binding_key=binding.binding_key,
            body_yaml_path=self.resolver.body_yaml_path,
            eurdf_profile=eurdf_profile,
            eurdf_hash=eurdf_hash,
            patched_paths=patched_paths,
            missing_required_fields=missing,
        )
