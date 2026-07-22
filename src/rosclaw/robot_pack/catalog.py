"""Built-in Robot Pack catalog and reference resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rosclaw.robot_pack.schema import RobotPackManifest
from rosclaw.robot_pack.verifier import PackVerificationResult, verify_robot_pack


class RobotPackNotFoundError(LookupError):
    """Raised when a Robot Pack source cannot be resolved unambiguously."""


@dataclass(frozen=True)
class RobotPackCatalogEntry:
    root: Path
    manifest: RobotPackManifest
    verification: PackVerificationResult


class RobotPackCatalog:
    """Resolve built-in names, aliases, canonical refs, or explicit directories."""

    def __init__(self, *, builtin_root: str | Path | None = None) -> None:
        self.builtin_root = (
            Path(builtin_root).expanduser().resolve()
            if builtin_root is not None
            else Path(__file__).with_name("packs")
        )

    def list_builtin(self) -> list[RobotPackCatalogEntry]:
        entries: list[RobotPackCatalogEntry] = []
        if not self.builtin_root.is_dir():
            return entries
        for manifest_path in sorted(self.builtin_root.glob("*/robot-pack.yaml")):
            verification = verify_robot_pack(manifest_path.parent)
            manifest = verification.require_valid()
            entries.append(
                RobotPackCatalogEntry(
                    root=manifest_path.parent.resolve(),
                    manifest=manifest,
                    verification=verification,
                )
            )
        return entries

    def resolve(self, source: str | Path) -> RobotPackCatalogEntry:
        candidate = Path(source).expanduser()
        if candidate.exists():
            root = candidate.parent if candidate.is_file() else candidate
            verification = verify_robot_pack(root)
            return RobotPackCatalogEntry(
                root=root.resolve(),
                manifest=verification.require_valid(),
                verification=verification,
            )

        query = str(source).strip()
        matches: list[RobotPackCatalogEntry] = []
        for entry in self.list_builtin():
            manifest = entry.manifest
            names = {
                manifest.canonical_ref,
                f"{manifest.pack.namespace}/{manifest.pack.name}",
                f"{manifest.pack.namespace}/{manifest.pack.name}@{manifest.pack.version}",
                manifest.pack.name,
                *manifest.pack.aliases,
            }
            if query in names:
                matches.append(entry)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            refs = ", ".join(entry.manifest.canonical_ref for entry in matches)
            raise RobotPackNotFoundError(f"Robot Pack reference is ambiguous: {query!r}: {refs}")
        available = (
            ", ".join(entry.manifest.canonical_ref for entry in self.list_builtin()) or "none"
        )
        raise RobotPackNotFoundError(
            f"Robot Pack not found: {query!r}. Built-in packs: {available}"
        )


__all__ = [
    "RobotPackCatalog",
    "RobotPackCatalogEntry",
    "RobotPackNotFoundError",
]
