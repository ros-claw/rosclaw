"""ROSClaw URI parser and body reference constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RosclawURI:
    """Parse and represent a rosclaw:// URI."""

    raw: str
    resource_type: str = ""
    path: str = ""
    version: str | None = None
    qualifier: str | None = None

    SCHEME = "rosclaw"

    def __init__(self, uri: str):
        self.raw = uri
        self._parse()

    @classmethod
    def is_rosclaw_uri(cls, uri: str) -> bool:
        return uri.startswith(f"{cls.SCHEME}://")

    def _parse(self) -> None:
        if not self.is_rosclaw_uri(self.raw):
            raise ValueError(f"Not a rosclaw URI: {self.raw}")

        rest = self.raw[len(f"{self.SCHEME}://"):]
        parts = rest.split("/", 2)
        if len(parts) < 2:
            raise ValueError(f"Invalid rosclaw URI: {self.raw}")

        self.resource_type = parts[0]
        tail = parts[1]
        qualifier = parts[2] if len(parts) > 2 else None

        # Split version from tail (e.g., unitree-g1@1.0.0)
        if "@" in tail:
            self.path, self.version = tail.split("@", 1)
        else:
            self.path = tail

        if qualifier:
            self.qualifier = qualifier

    def __str__(self) -> str:
        return self.raw

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "resource_type": self.resource_type,
            "path": self.path,
            "version": self.version,
            "qualifier": self.qualifier,
        }


# Common URI helpers used across the codebase.


def eurdf_uri(profile_id: str, version: str | None = None) -> str:
    version_part = f"@{version}" if version else ""
    return f"rosclaw://eurdf/{profile_id}{version_part}"


def body_current_uri(qualifier: str | None = None) -> str:
    if qualifier:
        return f"rosclaw://body/current/{qualifier}"
    return "rosclaw://body/current"


def body_instance_uri(instance_id: str, qualifier: str | None = None) -> str:
    if qualifier:
        return f"rosclaw://body/{instance_id}/{qualifier}"
    return f"rosclaw://body/{instance_id}"
