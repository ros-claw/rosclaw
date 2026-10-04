"""Strict, opt-in binding for declared source files and a preparation CLI.

This does not discover Python imports/assets or authorize physical execution.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Sha256 = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
MAX_SOURCE_FILE_BYTES = 512 * 1024 * 1024
MAX_SOURCE_CLOSURE_BYTES = 1024 * 1024 * 1024


def relative_source_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or "\\" in value or path.is_absolute()
            or any(part in ("", ".", "..") for part in value.split("/"))
            or str(path) != value):
        raise ValueError("source path must be a canonical workspace-relative path")
    return value


class SourceFileV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sha256: Sha256
    size_bytes: Annotated[int, Field(ge=0, le=MAX_SOURCE_FILE_BYTES)]


class SourcePacketRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return relative_source_path(value)


class SourcePacketV1(BaseModel):
    # Security gate inputs reject unknown fields rather than interpreting arbitrary
    # nested user JSON as trusted provenance or silently ignoring a future schema.
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["rosclaw.source_packet.v1"]
    scope: Literal["source_preflight_only"]
    closure_kind: Literal["declared_files"]
    files: Annotated[dict[str, SourceFileV1], Field(min_length=1, max_length=256)]
    canonical_entry: SourcePacketRefV1
    preflight_argv: Annotated[list[str], Field(min_length=2, max_length=64)]

    @model_validator(mode="after")
    def bind_entry(self) -> SourcePacketV1:
        for path in self.files:
            relative_source_path(path)
        if sum(file.size_bytes for file in self.files.values()) > MAX_SOURCE_CLOSURE_BYTES:
            raise ValueError("declared source closure exceeds byte bound")
        entry = self.files.get(self.canonical_entry.path)
        if entry is None or entry.sha256 != self.canonical_entry.sha256:
            raise ValueError("canonical entry must match a declared file digest")
        if (self.preflight_argv[0] not in ("python3", "python")
                or self.preflight_argv[1] != self.canonical_entry.path):
            raise ValueError("preflight must directly execute the canonical Python file")
        if any("\x00" in arg or len(arg) > 4096 for arg in self.preflight_argv):
            raise ValueError("invalid preflight argument")
        return self
