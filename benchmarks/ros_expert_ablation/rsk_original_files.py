"""Identical bounded original-file access for new A/B/C/E; no ROS or writes."""

import hashlib
from typing import Annotated, Literal

from pydantic import Field, StrictStr, TypeAdapter

from benchmarks.ros_expert_ablation.physical_aci import Args, CaseArgs


class FileReadArgs(CaseArgs):
    file: StrictStr = Field(min_length=1, max_length=256)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=12000, ge=1, le=16000)


class FilesListCall(Args):
    operation: Literal["files.list"]
    arguments: CaseArgs


class FileReadCall(Args):
    operation: Literal["files.read"]
    arguments: FileReadArgs


FILE_CALLS = TypeAdapter(Annotated[FilesListCall | FileReadCall, Field(discriminator="operation")])


def normalize_original_read(request):
    if request.get("operation") == "files.read" and isinstance(request.get("arguments"), dict):
        for name in ["offset", "limit"]:
            if request["arguments"].get(name, False) is None:
                request["arguments"].pop(name)
    return request


class OriginalFilesMixin:
    def _extra_original_files(self):
        return {}

    def _original_files(self):
        self.base._bound(self.base.case.case_hash, read_only=True)
        files, extra = dict(self.base.tools.files), self._extra_original_files()
        if set(files) & set(extra):
            raise ValueError("ambiguous original file names")
        files.update(extra)
        if (
            any(type(text) is not str for text in files.values())
            or sum(len(text.encode()) for text in files.values()) > 16_000_000
        ):
            raise ValueError("bounded original raw task file corpus required")
        return files

    def original_files_catalog(self):
        return {
            "scope": "IDENTICAL_ORIGINAL_FILE_ACCESS_NO_DIAGNOSIS_OR_AUTHORITY",
            "files": [
                {
                    "file": name,
                    "utf8_bytes": len(text.encode()),
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
                for name, text in sorted(self._original_files().items())
            ],
            "staging_fields": self.base.tools.call("list_files", {})["allowed_staging_fields"],
            "current_authority": "NONE",
        }

    def original_file_call(self, request):
        call = FILE_CALLS.validate_python(request)
        self.base._bound(call.arguments.case_hash, read_only=True)
        if self.base.finished:
            raise ValueError("episode finished")
        if isinstance(call, FilesListCall):
            return self.original_files_catalog()
        args = call.arguments
        text = self._original_files()[args.file]
        return {
            "status": "ORIGINAL_FILE_SLICE_NOT_DIAGNOSIS",
            "file": args.file,
            "offset": args.offset,
            "total_characters": len(text),
            "text": text[args.offset : args.offset + args.limit],
            "truncated": args.offset + args.limit < len(text),
            "current_authority": "NONE",
        }
