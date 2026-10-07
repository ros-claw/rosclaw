"""Bounded CLI example for binary cloud quality analysis.

Usage:
    python -B examples/perception_binary_cloud_quality.py \
        --metadata ABS_JSON --data ABS_BIN --root ABS_OWNED_ROOT --sha256 64HEX

stdout carries exactly one JSON object: the seven-key result dict produced by
``analyze_cloud_file``. Errors go to stderr as exactly one JSON object
``{"status": "ERROR", "code": ..., "error_class": ...}`` (<= 512 UTF-8 bytes,
no caller path or input echo) with a non-zero exit code.

Pure generated-binary software: no ROS/DDS, no neural network, no hardware,
no Artifact resolver/catalog. The metadata JSON file is bounded to 16 KiB
and the binary file to 8 MiB inside the caller-supplied root.

Metadata admission: the metadata path must name an ordinary regular file.
Non-regular files (FIFO, directory, device) and leaf symlinks are rejected
before any blocking stream is opened; oversize, malformed-JSON and
non-object metadata are rejected before the canonical payload function is
called.
"""

import argparse
import json
import os
import stat
import sys

from rosclaw.perception.binary_cloud_quality import (
    MAX_BINARY_BYTES,
    analyze_cloud_file,
)

_MAX_METADATA_JSON_BYTES = 16384


class MetadataAdmissionError(Exception):
    """Metadata input failed admission before the payload function."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _load_metadata(path: str) -> dict:
    if type(path) is not str:
        raise MetadataAdmissionError("metadata_path_not_plain_str")
    # lstat first: never open a blocking stream on a non-regular file and
    # never follow a leaf symlink.
    try:
        st = os.lstat(path)
    except OSError:
        raise MetadataAdmissionError("metadata_not_readable") from None
    if not stat.S_ISREG(st.st_mode):
        raise MetadataAdmissionError("metadata_not_regular_file")
    if st.st_size > _MAX_METADATA_JSON_BYTES:
        raise MetadataAdmissionError("metadata_exceeds_byte_budget")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        raise MetadataAdmissionError("metadata_not_readable") from None
    try:
        try:
            st_fd = os.fstat(fd)
        except OSError:
            raise MetadataAdmissionError("metadata_not_readable") from None
        if not stat.S_ISREG(st_fd.st_mode):
            raise MetadataAdmissionError("metadata_not_regular_file")
        if st_fd.st_size > _MAX_METADATA_JSON_BYTES:
            raise MetadataAdmissionError("metadata_exceeds_byte_budget")
        try:
            raw = os.read(fd, _MAX_METADATA_JSON_BYTES + 1)
        except OSError:
            raise MetadataAdmissionError("metadata_not_readable") from None
    finally:
        os.close(fd)
    if len(raw) > _MAX_METADATA_JSON_BYTES:
        raise MetadataAdmissionError("metadata_exceeds_byte_budget")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise MetadataAdmissionError("metadata_malformed_json") from None
    if type(parsed) is not dict:
        raise MetadataAdmissionError("metadata_not_json_object")
    return parsed


def _emit_error(code: str, error_class: str) -> int:
    payload = json.dumps({"status": "ERROR", "code": str(code), "error_class": str(error_class)})
    encoded = payload.encode("utf-8")
    if len(encoded) > 512:
        encoded = b'{"status":"ERROR","code":"internal_error","error_class":"Error"}'
    sys.stderr.write(encoded.decode("utf-8") + "\n")
    return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bounded binary cloud quality analysis")
    parser.add_argument("--metadata", required=True, help="absolute path to metadata JSON")
    parser.add_argument("--data", required=True, help="absolute path to binary cloud data")
    parser.add_argument("--root", required=True, help="caller-owned allowed root for --data")
    parser.add_argument("--sha256", required=True, help="expected sha256 of --data (64 hex)")
    args = parser.parse_args(argv)
    try:
        metadata = _load_metadata(args.metadata)
    except MetadataAdmissionError as exc:
        return _emit_error(exc.code, "MetadataAdmissionError")
    try:
        result = analyze_cloud_file(
            metadata,
            args.data,
            allowed_root=args.root,
            expected_sha256=args.sha256,
        )
    except (TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return _emit_error("payload_rejected", type(exc).__name__)
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


assert MAX_BINARY_BYTES == 8 * 1024 * 1024  # documented bound, kept explicit
