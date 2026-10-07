"""Native CLI metadata-admission regression tests.

Spawns the real example CLI as a subprocess against owned finite fixtures:
an ordinary 16KiB-bounded metadata JSON (positive), a FIFO, a directory, a
leaf symlink, an oversize JSON, a malformed JSON and a non-object JSON.
Every rejection must be prompt (no FIFO hang), exit non-zero, carry exactly
one structured JSON error object on stderr (status ERROR / code /
error_class, <= 512 UTF-8 bytes, no path echo), and the ordinary positive
must preserve the seven-key VALID_CLOUD result.
"""

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[1]
EXAMPLE = WORKSPACE / "examples" / "perception_binary_cloud_quality.py"

_TIMEOUT_SEC = 10.0


def _valid_metadata() -> dict:
    return {
        "width": 1,
        "height": 1,
        "point_step": 12,
        "row_step": 12,
        "is_bigendian": False,
        "is_dense": True,
        "fields": [
            {"name": name, "offset": i * 4, "datatype": 7, "count": 1}
            for i, name in enumerate("xyz")
        ],
    }


@pytest.fixture()
def fixtures(tmp_path: Path) -> dict:
    raw = struct.pack("<fff", 1.0, 2.0, 3.0)
    data = tmp_path / "cloud.bin"
    data.write_bytes(raw)
    paths = {"root": tmp_path, "data": data, "sha256": hashlib.sha256(raw).hexdigest()}
    ordinary = tmp_path / "metadata.json"
    ordinary.write_text(json.dumps(_valid_metadata()))
    paths["ordinary"] = ordinary
    fifo = tmp_path / "metadata.pipe"
    os.mkfifo(fifo, 0o600)
    paths["fifo"] = fifo
    directory = tmp_path / "metadata.dir"
    directory.mkdir()
    paths["directory"] = directory
    link = tmp_path / "metadata.link"
    link.symlink_to("metadata.json")
    paths["leaf_symlink"] = link
    oversize = tmp_path / "oversize.json"
    oversize.write_text(" " * 16385)
    paths["oversize"] = oversize
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json")
    paths["malformed"] = malformed
    nonobject = tmp_path / "nonobject.json"
    nonobject.write_text("[]")
    paths["nonobject"] = nonobject
    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text(json.dumps(_valid_metadata()))
    unreadable.chmod(0)
    paths["unreadable"] = unreadable
    return paths


def _run_cli(metadata: Path, fixtures: dict) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = f"{WORKSPACE / 'src'}:{WORKSPACE}"
    return subprocess.run(
        [
            sys.executable,
            "-B",
            str(EXAMPLE),
            "--metadata",
            str(metadata),
            "--data",
            str(fixtures["data"]),
            "--root",
            str(fixtures["root"]),
            "--sha256",
            fixtures["sha256"],
        ],
        capture_output=True,
        timeout=_TIMEOUT_SEC,
        env=env,
    )


def _assert_structured_error(result: subprocess.CompletedProcess, metadata: Path) -> None:
    assert result.returncode != 0
    assert result.stdout == b""
    raw = result.stderr
    assert len(raw) <= 512
    lines = raw.decode("utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert type(parsed) is dict
    assert parsed.get("status") == "ERROR"
    assert type(parsed.get("code")) is str and parsed["code"]
    assert type(parsed.get("error_class")) is str and parsed["error_class"]
    # no caller path or input echo
    assert str(metadata) not in lines[0]
    assert metadata.name not in lines[0]


def test_ordinary_metadata_accepted_valid_cloud(fixtures: dict) -> None:
    result = _run_cli(fixtures["ordinary"], fixtures)
    assert result.returncode == 0, result.stderr
    assert result.stderr == b""
    out = json.loads(result.stdout.decode("utf-8"))
    assert out.get("status") == "VALID_CLOUD"
    assert len(out) == 7


@pytest.mark.parametrize(
    "kind",
    ["fifo", "directory", "leaf_symlink", "oversize", "malformed", "nonobject"],
)
def test_nonadmissible_metadata_rejected_promptly(fixtures: dict, kind: str) -> None:
    metadata = fixtures[kind]
    result = _run_cli(metadata, fixtures)  # raises TimeoutExpired on a hang
    _assert_structured_error(result, metadata)


def test_fifo_rejected_without_blocking(fixtures: dict) -> None:
    # Dedicated FIFO case: subprocess.run timeout proves no blocking open.
    import time

    start = time.monotonic()
    result = _run_cli(fixtures["fifo"], fixtures)
    assert time.monotonic() - start < _TIMEOUT_SEC
    _assert_structured_error(result, fixtures["fifo"])


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses mode-000 permission checks")
def test_unreadable_mode000_metadata_rejected_structured(fixtures: dict) -> None:
    # Ordinary regular file with mode 000: lstat passes but os.open raises
    # PermissionError. Must be a bounded structured rejection, no traceback.
    metadata = fixtures["unreadable"]
    assert not os.access(metadata, os.R_OK)
    result = _run_cli(metadata, fixtures)
    _assert_structured_error(result, metadata)
    parsed = json.loads(result.stderr.decode("utf-8").strip())
    assert parsed["code"] == "metadata_not_readable"
    assert "Traceback" not in result.stderr.decode("utf-8")
