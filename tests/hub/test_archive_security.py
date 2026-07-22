"""Version-independent safe extraction tests for remote Hub bundles."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from rosclaw.hub._compat import extractall_tar
from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.installer import Installer, InstallOptions


def _archive(*members: tuple[tarfile.TarInfo, bytes]) -> tarfile.TarFile:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for info, content in members:
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    buffer.seek(0)
    return tarfile.open(fileobj=buffer, mode="r:gz")


def test_extract_rejects_parent_path_escape(tmp_path: Path) -> None:
    archive = _archive((tarfile.TarInfo("../outside.txt"), b"outside"))
    with archive, pytest.raises(ValueError, match="unsafe path"):
        extractall_tar(archive, tmp_path / "target")
    assert not (tmp_path / "outside.txt").exists()


def test_extract_rejects_absolute_path(tmp_path: Path) -> None:
    archive = _archive((tarfile.TarInfo("/tmp/rosclaw-hub-escape"), b"outside"))
    with archive, pytest.raises(ValueError, match="unsafe path"):
        extractall_tar(archive, tmp_path / "target")


def test_extract_rejects_symbolic_link(tmp_path: Path) -> None:
    info = tarfile.TarInfo("payload-link")
    info.type = tarfile.SYMTYPE
    info.linkname = "../outside.txt"
    archive = _archive((info, b""))
    with archive, pytest.raises(ValueError, match="link"):
        extractall_tar(archive, tmp_path / "target")


def test_extract_rejects_duplicate_member(tmp_path: Path) -> None:
    archive = _archive(
        (tarfile.TarInfo("manifest.yaml"), b"first"),
        (tarfile.TarInfo("manifest.yaml"), b"second"),
    )
    with archive, pytest.raises(ValueError, match="duplicate"):
        extractall_tar(archive, tmp_path / "target")


def test_extract_rejects_control_character_path(tmp_path: Path) -> None:
    archive = _archive((tarfile.TarInfo("bad\nname.txt"), b"payload"))
    with archive, pytest.raises(ValueError, match="unsafe path"):
        extractall_tar(archive, tmp_path / "target")


def test_extract_enforces_uncompressed_size_limit(tmp_path: Path) -> None:
    archive = _archive((tarfile.TarInfo("large.bin"), b"12345"))
    with archive, pytest.raises(ValueError, match="size limit"):
        extractall_tar(archive, tmp_path / "target", max_total_size=4)


def test_extract_enforces_member_limit(tmp_path: Path) -> None:
    archive = _archive(
        (tarfile.TarInfo("first.txt"), b"first"),
        (tarfile.TarInfo("second.txt"), b"second"),
    )
    with archive, pytest.raises(ValueError, match="member limit"):
        extractall_tar(archive, tmp_path / "target", max_members=1)


def test_extract_regular_files(tmp_path: Path) -> None:
    archive = _archive(
        (tarfile.TarInfo("manifest.yaml"), b"schema_version: hub.asset.v1\n"),
        (tarfile.TarInfo("artifacts/data.bin"), b"payload"),
    )
    target = tmp_path / "target"
    with archive:
        extractall_tar(archive, target)
    assert (target / "manifest.yaml").is_file()
    assert (target / "artifacts" / "data.bin").read_bytes() == b"payload"


def test_local_bundle_install_rejects_path_escape(tmp_path: Path) -> None:
    bundle_path = tmp_path / "malicious.rosclaw"
    with tarfile.open(bundle_path, mode="w:gz") as archive:
        info = tarfile.TarInfo("../outside.txt")
        content = b"outside"
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    with pytest.raises(HubError) as exc_info:
        Installer(home=tmp_path / "home").install_local(
            bundle_path,
            options=InstallOptions(
                accept_license=True,
                skip_health=True,
                skip_mcp_merge=True,
            ),
        )

    assert exc_info.value.code == HubErrorCode.INDEX_VERIFY_FAILED
    assert "unsafe path" in exc_info.value.message.lower()
    assert not (tmp_path / "outside.txt").exists()
