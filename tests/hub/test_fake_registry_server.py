"""Security and ingestion checks for the fixture HTTP registry server."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from rosclaw.hub.publisher import Publisher, PublishOptions
from tests.fixtures.fake_registry.server import _AuthHandler

FIXTURES = Path(__file__).parent.parent / "fixtures" / "hub_assets"


def test_fixture_server_ingests_bundle_before_temp_cleanup(tmp_path: Path) -> None:
    bundle_path = tmp_path / "skill.rosclaw"
    result = Publisher(
        PublishOptions(home=tmp_path / "publisher-home", output=bundle_path)
    ).publish(FIXTURES / "skill_valid")
    assert result.bundle_path is not None

    registry = tmp_path / "registry"
    upload_path = "/upload/skill/rosclaw/g1-pick-place/1.2.0.rosclaw"
    uploaded = registry / upload_path.lstrip("/")
    uploaded.parent.mkdir(parents=True)
    uploaded.write_bytes(result.bundle_path.read_bytes())
    (registry / "catalog.jsonl").write_text("", encoding="utf-8")

    handler = object.__new__(_AuthHandler)
    response = handler._ingest_bundle(uploaded.read_bytes(), registry, upload_path)

    assert (registry / str(response["manifest_url"])).is_file()
    assert (registry / str(response["bundle_url"])).is_file()
    assert any((registry / "blobs" / "sha256").iterdir())
    catalog = json.loads((registry / "catalog.jsonl").read_text(encoding="utf-8"))
    assert catalog["ref"] == "rosclaw://skill/rosclaw/g1-pick-place@1.2.0"


def test_fixture_server_rejects_archive_path_escape(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("../outside.txt")
        content = b"outside"
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    handler = object.__new__(_AuthHandler)
    with pytest.raises(ValueError, match="unsafe path"):
        handler._ingest_bundle(
            buffer.getvalue(),
            tmp_path / "registry",
            "/upload/skill/rosclaw/escape/1.0.0.rosclaw",
        )

    assert not (tmp_path / "outside.txt").exists()
