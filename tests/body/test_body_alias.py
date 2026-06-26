"""Tests for BODY.md alias behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.body.service import BodyInstanceService


@pytest.fixture
def linked_body(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    service = BodyInstanceService(workspace=tmp_path)
    service.create_or_init(robot="unitree-g1", name="test-001", mode="single")
    return tmp_path


def test_body_md_exists_after_init(linked_body: Path):
    assert (linked_body / "body" / "BODY.md").exists()


def test_body_md_references_embodiment(linked_body: Path):
    body_md = (linked_body / "body" / "BODY.md").read_text(encoding="utf-8")
    assert "EMBODIMENT.md" in body_md
    assert "alias" in body_md.lower() or "Canonical file: EMBODIMENT.md" in body_md


def test_body_md_is_not_second_source(linked_body: Path):
    body_md = (linked_body / "body" / "BODY.md").read_text(encoding="utf-8")
    assert "Do not edit this copy directly" in body_md


def test_body_md_refreshed_on_render(linked_body: Path):
    body_md_path = linked_body / "body" / "BODY.md"
    body_md_path.write_text("stale content", encoding="utf-8")

    resolver = BodyResolver(workspace=linked_body)
    resolver.refresh_all_artifacts(reason="test body md refresh")

    body_md = body_md_path.read_text(encoding="utf-8")
    assert "stale content" not in body_md
    assert "EMBODIMENT.md" in body_md
