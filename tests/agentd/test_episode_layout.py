"""Regression for file-only staging dropping empty receipt directories."""

import json
import shutil
from pathlib import Path

import pytest

from rosclaw.agentd.episode_layout import (
    EpisodeLayoutError,
    validate_episode_output_layout,
)


def stage_files(source: Path, target: Path) -> None:
    target.mkdir()
    for path in source.rglob("*"):
        if path.is_file():
            dest = target / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)


def test_file_only_staging_rejects_missing_parent_before_dispatch(tmp_path: Path):
    source = tmp_path / "prepared"
    (source / "operator").mkdir(parents=True)
    (source / "spec.json").write_text("{}\n")
    staged = tmp_path / "staged"
    stage_files(source, staged)
    before = sorted(str(p) for p in staged.rglob("*"))
    dispatched = []
    with pytest.raises(EpisodeLayoutError, match="MISSING_OUTPUT_PARENT:clock"):
        validate_episode_output_layout(staged, {"clock": Path("operator/clock.json")})
        dispatched.append(True)
    assert not dispatched
    assert sorted(str(p) for p in staged.rglob("*")) == before


def test_pinned_marker_survives_transfer_and_real_first_write(tmp_path: Path):
    source = tmp_path / "prepared"
    (source / "operator").mkdir(parents=True)
    (source / "operator/.keep").write_text("Prepared output directory.\n")
    staged = tmp_path / "staged"
    stage_files(source, staged)
    paths = validate_episode_output_layout(
        staged,
        {"clock": Path("operator/clock.json"), "birth": Path("operator/birth.json")},
    )
    # Admission has no output side effects. Exercise the real exclusive write
    # boundary after admission in this disposable fixture.
    assert not Path(paths["clock"]).exists()
    with Path(paths["clock"]).open("x") as stream:
        json.dump({"fixture": True}, stream)
    assert json.loads(Path(paths["clock"]).read_text()) == {"fixture": True}
    with pytest.raises(EpisodeLayoutError, match="OUTPUT_ALREADY_EXISTS:clock"):
        validate_episode_output_layout(staged, {"clock": Path(paths["clock"])})


def test_resolved_private_alias_and_duplicate_output(tmp_path: Path):
    root = tmp_path / "case"
    (root / "operator").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    paths = validate_episode_output_layout(alias, {"clock": alias / "operator/c.json"})
    assert paths["clock"] == str(root / "operator/c.json")
    with pytest.raises(EpisodeLayoutError, match="DUPLICATE_OUTPUT_PATH"):
        validate_episode_output_layout(
            root, {"clock": Path("operator/c.json"), "birth": alias / "operator/c.json"}
        )


def test_escape_and_dangling_output_symlink_rejected(tmp_path: Path):
    root = tmp_path / "case"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "operator").symlink_to(outside, target_is_directory=True)
    with pytest.raises(EpisodeLayoutError, match="OUTPUT_PARENT_OUTSIDE_CASE"):
        validate_episode_output_layout(root, {"clock": Path("operator/c.json")})
    (root / "dangling.json").symlink_to(tmp_path / "missing.json")
    with pytest.raises(EpisodeLayoutError, match="OUTPUT_ALREADY_EXISTS"):
        validate_episode_output_layout(root, {"clock": Path("dangling.json")})
