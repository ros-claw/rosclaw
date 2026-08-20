from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.simforge.g1_promo_reel import (
    G1PromoArtifact,
    G1PromoPackResult,
    G1PromoSource,
    _clip_offsets,
    _file_hash,
    _require_sibling_evidence,
    render_g1_promo_pack,
)


def test_promo_pack_cannot_be_written_inside_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        render_g1_promo_pack(
            precision_manifest_path=tmp_path / "precision.json",
            coupled_manifest_path=tmp_path / "coupled.json",
            moving_manifest_path=tmp_path / "moving.json",
            output_dir=tmp_path / "promo",
            source_checkout=tmp_path,
        )


def test_promo_manifest_never_claims_left_foot_physics(tmp_path: Path) -> None:
    source = G1PromoSource(
        source_id="precision",
        manifest_path=str(tmp_path / "source.json"),
        manifest_hash="sha256:" + "1" * 64,
        video_path=str(tmp_path / "source.mp4"),
        video_hash="sha256:" + "2" * 64,
        width=1920,
        height=1080,
        duration_sec=12.0,
        strict_physics_source=False,
        simultaneous_two_body_physics=False,
        candidate_only=True,
    )
    artifact = G1PromoArtifact(
        artifact_id="dual-foot-study",
        output_path=str(tmp_path / "dual.mp4"),
        video_hash="sha256:" + "3" * 64,
        width=1920,
        height=1080,
        fps=30,
        duration_sec=12.0,
        clip_count=2,
        contains_symmetry_augmented_left_foot=True,
    )
    result = G1PromoPackResult(
        output_dir=str(tmp_path),
        manifest_path=str(tmp_path / "manifest.json"),
        report_path=str(tmp_path / "report.md"),
        sources=(source,),
        artifacts=(artifact,),
    )

    value = result.to_dict()

    assert value["left_foot_physics_claimed"] is False
    assert value["claims"]["actual_left_foot_physics"] is False
    assert value["claims"]["left_foot_is_symmetry_augmented_visualization"] is True
    assert value["pixels_used_for_scoring"] is False


def test_promo_clip_offsets_are_cumulative() -> None:
    clips = [
        {"duration_sec": 2.5},
        {"duration_sec": 4.0},
        {"duration_sec": 1.25},
    ]

    assert _clip_offsets(clips) == (0.0, 2.5, 6.5)


def test_promo_source_requires_passing_sibling_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"passed": true}\n', encoding="utf-8")
    manifest = tmp_path / "video.json"
    manifest.write_text("{}\n", encoding="utf-8")

    _require_sibling_evidence(
        manifest=manifest,
        name=evidence.name,
        expected_hash=_file_hash(evidence),
    )

    evidence.write_text('{"passed": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _require_sibling_evidence(
            manifest=manifest,
            name=evidence.name,
            expected_hash="sha256:" + "0" * 64,
        )
