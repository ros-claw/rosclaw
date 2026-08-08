from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import rosclaw.collective.sources.motiondecode.motion_prior as module
from rosclaw.collective.sources.motiondecode.audit import MotionDecodeAuditThresholds
from rosclaw.collective.sources.motiondecode.repair import MotionRepairDisposition


def test_repaired_q1_prior_episode_replays_manifest_instead_of_raw_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repaired_episode = object()
    manifest = object()
    captured: dict[str, object] = {}

    def replay(*args: object, **kwargs: object) -> object:
        captured["args"] = args
        captured.update(kwargs)
        return repaired_episode

    def raw_parse(*args: object, **kwargs: object) -> object:
        raise AssertionError("a repaired Q1 clip must not be parsed from the raw CSV")

    monkeypatch.setattr(module, "replay_segmentation_repair", replay)
    monkeypatch.setattr(module, "parse_motion_csv", raw_parse)
    registration = SimpleNamespace(
        manifest=SimpleNamespace(manifest_hash="sha256:" + "1" * 64, sample_rate_hz=120.0)
    )
    repair_result = SimpleNamespace(
        disposition=MotionRepairDisposition.REPAIRED_Q1,
        repair_manifest=manifest,
    )

    result = module._load_motion_prior_episode(
        registration=cast(Any, registration),
        dataset_root=tmp_path,
        model_path=tmp_path / "g1.xml",
        verified_path=tmp_path / "clip.csv",
        record=cast(Any, SimpleNamespace(content_hash="sha256:" + "2" * 64)),
        target_body_hash="sha256:" + "3" * 64,
        repair_result=cast(Any, repair_result),
        repair_thresholds=MotionDecodeAuditThresholds(),
    )

    assert result is repaired_episode
    assert captured["manifest"] is manifest
    assert captured["target_model_path"] == tmp_path / "g1.xml"
