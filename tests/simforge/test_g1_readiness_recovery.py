from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.simforge.g1_readiness_recovery import (
    G1ReadinessRecoveryConfig,
    run_g1_readiness_recovery,
)
from rosclaw.simforge.g1_readiness_recovery_video import (
    render_g1_readiness_recovery_video,
)


def test_readiness_recovery_config_keeps_a_bounded_physical_de_load_window() -> None:
    config = G1ReadinessRecoveryConfig()

    assert config.neural_deceleration_duration_sec == 1.80
    assert config.hold_duration_sec == 1.20
    assert config.gain_scale == 0.75
    assert config.maximum_final_speed_mps == 0.20

    with pytest.raises(ValueError, match="neural deceleration duration"):
        G1ReadinessRecoveryConfig(neural_deceleration_duration_sec=0.39)
    with pytest.raises(ValueError, match="hold duration"):
        G1ReadinessRecoveryConfig(hold_duration_sec=3.01)
    with pytest.raises(ValueError, match="gain scale"):
        G1ReadinessRecoveryConfig(gain_scale=1.01)
    with pytest.raises(ValueError, match="finite"):
        G1ReadinessRecoveryConfig(maximum_final_speed_mps=float("nan"))


def test_readiness_recovery_rejects_unknown_evidence_domains(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evidence domain"):
        run_g1_readiness_recovery(
            asset_root=tmp_path / "assets",
            sonic_model_root=tmp_path / "sonic",
            output_dir=tmp_path.parent / "outside-evidence",
            source_checkout=tmp_path,
            router=None,  # type: ignore[arg-type]
            readiness_gate=None,  # type: ignore[arg-type]
            sonic_config=None,  # type: ignore[arg-type]
            evidence_domain="UNKNOWN",
        )


def test_readiness_recovery_video_must_remain_outside_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the checkout"):
        render_g1_readiness_recovery_video(
            evaluation_path=tmp_path / "evaluation.json",
            evidence_paths=(),
            asset_root=tmp_path / "assets",
            output_path=tmp_path / "video.mp4",
            source_checkout=tmp_path,
        )
