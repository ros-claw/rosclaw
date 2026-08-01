from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.collective.cli import dispatch_collective_argv
from rosclaw.collective.sources.motiondecode.audit import run_motiondecode_pilot
from rosclaw.collective.sources.motiondecode.license import inspect_motiondecode_license
from rosclaw.collective.sources.motiondecode.manifest import inspect_motiondecode_source
from rosclaw.collective.sources.motiondecode.motion_prior import (
    load_g1_motion_prior_artifact,
    train_motion_prior_worker,
)
from rosclaw.collective.sources.motiondecode.parser import (
    MOTIONDECODE_HEADER,
    parse_motiondecode_csv,
)
from rosclaw.collective.sources.motiondecode.taxonomy import (
    MotionDecodeStratum,
    select_motiondecode_pilot,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES

_REVISION = "a" * 40


def _write_license(root: Path) -> None:
    (root / "LICENSE").write_text("")
    (root / "LICENSE.md").write_text(
        "Permitted uses: academic research and prototyping (non-commercial).\n"
        "Prohibited without written permission: commercial distribution.\n"
        "Users must retain attribution.\n"
    )


def _write_csv(path: Path, *, invalid_header: bool = False, nonfinite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(MOTIONDECODE_HEADER)
    if invalid_header:
        header[-1] = "wrong_joint"
    frames = []
    for index in range(80):
        row = [0.01 * index, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, *([0.0] * 29)]
        frames.append(row)
    if nonfinite:
        frames[2][-1] = float("nan")
    lines = [",".join(header), *(",".join(str(value) for value in row) for row in frames)]
    path.write_text("\n".join(lines) + "\n")


def _dataset(root: Path) -> Path:
    root.mkdir()
    _write_license(root)
    metadata = root / "metadata"
    metadata.mkdir()
    metadata.joinpath("index.csv").write_text(
        "Label,Primary class name\n"
        "1.1.2.2,Basic_Movement_Category\n"
        "1.3.1.1,Basic_Gait_Category\n"
        "1.5.1.1,Balance_Control_Actions\n"
        "3.3.1.1,Ball_Game_Interaction\n"
    )
    relative = (
        "samples/1.1.Basic_Movement_Category/1.1.2.Ground_Recovery_Movement/"
        "1.1.2.2.Sit_to_Stand/recovery.csv"
    )
    _write_csv(root / relative)
    _write_csv(
        root
        / "samples/1.3.Basic_Gait_Category/1.3.1.Walking/1.3.1.1.Normal_Walking/gait.csv"
    )
    _write_csv(
        root
        / "samples/1.5.Balance_Control_Actions/1.5.1.Dynamic_Balance/"
        "1.5.1.1.Single_Leg_Standing/balance.csv"
    )
    _write_csv(
        root
        / "samples/3.3.Ball_Game_Interaction/3.3.1.Football/"
        "3.3.1.1.Instep_Kick/football.csv"
    )
    return root


def _model(path: Path) -> Path:
    bodies = []
    closing = []
    indent = "      "
    for index, name in enumerate(G1_DDS_JOINT_NAMES):
        bodies.append(
            f'{indent}<body name="link_{index}" pos="0 0 0.001">\n'
            f'{indent}  <joint name="{name}" type="hinge" axis="1 0 0" range="-3 3"/>\n'
            f'{indent}  <geom type="sphere" size="0.001" contype="0" conaffinity="0" mass="0.01"/>'
        )
        closing.append(f"{indent}</body>")
        indent += "  "
    xml = (
        '<mujoco model="motiondecode_test">\n'
        '  <option gravity="0 0 -9.81"/>\n'
        "  <worldbody>\n"
        '    <geom name="floor" type="plane" size="5 5 0.1"/>\n'
        '    <body name="pelvis" pos="0 0 0.8">\n'
        '      <freejoint name="floating_base_joint"/>\n'
        '      <geom type="sphere" size="0.01" contype="0" conaffinity="0" mass="1"/>\n'
        + "\n".join(bodies)
        + "\n"
        + "\n".join(reversed(closing))
        + "\n    </body>\n  </worldbody>\n</mujoco>\n"
    )
    path.write_text(xml)
    return path


def test_source_manifest_is_honest_about_revision_license_and_modalities(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "motiondecode")
    manifest, paths = inspect_motiondecode_source(root, revision=_REVISION)

    assert len(paths) == 4
    assert manifest.source.revision_binding == "UNVERIFIED_LOCAL_SNAPSHOT"
    assert manifest.license.commercial_use_status == "WRITTEN_PERMISSION_REQUIRED"
    assert manifest.license.redistribution_permitted is False
    assert manifest.football_files == 1
    assert manifest.object_pose_files == 0
    assert manifest.capsule.action_semantics == ()
    assert manifest.capsule.training_eligible is False


def test_license_rejects_unpermitted_commercial_use(tmp_path: Path) -> None:
    root = tmp_path / "motiondecode"
    root.mkdir()
    _write_license(root)
    result = inspect_motiondecode_license(root, requested_usage="commercial")
    assert result.permitted is False


def test_parser_derives_120hz_state_but_not_actions(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "motiondecode")
    path = next((root / "samples").rglob("recovery.csv"))
    episode = parse_motiondecode_csv(path, dataset_root=root)

    assert episode.joint_position.shape == (80, 29)
    assert episode.joint_velocity.shape == (80, 29)
    assert episode.sample_rate_hz == 120.0
    assert episode.action_semantics == "ABSENT"
    assert episode.reward_semantics == "ABSENT"
    np.testing.assert_allclose(episode.time_sec[-1], 79 / 120)


@pytest.mark.parametrize("mode", ["header", "nonfinite"])
def test_parser_fails_closed_on_malformed_payload(tmp_path: Path, mode: str) -> None:
    root = tmp_path / "motiondecode"
    root.mkdir()
    path = root / "samples" / "bad.csv"
    _write_csv(path, invalid_header=mode == "header", nonfinite=mode == "nonfinite")
    with pytest.raises(ValueError):
        parse_motiondecode_csv(path, dataset_root=root)


def test_selection_backfills_without_mislabeling_missing_football() -> None:
    values = [
        *(Path(f"samples/Ground_Recovery/recovery_{index}.csv") for index in range(2)),
        *(Path(f"samples/Walking/gait_{index}.csv") for index in range(2)),
        *(Path(f"samples/Balance/balance_{index}.csv") for index in range(2)),
        *(Path(f"samples/Dance/dance_{index}.csv") for index in range(4)),
    ]
    selection = select_motiondecode_pilot(values, limit=8, seed=7)

    assert selection.shortages == {MotionDecodeStratum.FOOTBALL.value: 2}
    assert selection.substitutions == {MotionDecodeStratum.COORDINATION_SUPPLEMENT.value: 2}
    assert selection.selected_counts[MotionDecodeStratum.FOOTBALL.value] == 0
    assert len(selection.selected) == 8
    assert selection == select_motiondecode_pilot(reversed(values), limit=8, seed=7)


def test_pilot_emits_only_q1_kinematic_evidence_and_no_raw_data(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "motiondecode")
    model = _model(tmp_path / "g1.xml")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    output = tmp_path / "evidence"
    report = run_motiondecode_pilot(
        dataset_root=dataset,
        revision=_REVISION,
        model_path=model,
        output_dir=output,
        source_checkout=checkout,
        limit=4,
        seed=42,
    )

    assert report.pipeline_passed is True
    assert report.decision == "MOTION_PRIOR_ONLY"
    assert report.aggregates["q1_kinematic_only_count"] == 4
    assert report.aggregates["q2_or_higher_count"] == 0
    assert report.raw_data_exported is False
    assert report.hardware_authorized is False
    payload = json.loads((output / "motiondecode-pilot-report.json").read_text())
    assert payload["claims"]["direct_torque_labels"] is False
    assert not list(output.rglob("*.csv"))


def test_collective_inspect_cli_reports_machine_readable_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = _dataset(tmp_path / "motiondecode")
    code = dispatch_collective_argv(
        [
            "collective",
            "source",
            "inspect",
            "motiondecode",
            "--dataset-root",
            str(dataset),
            "--revision",
            _REVISION,
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["csv_file_count"] == 4
    assert payload["safe_to_read"] is True


def test_motion_prior_pack_cli_and_cpu_worker_are_safe_and_loadable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = _dataset(tmp_path / "motiondecode")
    model = _model(tmp_path / "g1.xml")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    evidence = tmp_path / "evidence"
    report = run_motiondecode_pilot(
        dataset_root=dataset,
        revision=_REVISION,
        model_path=model,
        output_dir=evidence,
        source_checkout=checkout,
        limit=4,
        seed=42,
    )
    assert report.pipeline_passed
    pack_path = tmp_path / "pack" / "motion-prior.npz"
    command = [
        "collective",
        "prior",
        "build",
        "--pilot-report",
        str(evidence / "motiondecode-pilot-report.json"),
        "--dataset-root",
        str(dataset),
        "--model-path",
        str(model),
        "--output",
        str(pack_path),
        "--sequence-length",
        "32",
        "--maximum-windows",
        "8",
        "--seed",
        "9",
        "--stratum",
        "balance_proxy",
        "--stratum",
        "gait",
        "--stratum",
        "transition_recovery",
        "--json",
    ]
    code = dispatch_collective_argv(command)
    cli_payload = json.loads(capsys.readouterr().out)
    metadata = json.loads(pack_path.with_suffix(".json").read_text())
    assert code == 0
    assert cli_payload["pack_hash"] == metadata["pack_hash"]
    assert cli_payload["feature_count"] == 61
    assert metadata["training_windows"] == 6
    assert metadata["validation_windows"] == 2
    assert metadata["allowed_strata"] == ["balance_proxy", "gait", "transition_recovery"]
    assert metadata["raw_data_exported"] is False
    assert dispatch_collective_argv(command) == 2
    assert "already exists" in json.loads(capsys.readouterr().out)["error"]
    artifact_value = train_motion_prior_worker(
        pack_path=pack_path,
        output_dir=tmp_path / "worker",
        device="cpu",
        seed=11,
        epochs=2,
        hidden_dim=8,
        batch_size=4,
    )
    artifact = load_g1_motion_prior_artifact(
        tmp_path / "worker" / "motion-prior-artifact.json"
    )
    assert artifact.artifact_hash == artifact_value["artifact_hash"]
    assert artifact.action_semantics == "ABSENT"
    assert artifact.activation_ceiling == "SIM_ONLY_REPRESENTATION_INITIALIZATION"
    assert artifact.tensors["gru.weight_ih_l0"].shape == (24, 61)
