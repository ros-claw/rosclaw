from __future__ import annotations

import json
from pathlib import Path

from rosclaw.dataset import (
    DatasetSnapshotState,
    FileHashMode,
    inspect_dataset_root,
    write_dataset_doctor_artifacts,
)
from rosclaw.dataset.cli import dispatch_dataset_argv


def _fixture(root: Path) -> Path:
    dataset = root / "OmniContact"
    (dataset / "npz" / "soccer" / "case1_kick_forward").mkdir(parents=True)
    (dataset / "LICENSE").write_text("CC-BY-4.0 fixture\n", encoding="utf-8")
    (dataset / "npz" / "soccer" / "case1_kick_forward" / "kick.npz").write_bytes(b"npz")
    (dataset / "empty.csv").write_bytes(b"")
    return dataset


def test_doctor_inventory_is_content_addressed_and_never_training_truth(tmp_path: Path) -> None:
    dataset = _fixture(tmp_path)
    (tmp_path / "MotionDecode").mkdir()
    (dataset / "download.tar.part").write_bytes(b"in progress")
    report = inspect_dataset_root(tmp_path, hash_mode=FileHashMode.SHA256)
    by_id = {value.dataset_id: value for value in report.inventories}

    assert by_id["OmniContact"].state is DatasetSnapshotState.TRANSFERRING
    assert by_id["OmniContact"].issue_counts["partial_transfer_file"] == 1
    assert by_id["OmniContact"].issue_counts["zero_byte_file"] == 1
    assert by_id["OmniContact"].football_matches
    assert by_id["MotionDecode"].state is DatasetSnapshotState.EMPTY
    assert report.snapshot_complete is False
    assert report.training_eligible is False
    assert report.report_hash.startswith("sha256:")


def test_transfer_assertion_prevents_clean_prefix_from_looking_complete(tmp_path: Path) -> None:
    _fixture(tmp_path)
    report = inspect_dataset_root(tmp_path, transfer_active=True)

    assert report.transfer_active is True
    assert report.inventories[0].state is DatasetSnapshotState.TRANSFERRING
    assert report.snapshot_complete is False


def test_lfs_pointer_is_fail_closed_partial_snapshot(tmp_path: Path) -> None:
    dataset = tmp_path / "GEAR-SONIC"
    dataset.mkdir()
    (dataset / "model.onnx").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08\n"
        "size 1000\n",
        encoding="utf-8",
    )
    report = inspect_dataset_root(tmp_path)

    inventory = report.inventories[0]
    assert inventory.state is DatasetSnapshotState.PARTIAL
    assert inventory.issue_counts == {"git_lfs_pointer": 1}


def test_football_assets_require_sport_context_and_exclude_cache_mirrors(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "MotionDecode"
    paths = (
        "samples/Football/Short_Pass/clip.csv",
        "samples/actions/kick_ball.csv",
        "samples/Object_Passing/pass_item.csv",
        "samples/Basketball/Dribbling/clip.csv",
        "samples/dance/Front_Leg_Kick/clip.csv",
        ".cache/huggingface/download/samples/Football/clip.csv.metadata",
    )
    for relative in paths:
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sample")

    inventory = inspect_dataset_root(tmp_path).inventories[0]

    assert inventory.football_matches == (
        "samples/Football/Short_Pass/clip.csv",
        "samples/actions/kick_ball.csv",
    )


def test_writer_emits_inventory_quality_license_and_asset_matrix(tmp_path: Path) -> None:
    _fixture(tmp_path)
    report = inspect_dataset_root(tmp_path)
    output = tmp_path / "evidence"
    paths = write_dataset_doctor_artifacts(report, output)

    assert set(paths) == {
        "inventory",
        "quality_report",
        "license_manifest",
        "football_asset_matrix",
    }
    inventory = json.loads((output / "dataset_inventory.json").read_text(encoding="utf-8"))
    licenses = json.loads((output / "license_manifest.json").read_text(encoding="utf-8"))
    assert inventory["report_hash"] == report.report_hash
    assert licenses["datasets"][0]["decision"] == "pending_operator_review"
    assert "OmniContact" in (output / "dataset_quality_report.html").read_text(encoding="utf-8")


def test_cli_writes_outside_source_and_reports_incomplete_without_failing(
    tmp_path: Path,
    capsys: object,
) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    _fixture(root)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    output = tmp_path / "evidence"

    result = dispatch_dataset_argv(
        [
            "dataset",
            "doctor",
            "--root",
            str(root),
            "--output-dir",
            str(output),
            "--transfer-active",
            "--source-checkout",
            str(checkout),
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["ok"] is True
    assert payload["snapshot_complete"] is False
    assert payload["training_eligible"] is False
    assert payload["hardware_authorized"] is False
