from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from rosclaw import extension_discovery
from rosclaw.dataset import (
    DATASET_SOURCE_GROUP,
    DatasetSnapshotState,
    DatasetSourceDescriptor,
    DatasetSourceRegistry,
    FileHashMode,
    inspect_dataset_root,
    write_dataset_doctor_artifacts,
)
from rosclaw.dataset.cli import dispatch_dataset_argv


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


@dataclass
class _EntryPoint:
    name: str
    group: str
    value: Any
    error: Exception | None = None

    def load(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.value


class _EntryPoints(list[_EntryPoint]):
    def select(self, *, group: str) -> _EntryPoints:
        return _EntryPoints(item for item in self if item.group == group)


@dataclass
class _NameSource:
    descriptor: DatasetSourceDescriptor
    rules: tuple[tuple[str, str], ...]
    observed_inputs: list[tuple[str, str]] = field(default_factory=list)

    def classify_file(self, dataset_id: str, relative_path: str) -> tuple[str, ...]:
        self.observed_inputs.append((dataset_id, relative_path))
        lowered = relative_path.lower()
        return tuple(label for token, label in self.rules if token in lowered)


def _source(
    *,
    source_id: str,
    dataset_id: str,
    labels: tuple[str, ...],
    rules: tuple[tuple[str, str], ...],
) -> _NameSource:
    return _NameSource(
        descriptor=DatasetSourceDescriptor(
            source_id=source_id,
            dataset_ids=(dataset_id,),
            label_ids=labels,
            source_uri=f"https://datasets.example/{source_id}",
            revision="fixture-v1",
            manifest_hash=_hash(source_id),
        ),
        rules=rules,
    )


def test_registry_discovers_two_task_domains_without_runtime_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "navigation",
                DATASET_SOURCE_GROUP,
                _source(
                    source_id="navigation.corridors",
                    dataset_id="Nav Set",
                    labels=("corridor", "turn"),
                    rules=(("corridor", "corridor"), ("turn", "turn")),
                ),
            ),
            _EntryPoint(
                "manipulation",
                DATASET_SOURCE_GROUP,
                _source(
                    source_id="manipulation.tabletop",
                    dataset_id="ArmSet",
                    labels=("grasp", "release"),
                    rules=(("grasp", "grasp"), ("release", "release")),
                ),
            ),
        ]
    )
    monkeypatch.setattr(extension_discovery.importlib.metadata, "entry_points", lambda: entries)
    registry = DatasetSourceRegistry()

    report = registry.discover()

    assert registry.source_ids == (
        "manipulation.tabletop",
        "navigation.corridors",
    )
    assert report.loaded == registry.source_ids
    assert report.errors == ()


def test_registry_isolates_broken_plugins_and_out_of_vocabulary_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy = _source(
        source_id="navigation.corridors",
        dataset_id="NavSet",
        labels=("corridor",),
        rules=(("corridor", "corridor"),),
    )
    invalid = _source(
        source_id="navigation.invalid",
        dataset_id="NavSet",
        labels=("turn",),
        rules=(("corridor", "undeclared"),),
    )
    entries = _EntryPoints(
        [
            _EntryPoint("broken", DATASET_SOURCE_GROUP, None, RuntimeError("bad package")),
            _EntryPoint("healthy", DATASET_SOURCE_GROUP, healthy),
            _EntryPoint("invalid", DATASET_SOURCE_GROUP, invalid),
        ]
    )
    monkeypatch.setattr(extension_discovery.importlib.metadata, "entry_points", lambda: entries)
    registry = DatasetSourceRegistry()

    discovery = registry.discover()
    resolution = registry.classify(dataset_id="NavSet", relative_path="corridor/001.json")

    assert discovery.loaded == ("navigation.corridors", "navigation.invalid")
    assert discovery.errors == ("broken: bad package",)
    assert tuple(value.source_id for value in resolution.annotations) == ("navigation.corridors",)
    assert resolution.errors == (
        "navigation.invalid: ValueError: classify_file returned a label outside its descriptor vocabulary",
    )


def test_dataset_source_discovery_has_operator_recovery_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROSCLAW_DISABLE_DATASET_EXTENSIONS", "true")
    monkeypatch.setattr(
        extension_discovery.importlib.metadata,
        "entry_points",
        lambda: pytest.fail("disabled discovery must not inspect packages"),
    )

    report = DatasetSourceRegistry().discover()

    assert report.disabled is True


def test_cross_domain_doctor_inventory_is_generic_and_provenance_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    nav = root / "Nav Set"
    arm = root / "ArmSet"
    nav.mkdir(parents=True)
    arm.mkdir()
    (nav / "LICENSE").write_text("fixture license", encoding="utf-8")
    (nav / "corridor_turn_001.json").write_text("{}", encoding="utf-8")
    (arm / "grasp_001.json").write_text("{}", encoding="utf-8")
    (arm / "release_002.json").write_text("{}", encoding="utf-8")

    nav_source = _source(
        source_id="navigation.corridors",
        dataset_id="Nav Set",
        labels=("corridor", "turn"),
        rules=(("corridor", "corridor"), ("turn", "turn")),
    )
    arm_source = _source(
        source_id="manipulation.tabletop",
        dataset_id="ArmSet",
        labels=("grasp", "release"),
        rules=(("grasp", "grasp"), ("release", "release")),
    )
    registry = DatasetSourceRegistry()
    registry.register_source(nav_source)
    registry.register_source(arm_source)

    report = inspect_dataset_root(
        root,
        hash_mode=FileHashMode.SHA256,
        source_registry=registry,
        discover_sources=False,
    )

    inventories = {value.dataset_id: value for value in report.inventories}
    assert report.snapshot_complete is True
    assert tuple(value.source_id for value in report.sources) == (
        "manipulation.tabletop",
        "navigation.corridors",
    )
    assert inventories["Nav Set"].label_counts == {"corridor": 1, "turn": 1}
    assert inventories["ArmSet"].label_counts == {"grasp": 1, "release": 1}
    assert inventories["Nav Set"].license_evidence_present is True
    assert inventories["Nav Set"].training_eligible is False
    assert report.to_dict()["promotion_truth_allowed"] is False
    assert nav_source.observed_inputs == [
        ("Nav Set", "LICENSE"),
        ("Nav Set", "corridor_turn_001.json"),
    ]
    assert all(not Path(relative).is_absolute() for _, relative in nav_source.observed_inputs)

    artifacts = write_dataset_doctor_artifacts(report, tmp_path / "evidence")
    assert set(artifacts) == {
        "inventory",
        "quality_report",
        "license_manifest",
        "source_manifest",
        "label_matrix",
    }
    source_manifest = json.loads(Path(artifacts["source_manifest"]).read_text(encoding="utf-8"))
    assert source_manifest["training_eligible"] is False
    assert len(source_manifest["sources"]) == 2


def test_transfer_and_plugin_errors_fail_snapshot_closed(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = root / "NavSet"
    dataset.mkdir(parents=True)
    (dataset / "sample.json.part").write_text("partial", encoding="utf-8")
    invalid = _source(
        source_id="navigation.invalid",
        dataset_id="NavSet",
        labels=("turn",),
        rules=(("sample", "undeclared"),),
    )
    registry = DatasetSourceRegistry()
    registry.register_source(invalid)

    report = inspect_dataset_root(
        root,
        source_registry=registry,
        discover_sources=False,
    )
    inventory = report.inventories[0]

    assert inventory.state is DatasetSnapshotState.TRANSFERRING
    assert inventory.source_error_count == 1
    assert report.snapshot_complete is False


def test_sha256_inventory_detects_duplicate_content(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset = root / "GenericSet"
    dataset.mkdir(parents=True)
    (dataset / "a.bin").write_bytes(b"same")
    (dataset / "b.bin").write_bytes(b"same")

    report = inspect_dataset_root(
        root,
        hash_mode=FileHashMode.SHA256,
        discover_sources=False,
    )

    assert report.inventories[0].duplicate_content_groups == (("a.bin", "b.bin"),)


def test_cli_no_extension_smoke_writes_generic_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "datasets"
    dataset = root / "GenericSet"
    dataset.mkdir(parents=True)
    (dataset / "sample.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "output"

    status = dispatch_dataset_argv(
        [
            "dataset",
            "doctor",
            "--root",
            str(root),
            "--output-dir",
            str(output),
            "--source-checkout",
            str(Path(__file__).resolve().parents[2]),
            "--no-source-extensions",
        ]
    )
    receipt = json.loads(capsys.readouterr().out)

    assert status == 0
    assert receipt["ok"] is True
    assert receipt["source_ids"] == []
    assert receipt["hardware_authorized"] is False
    assert (output / "dataset_label_matrix.csv").is_file()


def test_registry_rejects_objects_without_source_contract() -> None:
    with pytest.raises(TypeError, match="DatasetSource protocol"):
        DatasetSourceRegistry().register_source(object())
