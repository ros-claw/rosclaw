"""Tests for packaging and upload."""

import json
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from rosclaw.skill.cli import _copy_template, _init_context
from rosclaw.skill.eval import evaluate_skill
from rosclaw.skill.hash import compute_skill_hashes
from rosclaw.skill.mining import mine_skill_candidate
from rosclaw.skill.models import SkillPackage
from rosclaw.skill.package import (
    package_skill,
    scan_forbidden_content,
    verify_package,
)
from rosclaw.skill.promote import promote_candidate
from rosclaw.skill.upload import build_hub_payload, upload_skill
from rosclaw.skill.validators import validate_package
from tests.skill.evidence_helpers import write_promotion_evidence


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


TEMPLATE_DIR = (
    Path(__file__).parent.parent.parent / "src" / "rosclaw" / "skill" / "templates" / "default"
)
FIXTURES = Path(__file__).parent / "fixtures" / "practice_sessions"


def _validated_pkg(tmp_path: Path):
    dest = tmp_path / "g1_kick_ball"
    context = _init_context("g1_kick_ball", "ur5e", "manipulation", "ros-claw")
    _copy_template(TEMPLATE_DIR, dest, context)
    pkg = SkillPackage(dest).try_load()
    mine_skill_candidate(pkg, FIXTURES, candidate_id="candidate_0001")
    pkg = SkillPackage(dest).try_load()
    write_promotion_evidence(pkg, "candidate_0001")
    pkg = SkillPackage(dest).try_load()
    evaluate_skill(pkg, candidate_id="candidate_0001", mode="replay")
    pkg = SkillPackage(dest).try_load()
    promote_candidate(pkg, "candidate_0001", "0.1.0")
    return SkillPackage(dest).try_load()


def test_hash_stability(tmp_path: Path):
    pkg = _validated_pkg(tmp_path)
    h1 = compute_skill_hashes(pkg.root, include_evidence=False)
    h2 = compute_skill_hashes(pkg.root, include_evidence=False)
    assert h1["package_hash"] == h2["package_hash"]


def test_promoted_package_hashes_and_snapshot_match_final_state(tmp_path: Path):
    pkg = _validated_pkg(tmp_path)
    validation = validate_package(pkg)
    assert validation.checks["package_integrity"] is True

    snapshot = SkillPackage(pkg.root / ".rosclaw" / "snapshots" / "0.1.0").try_load()
    assert snapshot.skill.metadata.stage == "validated"
    assert snapshot.skill.metadata.version == "0.1.0"


def test_package_creates_archive(tmp_path: Path):
    pkg = _validated_pkg(tmp_path)
    archive = package_skill(pkg, output_dir=tmp_path / "dist")
    assert archive.exists()
    assert archive.name == "g1_kick_ball-0.1.0.tar.gz"


def test_verify_package_passes(tmp_path: Path):
    pkg = _validated_pkg(tmp_path)
    archive = package_skill(pkg, output_dir=tmp_path / "dist")
    result = verify_package(archive)
    assert result["ok"] is True


def test_secret_scan_detects_key(tmp_path: Path):
    pkg = _validated_pkg(tmp_path)
    (pkg.root / "leak.txt").write_text("OPENAI_API_KEY=sk-1234567890abcdef", encoding="utf-8")
    secrets, _ = scan_forbidden_content(pkg.root)
    assert any("OPENAI_API_KEY" in s for s in secrets)


def test_build_hub_payload(tmp_path: Path):
    pkg = _validated_pkg(tmp_path)
    payload = build_hub_payload(pkg, visibility="private")
    assert payload["name"] == "ros-claw/g1_kick_ball"
    assert payload["version"] == "0.1.0"
    assert payload["category"] == "manipulation"
    assert "ur5e" in payload["compatible_robots"]
    assert "manipulator" in payload["robot_types"]


class FakeHubClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.base_url = base_url
        self.api_key = api_key

    def create_skill(self, payload: dict) -> dict:
        self.calls.append(("POST", "/api/skills", payload))
        return {"status_code": 201, "id": "uuid-1"}

    def update_skill(self, name: str, payload: dict) -> dict:
        self.calls.append(("PUT", f"/api/skills/{name}", payload))
        return {"status_code": 200, "id": "uuid-1"}

    def get_skill(self, name: str) -> dict | None:
        return None


def test_upload_dry_run(tmp_path: Path, monkeypatch: MonkeyPatch):
    pkg = _validated_pkg(tmp_path)
    monkeypatch.setenv("ROSCLAW_ADMIN_API_KEY", "test-key")
    result = upload_skill(pkg, dry_run=True)
    assert result["dry_run"] is True
    assert result["payload"]["name"] == "ros-claw/g1_kick_ball"


def test_upload_receipt_written(tmp_path: Path, monkeypatch: MonkeyPatch):
    pkg = _validated_pkg(tmp_path)
    monkeypatch.setenv("ROSCLAW_ADMIN_API_KEY", "test-key")
    monkeypatch.setattr("rosclaw.skill.upload.SkillHubClient", FakeHubClient)
    monkeypatch.setattr("rosclaw.skill.upload._verify_repo_url", lambda _url: True)
    result = upload_skill(pkg)
    assert result["ok"] is True
    receipt_path = pkg.root / ".rosclaw" / "upload_receipt.json"
    assert receipt_path.exists()
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert data["skill_name"] == "ros-claw/g1_kick_ball"


def test_upload_missing_key(tmp_path: Path):
    pkg = _validated_pkg(tmp_path)
    with pytest.raises(RuntimeError, match="Missing API key"):
        upload_skill(pkg)
