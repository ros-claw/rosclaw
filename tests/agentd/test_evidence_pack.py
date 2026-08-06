"""T6（二次复核 R7）：证据包验证——secret corpus、签名、缺失检测。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.agentd.bench.evidence_levels import EvidenceLevel
from rosclaw.agentd.bench.evidence_pack import EvidencePackError, EvidencePackWriter
from rosclaw.evidence_verify import verify_pack


def _make_pack(root: Path) -> Path:
    pack = EvidencePackWriter(root, run_id="run_test")
    pack.write_environment(provider="mock")
    pack.write_commands(["pytest tests/x"])
    pack.write_events([{"type": "turn.accepted", "payload": {"text": "hi"}}])
    pack.write_mission_snapshot({"mission_id": "mis_x"})
    pack.write_public_records(approvals=[{"request_id": "r1"}], permits=[], receipts=[])
    pack.write_metrics({"passed": True})
    pack.write_observer("SIM 验收。")
    pack.finalize(
        level=EvidenceLevel.E3_SIM_VERIFIED,
        git_commit="abc123",
        dirty=False,
        test_ids=["T6"],
        operator="test",
    )
    return pack.dir


class TestSecretCorpus:
    @pytest.mark.parametrize(
        "material",
        [
            "sk-proj-abc123",
            "sk-ant-api03-xyz",
            "ghp_abcdefghijklmnop",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig",
            "-----BEGIN RSA PRIVATE KEY-----",
            "Bearer abc.def.ghi",
        ],
    )
    def test_secret_corpus_fails_finalize(self, tmp_path: Path, material: str) -> None:
        pack = EvidencePackWriter(tmp_path)
        pack.write_events([{"payload": {"note": material}}])
        with pytest.raises(EvidencePackError, match="secret"):
            pack.finalize(
                level=EvidenceLevel.E3_SIM_VERIFIED,
                git_commit="x",
                dirty=False,
                test_ids=["T6"],
                operator="test",
            )
        # 失败也必须在盘上留下 INVALID 标记（不是静默消失）。
        manifest = json.loads((pack.dir / "run_manifest.json").read_text())
        assert manifest["invalid"] is True
        assert manifest["reason"] == "secret_scan_findings"


class TestPackVerifier:
    def test_valid_pack_verifies(self, tmp_path: Path) -> None:
        pack_dir = _make_pack(tmp_path)
        report = verify_pack(pack_dir)
        assert report["verified"]
        assert report["secret_scan_clean"]

    def test_missing_artifact_fails(self, tmp_path: Path) -> None:
        pack_dir = _make_pack(tmp_path)
        (pack_dir / "events.jsonl").unlink()
        with pytest.raises(SystemExit) as exit_info:
            verify_pack(pack_dir)
        assert exit_info.value.code == 2

    def test_tampered_artifact_fails(self, tmp_path: Path) -> None:
        pack_dir = _make_pack(tmp_path)
        (pack_dir / "events.jsonl").write_text('{"tampered": true}\n')
        with pytest.raises(SystemExit) as exit_info:
            verify_pack(pack_dir)
        assert exit_info.value.code == 2

    def test_observer_labeled_automated(self, tmp_path: Path) -> None:
        pack_dir = _make_pack(tmp_path)
        text = (pack_dir / "automated_observation.md").read_text()
        assert "automated_observation" in text
        assert not (pack_dir / "operator_observer.md").exists()
