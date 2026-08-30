"""P0-5 红测试（0823 审计 §三.P0-5）：安装一致性诊断。

红测试先行——version_diag 模块不存在时必须红。

0823 事故：报告里声明的能力在产品里 OUTPUT_SCHEMA_MISSING——
实现与部署漂移而不可察觉。`rosclaw version --diagnostic --json`
必须报告安装的真实构成：wheel commit / TS dist digest / extension
digest / kit digest / migration revision / e-URDF 内容代数；
两侧 commit 不一致 → INSTALLATION_VERSION_MISMATCH（chat 阻断）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestCollectDiagnostics:
    def test_required_fields_present(self, tmp_path: Path) -> None:
        from rosclaw.version_diag import collect_diagnostics

        diag = collect_diagnostics(home=tmp_path)
        for key in (
            "rosclaw_version", "wheel_commit", "ts_dist", "extension_digest",
            "kit_digest", "migration_revision", "eurdf_generation",
            "mixed_build",
        ):
            assert key in diag, f"诊断缺字段: {key}"
        assert diag["rosclaw_version"], "版本号为空"
        ts = diag["ts_dist"]
        assert "entry" in ts and "commit" in ts and "dist_digest" in ts
        assert isinstance(diag["migration_revision"], int)
        assert diag["migration_revision"] >= 31, "migration 代数漂移"

    def test_mixed_build_detected(self, tmp_path: Path) -> None:
        from rosclaw.version_diag import mixed_build_reason

        diag = {
            "wheel_commit": "abc123",
            "ts_dist": {"commit": "def456", "entry": "x", "dist_digest": "y"},
        }
        reason = mixed_build_reason(diag)
        assert reason is not None and "INSTALLATION_VERSION_MISMATCH" in reason

    def test_matching_commits_not_mixed(self) -> None:
        from rosclaw.version_diag import mixed_build_reason

        diag = {
            "wheel_commit": "abc123",
            "ts_dist": {"commit": "abc123", "entry": "x", "dist_digest": "y"},
        }
        assert mixed_build_reason(diag) is None

    def test_unknown_commit_not_mixed(self) -> None:
        """commit 不可知（开发环境无 stamp）不谎报混合——只报 unknown。"""
        from rosclaw.version_diag import mixed_build_reason

        diag = {
            "wheel_commit": "unknown",
            "ts_dist": {"commit": "unknown", "entry": "x", "dist_digest": "y"},
        }
        assert mixed_build_reason(diag) is None

    def test_assert_coherent_raises_on_mixed(self) -> None:
        from rosclaw.version_diag import assert_installation_coherent

        with pytest.raises(SystemExit) as exc:
            assert_installation_coherent({
                "wheel_commit": "aaa",
                "ts_dist": {"commit": "bbb", "entry": "x", "dist_digest": "y"},
            })
        assert "INSTALLATION_VERSION_MISMATCH" in str(exc.value)


class TestVersionCli:
    def test_version_diagnostic_json(self, tmp_path: Path, capsys) -> None:
        from rosclaw.version_diag import cmd_version

        rc = cmd_version(diagnostic=True, as_json=True, home=tmp_path)
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["rosclaw_version"]
        assert "mixed_build" in payload

    def test_kit_digest_stable_and_sensitive(self, tmp_path: Path) -> None:
        from rosclaw.version_diag import kit_digest

        first = kit_digest()
        assert first.startswith("sha256:")
        assert kit_digest() == first, "kit digest 不稳定"

    def test_eurdf_generation_is_content_addressed(self) -> None:
        from rosclaw.version_diag import eurdf_generation

        gen = eurdf_generation()
        assert gen and gen != "unknown", "e-URDF 内容代数不可知"
