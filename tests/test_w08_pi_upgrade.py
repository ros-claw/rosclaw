"""W08 红测试（规格 2026-09-08 §12）：Pi 升级候选纪律。

- pin 单源一致：package.json deps/overrides 与
  pi-upstream.lock.json 的 package_version 一致（升级只走
  candidate 流程同时改两边）；
- build-info 不写死 pi 版本（从锁定记录读——§12.1-6 manifest
  记录有效运行版本）；
- 补丁在钉版上可重放（node_modules 缺失时 NOT_RUN 跳过——
  不合成冒充）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT = REPO_ROOT / "packages" / "rosclaw-agent"


class TestPinSingleSource:
    def test_package_json_matches_upstream_lock(self) -> None:
        pkg = json.loads((AGENT / "package.json").read_text())
        upstream = json.loads((AGENT / "pi-upstream.lock.json").read_text())
        pin = upstream["package_version"]
        for section in ("dependencies", "overrides"):
            for name, ver in (pkg.get(section) or {}).items():
                if name.startswith("@earendil-works/pi-"):
                    assert ver == pin, (
                        f"{section}.{name}={ver} 与 pi-upstream.lock "
                        f"{pin} 不一致——升级必须走 candidate 流程"
                    )

    def test_lockfile_digest_matches_record(self) -> None:
        """pi-upstream.lock.json 的 package_lock_sha256 = 实际
        package-lock.json 摘要（漂移即发现）。"""
        import hashlib

        upstream = json.loads((AGENT / "pi-upstream.lock.json").read_text())
        actual = hashlib.sha256(
            (AGENT / "package-lock.json").read_bytes()
        ).hexdigest()
        assert upstream["package_lock_sha256"] == actual, (
            "lock 漂移——重跑 pi_upgrade_candidate.sh 重生成记录"
        )


class TestBuildInfoDynamic:
    def test_build_release_no_hardcoded_pi_version(self) -> None:
        text = (REPO_ROOT / "scripts" / "build_release.sh").read_text(
            encoding="utf-8"
        )
        assert not re.search(r'"pi_version":\s*"0\.', text), (
            "build-info 写死 pi 版本——必须从 pi-upstream.lock.json 读"
        )
        assert "pi-upstream.lock.json" in text


class TestPatchReplayOnPin:
    def test_patches_apply_on_current_pin(self) -> None:
        """补丁在钉版上幂等可重放（node_modules 不在场 = NOT_RUN
        跳过，不合成冒充）。"""
        import shutil
        import subprocess

        node = shutil.which("node")
        nm = AGENT / "node_modules"
        if not node or not (nm / "@earendil-works").exists():
            import pytest

            pytest.skip("NOT_RUN: node/node_modules 不在场")
        result = subprocess.run(
            [node, "patches/apply-upstream-patches.mjs"],
            cwd=AGENT, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr[-300:]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
