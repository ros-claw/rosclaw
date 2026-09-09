"""W01 §5.3 防回归：内置签名 Skill 的 manifest digest 必须与
SKILL.md 内容一致——R1-2d 改 Skill 未更新签名导致 digest 不符被
fail-closed 排除（rosclaw-embodied 从模型面消失，n2 在 main 上红）。

闭环断言：verifyBundledSkills 对每个 manifest 条目给 verified
（零 excluded）。
"""

from __future__ import annotations

import pytest


def test_bundled_skill_manifest_digests_match() -> None:
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    skills_dir = repo / "packages/rosclaw-agent" / "skills"
    script = (
        "import { verifyBundledSkills } from './src/extension/bundled-skills.ts';"
        f"const r = verifyBundledSkills({str(skills_dir)!r});"
        "console.log(JSON.stringify(r));"
    )
    result = subprocess.run(
        ["npx", "tsx", "-e", script],
        cwd=repo / "packages/rosclaw-agent",
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-300:]
    payload = __import__("json").loads(result.stdout)
    assert payload["verified"], payload
    assert not payload["excluded"], (
        f"Skill 被签名排除（manifest digest 与内容漂移）: {payload['excluded']}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
