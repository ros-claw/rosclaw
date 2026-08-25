"""P1-A6 红测试（0824 总纲 §10.1 收尾）：rosclaw-modeld 包移除。

P1-A5 已断 Python 侧全部引用；本闭环删除实体包与其构建/发布/
CI 配套——发布物里不再存在第二条模型运行时。

断言：
1. packages/rosclaw-modeld 不存在；
2. CI 无 node-modeld-unit job；
3. build_release.sh / install_release.sh 不再构建/安装 modeld；
4. release packaging 测试事实源（bundle 内容/SBOM/vendor 包）不含
   modeld；
5. pna10/dependency-boundary 不再有 modeld 泳道。
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_modeld_package_gone() -> None:
    assert not (REPO / "packages" / "rosclaw-modeld").exists()


def test_ci_has_no_modeld_job() -> None:
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "node-modeld-unit" not in ci
    assert "packages/rosclaw-modeld" not in ci


def test_release_scripts_skip_modeld() -> None:
    for script in ("scripts/build_release.sh", "scripts/release/install_release.sh"):
        text = (REPO / script).read_text(encoding="utf-8")
        assert "rosclaw-modeld" not in text, f"{script} 仍引用 modeld"


def test_packaging_tests_skip_modeld() -> None:
    for test in (
        "tests/agentd/test_release_packaging.py",
        "tests/agentd/test_pna10_release.py",
        "tests/architecture/test_pi_dependency_boundary.py",
    ):
        text = (REPO / test).read_text(encoding="utf-8")
        assert "rosclaw-modeld" not in text, f"{test} 仍引用 modeld"
