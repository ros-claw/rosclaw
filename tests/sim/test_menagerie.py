"""Menagerie 正式接入测试（MH16，0916 优化 §二十四，红→绿）。

纪律：
- 官方 `mujoco_menagerie` package（锁版本），**绝不自动最新版
  下载**——ModelReference 记录 provider/package_version/
  model_revision/asset_digest/license（§24.2）；
- Menagerie 模型 ≠ 能力声明（§24.3）——导入默认
  capability=UNDECLARED（声明→证明绑定走 MH9 既有机制）；
- 内容寻址幂等：同名导入同 model_ref。
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco_menagerie", reason="mujoco-menagerie not installed")

#: 小模型快速夹具（MIT，0.8MB 下载）。
SMALL_MODEL = "dynamixel_2r"


@pytest.fixture
def backend(tmp_path):
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    return MujocoBackend(tmp_path)


def test_menagerie_load_records_full_provenance(backend) -> None:
    """§24.2：provenance 五元组全部记录（provider/version/revision/
    digest/license）。"""
    import mujoco_menagerie as mm

    ref = backend.load_menagerie(SMALL_MODEL)
    manifest = backend.store.get(ref.model_ref)
    source = manifest["source"]
    assert source["kind"] == "menagerie"
    assert source["provider"] == "menagerie"
    assert source["package_version"] == mm.__version__
    assert source["model_name"] == SMALL_MODEL
    assert source["model_revision"]  # git oid
    assert source["asset_digest"].startswith("sha256:") or len(source["asset_digest"]) == 64
    assert source["license"]
    assert source["entry_point"]


def test_menagerie_idempotent_same_ref(backend) -> None:
    """内容寻址：同名重复导入同 model_ref（幂等，不重下）。"""
    ref1 = backend.load_menagerie(SMALL_MODEL)
    ref2 = backend.load_menagerie(SMALL_MODEL)
    assert ref1.model_ref == ref2.model_ref


def test_menagerie_compile_and_inspect(backend) -> None:
    """导入即可用：编译冒烟 + inspect 结构化（nq/nv/nu 真相）。"""
    ref = backend.load_menagerie(SMALL_MODEL)
    inspected = backend.inspect_model(ref.model_ref)
    assert inspected.nq > 0 and inspected.nv > 0
    assert inspected.nu == 2  # dynamixel_2r 双关节夹爪臂


def test_menagerie_capability_undeclared(backend) -> None:
    """§24.3：Menagerie 模型不自动获得能力语义——gripper
    capability 必须 UNDECLARED（不是 AVAILABLE 也不是 UNPROVEN）。"""
    from rosclaw.sim.world.capability import resolve_grasp_capability

    ref = backend.load_menagerie(SMALL_MODEL)
    capability = resolve_grasp_capability(
        backend, {"id": "bot", "kind": "task", "ref": "menagerie_bot.xml"}, model_ref=ref.model_ref
    )
    assert capability["status"] == "UNDECLARED"


def test_menagerie_version_pinned() -> None:
    """§24.2 锁版本：环境里的 menagerie 版本必须与 pyproject
    pin 一致（防"自动最新版"）。"""
    import re
    from pathlib import Path

    import mujoco_menagerie as mm
    from packaging.version import Version

    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    match = re.search(r'mujoco-menagerie>=([0-9.]+),<([0-9.]+)', pyproject)
    assert match, "pyproject 缺 menagerie pin"
    # 日历版本必须语义比较（'2026.10' < '2026.9.0' 字典序是坑）。
    assert Version(match.group(1)) <= Version(mm.__version__) < Version(match.group(2))


def test_menagerie_unknown_model_fail_closed(backend) -> None:
    with pytest.raises(ValueError, match="MODEL_NOT_FOUND"):
        backend.load_menagerie("no_such_robot_xyz")


def test_menagerie_scaffold_generation(backend, tmp_path) -> None:
    """§24.3 scaffold：生成 e-URDF 声明脚手架——模型复制 +
    capabilities.yaml（UNDECLARED 标记）+ semantic.yaml 骨架。"""
    out = backend.scaffold_eurdf_from_menagerie(SMALL_MODEL, tmp_path / "eurdf_scaffold")
    assert (out / "capabilities.yaml").is_file()
    text = (out / "capabilities.yaml").read_text(encoding="utf-8")
    assert "UNDECLARED" in text
    assert (out / "semantic.yaml").is_file()
    # scaffold 里的模型就是 menagerie 原件（不改编译内容）。
    assert any(p.suffix == ".xml" for p in out.rglob("*.xml"))
