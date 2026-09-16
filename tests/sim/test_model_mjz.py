"""便携模型工件 .mjz 测试（MH10b，0916 优化 §4.3，红→绿）。"""

from __future__ import annotations


def test_export_mjz_roundtrip(tmp_path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("mujoco")
    from rosclaw.runtime.eurdf_loader import _default_zoo_path

    if not (_default_zoo_path() / "ur5e" / "robot.mjcf.xml").exists():
        pytest.skip("e-urdf-zoo ur5e not available")

    import mujoco

    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    backend = MujocoBackend(tmp_path)
    ref = backend.load_model("ur5e")

    result = backend.export_model_mjz(ref.model_ref)
    assert result["format"] == "mjz"
    assert result["artifact_ref"].startswith("simmdl_")
    assert result["size_bytes"] > 1_000_000  # 31MB mesh 已嵌入
    assert result["model_digest"] == ref.model_digest

    # from_zip 自包含编译（不依赖原文件/资产目录）。
    blob = backend.store.get(result["artifact_ref"])
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mjz", delete=False) as tmp:
        tmp.write(blob)
        spec = mujoco.MjSpec.from_zip(tmp.name)
    model = spec.compile()
    assert model.nq == 6
    assert model.nmesh == 20


def test_export_mjz_idempotent(tmp_path) -> None:
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    (tmp_path / "arm.xml").write_text(
        """<mujoco><worldbody><body name="b" pos="0 0 0.5">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="g" type="capsule" size="0.05 0.2" mass="1"/>
    </body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    backend = MujocoBackend(tmp_path)
    ref = backend.load_model("arm.xml")
    r1 = backend.export_model_mjz(ref.model_ref)
    r2 = backend.export_model_mjz(ref.model_ref)
    assert r1["artifact_ref"] == r2["artifact_ref"]
