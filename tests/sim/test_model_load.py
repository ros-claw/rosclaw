"""MujocoBackend load_model 测试（PR-MH2，ADR-0014，规格 §12.1，红→绿）。"""

from __future__ import annotations

import pytest

from rosclaw.sim.backends.mujoco.backend import MujocoBackend
from rosclaw.sim.contracts import ModelReference


def test_load_task_local_mjcf_returns_typed_ref(tiny_task_root) -> None:
    backend = MujocoBackend(tiny_task_root)
    ref = backend.load_model("arm.xml")

    assert isinstance(ref, ModelReference)
    assert ref.model_ref.startswith("simmdl_")
    assert ref.model_digest.startswith("sha256:")
    assert ref.backend == "mujoco"
    assert ref.compiled is True
    assert ref.created_at  # 来自不可变 manifest 的创建时间
    assert ref.source["kind"] == "task"

    body = ref.body_description
    assert body["nq"] == 2 and body["nv"] == 2 and body["nu"] == 2
    assert body["sites"] == ["tool0"]
    assert isinstance(ref.compile_warnings, list)
    assert ref.runtime_capabilities["backend"] == "mujoco"


def test_load_idempotent_same_content(tiny_task_root) -> None:
    backend = MujocoBackend(tiny_task_root)
    ref1 = backend.load_model("arm.xml")
    ref2 = backend.load_model("arm.xml")
    assert ref1.model_ref == ref2.model_ref
    assert len(backend.store.list_children("models")) == 1


def test_load_ur5e_zoo_integration(tmp_path) -> None:
    pytest.importorskip("mujoco")
    from rosclaw.runtime.eurdf_loader import _default_zoo_path

    if not (_default_zoo_path() / "ur5e" / "robot.mjcf.xml").exists():
        pytest.skip("e-urdf-zoo ur5e not available")

    backend = MujocoBackend(tmp_path)
    ref = backend.load_model("ur5e")
    assert ref.source["kind"] == "eurdf"
    assert ref.body_description["nq"] == 6

    # mesh 资产已捕获进 store；compile_model 可重建。
    inspection = backend.compile_model(ref.model_ref)
    assert inspection.nq == 6
    manifest = backend.store.get(ref.model_ref)
    assert len(manifest["assets"]) >= 20  # ur5e mesh 文件


def test_reject_external_uri(tiny_task_root) -> None:
    backend = MujocoBackend(tiny_task_root)
    for bad in ("https://evil.example/model.xml", "file:///etc/passwd"):
        with pytest.raises(ValueError, match="MODEL_NOT_FOUND"):
            backend.load_model(bad)


def test_reject_path_escape(tiny_task_root, tmp_path_factory) -> None:
    outside = tmp_path_factory.mktemp("outside") / "evil.xml"
    outside.write_text("<mujoco/>", encoding="utf-8")
    backend = MujocoBackend(tiny_task_root)
    with pytest.raises(ValueError, match="MODEL_PATH_ESCAPE"):
        backend.load_model(str(outside))
    with pytest.raises(ValueError, match="MODEL_PATH_ESCAPE"):
        backend.load_model("../outside/evil.xml")


def test_model_not_found(tiny_task_root) -> None:
    backend = MujocoBackend(tiny_task_root)
    with pytest.raises(ValueError, match="MODEL_NOT_FOUND"):
        backend.load_model("ghost.xml")


def test_compile_failure_fail_closed(tiny_task_root) -> None:
    (tiny_task_root / "broken.xml").write_text(
        "<mujoco><worldbody><geom/></worldbody>", encoding="utf-8"
    )
    backend = MujocoBackend(tiny_task_root)
    with pytest.raises(ValueError, match="MODEL_COMPILE_FAILED"):
        backend.load_model("broken.xml")
    # store 无残留。
    assert backend.store.list_children("models") == []
