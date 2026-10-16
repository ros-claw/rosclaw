"""Inert boundary regressions: software contracts, never native GL evidence."""


def _modules():
    from benchmarks.harnessbench import vision_oracle
    from rosclaw.sim.backends.mujoco import backend

    return backend, vision_oracle


def _producer(monkeypatch, platform, camera, fail_first):
    import json
    import subprocess
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    backend, _ = _modules()
    calls = []
    stored = []
    instance = backend.MujocoBackend.__new__(backend.MujocoBackend)
    instance.store = SimpleNamespace(
        put=lambda partition, blob: stored.append(blob) or "render-ref"
    )
    instance._load_assets = lambda manifest: {}
    monkeypatch.setattr(backend.state_v2, "capture_state_v2", lambda m, d: (np.array([3.0]), 7))

    def run(argv, **kwargs):
        assert kwargs == {"capture_output": True, "text": True, "timeout": 120}
        assert argv[2] == (backend._CAMERA_WORKER_CODE if camera else backend._RENDER_WORKER_CODE)
        calls.append(argv[3])
        request = json.loads(Path(argv[4]).read_text())
        assert request["width"] == 8 and request["height"] == 6
        if fail_first and len(calls) == 1:
            return SimpleNamespace(returncode=1, stderr="inert egl failure")
        if camera:
            assert request["state_vector"] == [3.0] and request["state_spec"] == 7
            assert request["camera"] == "fixed"
            Path(request["metadata_out"]).write_text('{"artifact_semantics":"color_rgb"}')
        else:
            assert request["states"] == [{"t": 0.0}]
        Path(request["out"]).write_bytes(b"inert pixels")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("MUJOCO_GL", "foreign")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "foreign")
    manifest = {"mjcf_xml": "<mujoco/>", "assets": {}}
    if camera:
        data = SimpleNamespace(time=0.0, qpos=[], qvel=[], ctrl=[])
        result = instance._render_camera_frame_subprocess(
            manifest, object(), data, kind="camera_rgb", camera_name="fixed", width=8, height=6
        )
        assert result[1:] == ("render-ref", "uint8", {"artifact_semantics": "color_rgb"})
        assert stored == [b"inert pixels"]
        selected = result[0]
    else:
        blob, selected = instance._render_gif_subprocess(
            manifest, [{"t": 0.0}], camera="fixed", width=8, height=6
        )
        assert blob == b"inert pixels"
    return calls, selected


def test_darwin_camera_actual_producer(monkeypatch):
    assert _producer(monkeypatch, "darwin", True, False) == (["cgl"], "cgl")


def test_darwin_gif_actual_producer(monkeypatch):
    assert _producer(monkeypatch, "darwin", False, False) == (["cgl"], "cgl")


def test_linux_actual_producers_keep_fallback_order(monkeypatch):
    for camera in (True, False):
        with monkeypatch.context() as scoped:
            assert _producer(scoped, "linux", camera, True) == (["egl", "osmesa"], "osmesa")


def _oracle(monkeypatch, platform, renderer, expected_plugin):
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    _, oracle = _modules()
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[2] == oracle._WORKER
        env = kwargs["env"]
        assert env["MUJOCO_GL"] == renderer
        assert env["PYOPENGL_PLATFORM"] == expected_plugin
        assert env["ROSCLAW_INERT_MARKER"] == "preserved"
        assert kwargs["capture_output"] is True and kwargs["timeout"] == 60
        request = json.loads(Path(argv[3]).read_text())
        assert request["vector"] == [1.0] and request["camera"] == "fixed"
        np.savez(request["out"], segmentation=np.array([[[2, 5]]]), depth=np.array([[1.5]]))
        Path(request["metadata"]).write_text('{"geom_id":2}')
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(oracle.subprocess, "run", run)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("MUJOCO_GL", "osmesa" if renderer != "osmesa" else "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl" if expected_plugin != "egl" else "darwin")
    monkeypatch.setenv("ROSCLAW_INERT_MARKER", "preserved")
    pixels, metadata = oracle._independent_pixels(
        "<mujoco/>", np.array([1.0]), "fixed", "target", renderer
    )
    assert len(calls) == 1
    assert pixels["segmentation"].tolist() == [[[2, 5]]]
    assert pixels["depth"].tolist() == [[1.5]]
    assert metadata == {"geom_id": 2}


def test_darwin_actual_oracle_pins_cgl_darwin(monkeypatch):
    assert _oracle(monkeypatch, "darwin", "cgl", "darwin") is None


def test_linux_actual_oracle_pins_each_declared_renderer(monkeypatch):
    for renderer in ("egl", "osmesa"):
        with monkeypatch.context() as scoped:
            assert _oracle(scoped, "linux", renderer, renderer) is None


def test_oracle_rejects_cross_platform_and_unknown_bindings(monkeypatch):
    import sys

    import numpy as np
    import pytest

    _, oracle = _modules()

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid renderer reached subprocess")

    monkeypatch.setattr(oracle.subprocess, "run", forbidden)
    for platform, renderers in (
        ("darwin", ("egl", "osmesa", "auto", "darwin")),
        ("linux", ("cgl", "auto")),
    ):
        monkeypatch.setattr(sys, "platform", platform)
        for renderer in renderers:
            with pytest.raises(ValueError, match="CAMERA_RENDERER_BINDING_INVALID") as caught:
                oracle._independent_pixels("<mujoco/>", np.array([]), "fixed", "target", renderer)
            assert str(caught.value) == "CAMERA_RENDERER_BINDING_INVALID"


def _worker_boundary(monkeypatch, worker, platform, renderer, plugin):
    import builtins
    import os
    import sys

    import pytest

    original_import = builtins.__import__
    reached = []

    class EngineBoundaryError(Exception):
        pass

    def boundary(name, *args, **kwargs):
        if name == "mujoco":
            reached.append(name)
            assert os.environ["MUJOCO_GL"] == renderer
            assert os.environ["PYOPENGL_PLATFORM"] == plugin
            raise EngineBoundaryError
        assert not name.startswith("OpenGL")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "argv", ["inert", renderer, "not-opened"])
    monkeypatch.setenv("MUJOCO_GL", "foreign")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "foreign")
    monkeypatch.setattr(builtins, "__import__", boundary)
    with pytest.raises(EngineBoundaryError):
        exec(worker, {})
    return reached


def test_actual_workers_override_environment_before_engine_import(monkeypatch):
    backend, _ = _modules()
    for worker in (backend._CAMERA_WORKER_CODE, backend._RENDER_WORKER_CODE):
        for platform, renderer, plugin in (
            ("darwin", "cgl", "darwin"),
            ("linux", "egl", "egl"),
            ("linux", "osmesa", "osmesa"),
        ):
            with monkeypatch.context() as scoped:
                assert _worker_boundary(scoped, worker, platform, renderer, plugin) == ["mujoco"]


def _durability():
    import runpy
    from pathlib import Path

    return runpy.run_path(str(Path(__file__).with_name("test_store_durability.py")))


def test_portable_observer_real_file_and_directory_descriptors(tmp_path):
    import os

    helper = _durability()["_portable_native_fd_path"]
    target = tmp_path / "evidence"
    target.write_bytes(b"observed")
    for path in (target, tmp_path):
        fd = os.open(path, os.O_RDONLY)
        try:
            observed = helper(fd)
            assert os.path.samefile(observed, path)
            assert os.stat(observed).st_ino == os.fstat(fd).st_ino
        finally:
            os.close(fd)


def test_portable_darwin_f_getpath_boundary_and_errors(monkeypatch):
    import fcntl
    import sys

    import pytest

    helper = _durability()["_portable_native_fd_path"]
    calls = []

    def getpath(fd, command, buffer):
        calls.append((fd, command, buffer))
        return b"/private/tmp/descriptor-evidence\0" + b"\0" * 20

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(fcntl, "fcntl", getpath)
    assert helper(41) == "/private/tmp/descriptor-evidence"
    assert calls == [(41, 50, b"\0" * 1024)]

    def failure(*args):
        raise OSError("inert F_GETPATH failure")

    monkeypatch.setattr(fcntl, "fcntl", failure)
    with pytest.raises(OSError, match="inert F_GETPATH failure"):
        helper(41)


def test_original_targeted_injection_and_ancestor_retry(tmp_path, monkeypatch):
    original = _durability()
    (tmp_path / "targeted").mkdir()
    (tmp_path / "ancestors").mkdir()
    original["test_io_failure_never_acknowledges_and_retry_preserves_evidence"](
        tmp_path / "targeted", monkeypatch, "directory_sync"
    )
    with monkeypatch.context() as scoped:
        original["test_deep_directory_creation_retry_retains_original_ancestor_barriers"](
            tmp_path / "ancestors", scoped
        )
    assert list((tmp_path / "ancestors").rglob("simmdl_*.json"))
