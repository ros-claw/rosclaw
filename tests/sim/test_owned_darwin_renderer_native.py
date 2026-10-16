"""Authored SOURCE-only tests. No production GL, CGL, engine or model calls.

The inherited methods are compiled from the pinned, read-only upstream source
with fake globals, not imported from the installed MuJoCo package. This lets
output assertions exercise upstream math rather than a second implementation.
Root must run these later; authoring does not constitute a passing test run.
"""

import __future__

import ast
import ctypes
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw.sim.backends.mujoco import owned_renderer as owned

EXPECTED_FORMATS = (
    (99, 4096, 8, 24, 11, 8, 12, 24, 13, 8, 59, 55, 1, 56, 4, 73, 0),
    (99, 4096, 8, 24, 11, 8, 12, 24, 13, 8, 73, 0),
    (99, 4096, 8, 24, 11, 8, 12, 24, 13, 8, 0),
)


class FakeCGL:
    def __init__(self, choices=None, creations=None):
        self.choices = list(choices or [(0, 1, 11)])
        self.creations = list(creations or [(0, 22)])
        self.events = []
        self.current = 99
        self.set_rc = 0
        self.lock_rc = 0
        self.unlock_rc = 0
        self.CGLGetCurrentContext = self.get_current_context
        self.CGLChoosePixelFormat = self.choose_pixel_format
        self.CGLCreateContext = self.create_context
        self.CGLSetCurrentContext = self.set_current_context
        self.CGLLockContext = self.lock_context
        self.CGLUnlockContext = self.unlock_context
        self.CGLReleaseContext = self.release_context
        self.CGLReleasePixelFormat = self.release_pixel_format

    @staticmethod
    def put(output, value, ctype):
        ctypes.cast(output, ctypes.POINTER(ctype))[0] = value

    @staticmethod
    def value(pointer):
        return pointer.value if isinstance(pointer, ctypes.c_void_p) else pointer

    def get_current_context(self):
        return self.current

    def choose_pixel_format(self, attrs, pixel, count):
        self.events.append(("choose", tuple(attrs)))
        rc, npix, ptr = self.choices.pop(0)
        self.put(pixel, ptr, ctypes.c_void_p)
        self.put(count, npix, ctypes.c_int)
        return rc

    def create_context(self, pixel, share, context):
        assert share is None
        self.events.append(("create", self.value(pixel)))
        rc, ptr = self.creations.pop(0)
        self.put(context, ptr, ctypes.c_void_p)
        return rc

    def set_current_context(self, context):
        self.current = self.value(context)
        self.events.append(("set", self.current))
        # Only selection binding fails, never restoration to borrowed prior.
        return self.set_rc if self.current == 22 else 0

    def lock_context(self, context):
        self.events.append(("lock", self.value(context)))
        return self.lock_rc

    def unlock_context(self, context):
        self.events.append(("unlock", self.value(context)))
        return self.unlock_rc

    def release_context(self, context):
        self.events.append(("release_context", self.value(context)))

    def release_pixel_format(self, pixel):
        self.events.append(("release_pixel", self.value(pixel)))


def test_exact_finite_order_and_success_stops():
    assert owned._FORMATS == EXPECTED_FORMATS
    api = FakeCGL([(10002, 0, None), (0, 0, 12), (0, 1, 13)])
    context = owned._OwnedCGLContext(3, 2, api=api)
    assert context.candidate_index == 2
    assert [e[1] for e in api.events if e[0] == "choose"] == list(EXPECTED_FORMATS)
    assert ("release_pixel", 12) in api.events
    assert context.attempts == [(0, 10002, 0, False), (1, 0, 0, True), (2, 0, 1, True)]
    context.free()
    api = FakeCGL()
    context = owned._OwnedCGLContext(3, 2, api=api)
    assert len([e for e in api.events if e[0] == "choose"]) == 1
    context.free()


@pytest.mark.parametrize("choice", [(0, 1, None), (0, 0, 31), (9, 1, 31)])
def test_bad_choose_is_never_used_to_create(choice):
    api = FakeCGL([choice] * 3)
    with pytest.raises(RuntimeError, match="No owned CGL"):
        owned._OwnedCGLContext(3, 2, api=api)
    assert not any(e[0] == "create" for e in api.events)
    assert len([e for e in api.events if e[0] == "choose"]) == 3
    assert len([e for e in api.events if e[0] == "release_pixel"]) == (3 if choice[2] else 0)


@pytest.mark.parametrize("creation", [(8, 22), (0, None)])
def test_partial_creation_cleanup_before_next_attempt(creation):
    api = FakeCGL([(0, 1, 11), (0, 1, 12)], [creation, (0, 22)])
    context = owned._OwnedCGLContext(3, 2, api=api)
    expected = [("choose", EXPECTED_FORMATS[0]), ("create", 11)]
    if creation[1]:
        expected.append(("release_context", 22))
    expected.extend([("release_pixel", 11), ("choose", EXPECTED_FORMATS[1])])
    assert api.events[: len(expected)] == expected
    context.free()


def test_repeated_current_balances_one_lock_and_idempotent_free():
    api = FakeCGL()
    context = owned._OwnedCGLContext(3, 2, api=api)
    for _ in range(4):
        context.make_current()
    context.free()
    events = list(api.events)
    context.free()
    assert api.events == events
    assert api.events.count(("lock", 22)) == 1
    assert api.events[-4:] == [
        ("unlock", 22),
        ("set", 99),
        ("release_context", 22),
        ("release_pixel", 11),
    ]
    assert api.current == 99
    with pytest.raises(RuntimeError, match="closed"):
        context.make_current()


def test_thread_rejection_has_no_native_side_effects(monkeypatch):
    api = FakeCGL()
    context = owned._OwnedCGLContext(3, 2, api=api)
    before = list(api.events)
    with monkeypatch.context() as patch:
        patch.setattr(owned, "threading", SimpleNamespace(get_ident=lambda: -1))
        with pytest.raises(RuntimeError, match="different thread"):
            context.make_current()
        with pytest.raises(RuntimeError, match="different thread"):
            context.free()
    assert api.events == before
    context.free()


class FakeCamera:
    pass


class FakeEngine:
    def __init__(self, events):
        self.events = events
        self.fail_mjr = False
        self.fail_buffer = False
        self.mjtFontScale = SimpleNamespace(mjFONTSCALE_150=SimpleNamespace(value=150))
        self.mjtFramebuffer = SimpleNamespace(mjFB_OFFSCREEN=SimpleNamespace(value=1))
        self.mjtDepthMap = SimpleNamespace(mjDEPTH_ZEROFAR=0)
        self.mjtRndFlag = SimpleNamespace(mjRND_SEGMENT=0, mjRND_IDCOLOR=1)
        self.mjtObj = SimpleNamespace(mjOBJ_CAMERA=SimpleNamespace(value=7))
        self.mjtCamera = SimpleNamespace(mjCAMERA_FREE=0, mjCAMERA_FIXED=1)
        self.mjtCatBit = SimpleNamespace(mjCAT_ALL=SimpleNamespace(value=63))
        self.MjvCamera = FakeCamera
        self.updated_camera = None
        self.MjvScene = self.mjv_scene
        self.MjvOption = self.mjv_option
        self.MjrRect = self.mjr_rect
        self.MjrContext = self.mjr_context
        self.mjr_setBuffer = self.mjr_set_buffer
        self.mjr_readPixels = self.mjr_read_pixels
        self.mjv_defaultFreeCamera = self.mjv_default_free_camera
        self.mjv_updateScene = self.mjv_update_scene

    def mjv_scene(self, *, model, maxgeom):
        self.events.append(("scene", maxgeom))
        return SimpleNamespace(
            flags=np.zeros(2, dtype=bool),
            ngeom=1,
            geoms=[SimpleNamespace(segid=0, objid=42, objtype=5)],
        )

    def mjv_option(self):
        return object()

    def mjr_rect(self, x, y, width, height):
        return SimpleNamespace(width=width, height=height)

    def mjr_context(self, model, font):
        self.events.append(("mjr_create", font))
        if self.fail_mjr:
            raise RuntimeError("mjr construction failure")
        return SimpleNamespace(free=lambda: self.events.append(("mjr_free",)))

    def mjr_set_buffer(self, buffer, context):
        self.events.append(("buffer", buffer))
        if self.fail_buffer:
            raise RuntimeError("buffer failure")

    def mjr_render(self, rect, scene, context):
        self.events.append(("render",))

    def mjr_read_pixels(self, rgb, depth, rect, context):
        if depth is not None:
            depth[:] = 0
            depth[0, :] = 1
        else:
            rgb[:] = 0
            rgb[0, :, 0] = 1

    def mj_name2id(self, model, kind, name):
        return 0 if name == "cam" else -1

    def mjv_default_free_camera(self, model, camera):
        self.events.append(("free_camera",))

    def mjv_update_scene(self, model, data, option, perturb, camera, mask, scene):
        self.updated_camera = camera
        self.events.append(("update", mask))


@pytest.fixture
def boundary(monkeypatch):
    api = FakeCGL()
    engine = FakeEngine(api.events)
    primary = (
        Path(__file__).resolve().parents[2]
        / "history_inputs/primary/mujoco_classic_renderer.py.txt"
    )
    source = primary.read_bytes()
    assert (
        hashlib.sha256(source).hexdigest()
        == "c193df6a8b8cc1659819abd0ded5c17dab0e3e64edb7af6a1e8d249bb4a4a548"
    )
    tree = ast.parse(source)
    renderer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Renderer")
    # Never execute upstream imports, module assignments or its constructor.
    class_tree = ast.Module(body=[renderer], type_ignores=[])
    sentinel = object()
    namespace = {"mujoco": engine, "np": np, "gl_context": SimpleNamespace(GLContext=sentinel)}
    # Narrow fake metadata adaptation: the gate reads the exact pinned primary.
    namespace["__name__"] = "mujoco.rendering.classic.renderer"
    namespace["__file__"] = str(primary)
    engine.__version__ = "3.13.0"
    exec(
        compile(class_tree, str(primary), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    base = namespace["Renderer"]
    monkeypatch.setitem(
        sys.modules, base.__module__, SimpleNamespace(Renderer=base, __file__=str(primary))
    )
    original_init = base.__init__
    model = SimpleNamespace(
        vis=SimpleNamespace(
            global_=SimpleNamespace(offwidth=10, offheight=10),
            map=SimpleNamespace(znear=0.1, zfar=10.0),
        ),
        stat=SimpleNamespace(extent=2),
        ncam=1,
    )
    monkeypatch.setattr(owned, "_CGLAPI", lambda: api)
    cls = owned._renderer_type(base, engine)
    yield SimpleNamespace(
        api=api,
        engine=engine,
        cls=cls,
        base=base,
        model=model,
        namespace=namespace,
        sentinel=sentinel,
        original_init=original_init,
    )
    # No live CGL or model was imported or invoked by this fixture.


def test_upstream_unmodified_and_selected_context_used(boundary):
    b = boundary
    before = dict(vars(b.base))
    renderer = b.cls(b.model, 2, 3)
    assert dict(vars(b.base)) == before
    assert b.base.__init__ is b.original_init
    assert b.namespace["gl_context"].GLContext is b.sentinel
    assert b.cls.render is b.base.render
    assert b.cls.update_scene is b.base.update_scene
    for _ in range(3):
        renderer.render()
    assert b.api.events.count(("set", 22)) == 4
    renderer.close()
    assert b.api.events.index(("mjr_free",)) < b.api.events.index(("release_context", 22))
    before = list(b.api.events)
    renderer.close()
    assert b.api.events == before


@pytest.mark.parametrize("failure", ["mjr", "buffer", "set", "lock"])
def test_renderer_partial_failure_cleanup(boundary, failure):
    b = boundary
    b.engine.fail_mjr = failure == "mjr"
    b.engine.fail_buffer = failure == "buffer"
    b.api.set_rc = 7 if failure == "set" else 0
    b.api.lock_rc = 8 if failure == "lock" else 0
    with pytest.raises(RuntimeError):
        b.cls(b.model, 2, 3)
    assert b.api.current == 99
    assert b.api.events.count(("release_context", 22)) == 1
    assert b.api.events.count(("release_pixel", 11)) == 1
    assert (("mjr_free",) in b.api.events) == (failure == "buffer")
    if failure == "buffer":
        assert b.api.events.index(("mjr_free",)) < b.api.events.index(("unlock", 22))


def test_renderer_validation_before_native_acquisition(boundary):
    b = boundary
    with pytest.raises(ValueError, match="Image width"):
        b.cls(b.model, 2, 11)
    with pytest.raises(ValueError, match="Image height"):
        b.cls(b.model, 11, 3)
    assert b.api.events == []


def test_rgb_depth_segmentation_output_and_scene_contracts(boundary):
    b = boundary
    r = b.cls(b.model, 2, 3, 123)
    assert (r.height, r.width, r.model) == (2, 3, b.model)
    assert ("scene", 123) in b.api.events
    rgb = np.empty((2, 3, 3), dtype=np.uint8)
    assert r.render(out=rgb) is rgb
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[0], 0)
    np.testing.assert_array_equal(rgb[1, :, 0], 1)
    with pytest.raises(ValueError, match="out.shape"):
        r.render(out=np.empty((3, 2, 3)))
    flags = r.scene.flags.copy()
    r.enable_depth_rendering()
    assert r._depth_rendering and not r._segmentation_rendering
    depth = r.render()
    assert depth.shape == (2, 3) and depth.dtype == np.float32
    # Fake reverse-Z samples 0 and 1 become far=20 and near=0.2, then flip.
    np.testing.assert_allclose(depth[0], 20.0, rtol=1e-5)
    np.testing.assert_allclose(depth[1], 0.2, rtol=1e-5)
    np.testing.assert_array_equal(r.scene.flags, flags)
    r.enable_segmentation_rendering()
    assert r._segmentation_rendering and not r._depth_rendering
    segmentation = r.render()
    assert segmentation.shape == (2, 3, 2) and segmentation.dtype == np.int32
    np.testing.assert_array_equal(segmentation[0], np.full((3, 2), -1))
    np.testing.assert_array_equal(segmentation[1], np.tile([42, 5], (3, 1)))
    np.testing.assert_array_equal(r.scene.flags, flags)
    r.update_scene(object(), camera="cam")
    assert b.engine.updated_camera.fixedcamid == 0
    assert b.engine.updated_camera.type == 1
    with pytest.raises(ValueError, match="does not exist"):
        r.update_scene(object(), camera="absent")
    r.close()
    with pytest.raises(RuntimeError, match="after close"):
        r.render()


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_non_darwin_delegates_arguments_unchanged(monkeypatch, platform):
    calls = []
    result = object()

    def original(*args, **kwargs):
        return calls.append((args, kwargs)) or result

    fake = SimpleNamespace(Renderer=original)
    monkeypatch.setitem(sys.modules, "mujoco", fake)
    monkeypatch.setattr(owned, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(owned, "_CGLAPI", lambda: pytest.fail("Darwin probing"))
    model = object()
    assert owned.create_renderer(model, 2, 3, max_geom=17) is result
    assert calls == [((model, 2, 3), {"max_geom": 17})]
    assert fake.Renderer is original


def test_darwin_factory_does_not_call_original_constructor(boundary, monkeypatch):
    b = boundary
    fake = SimpleNamespace(**vars(b.engine))
    # Bind fake methods as well as enum/data attributes.
    for name in dir(b.engine):
        if not name.startswith("_"):
            setattr(fake, name, getattr(b.engine, name))
    fake.Renderer = lambda *a, **k: pytest.fail("Original constructor called")
    monkeypatch.setitem(sys.modules, "mujoco", fake)
    monkeypatch.setitem(
        sys.modules,
        "mujoco.rendering.classic.renderer",
        SimpleNamespace(Renderer=b.base, __file__=b.namespace["__file__"]),
    )
    monkeypatch.setattr(owned, "sys", SimpleNamespace(platform="darwin"))
    original = fake.Renderer
    renderer = owned.create_renderer(b.model, 2, 3)
    assert renderer.render().shape == (2, 3, 3)
    assert fake.Renderer is original
    renderer.close()


def test_repeated_renderer_lifecycles_release_each_acquisition(boundary):
    b = boundary
    b.api.choices = [(0, 1, 11)] * 3
    b.api.creations = [(0, 22)] * 3
    for _ in range(3):
        with b.cls(b.model, 2, 3) as renderer:
            assert renderer.render().shape == (2, 3, 3)
        renderer.close()
    assert b.api.events.count(("mjr_free",)) == 3
    assert b.api.events.count(("release_context", 22)) == 3
    assert b.api.events.count(("release_pixel", 11)) == 3
    assert b.api.current == 99


def test_renderer_thread_ownership_and_cleanup_failure_retry(boundary, monkeypatch):
    b = boundary
    renderer = b.cls(b.model, 2, 3)
    before = list(b.api.events)
    with monkeypatch.context() as patch:
        patch.setattr(owned, "threading", SimpleNamespace(get_ident=lambda: -1))
        with pytest.raises(RuntimeError, match="different thread"):
            renderer.render()
        with pytest.raises(RuntimeError, match="different thread"):
            renderer.close()
    assert b.api.events == before
    b.api.unlock_rc = 8
    with pytest.raises(RuntimeError, match="CGLUnlockContext"):
        renderer.close()
    assert ("mjr_free",) in b.api.events
    assert not any(e[0] == "release_context" for e in b.api.events)
    b.api.unlock_rc = 0
    renderer.close()
    assert b.api.events.count(("mjr_free",)) == 1
    assert b.api.events.count(("release_context", 22)) == 1


def test_public_query_bytes_none_not_pointer_indexing(monkeypatch):
    api = FakeCGL()
    context = owned._OwnedCGLContext(3, 2, api=api)
    values = {1: b"A" * 600, 2: b"Apple Software Renderer", 3: None}
    fake_gl = SimpleNamespace(GL_VENDOR=1, GL_RENDERER=2, GL_VERSION=3, glGetString=values.get)
    monkeypatch.setitem(sys.modules, "OpenGL", SimpleNamespace(GL=fake_gl))
    assert context.query_strings() == {
        "GL_VENDOR": "A" * 512,
        "GL_RENDERER": "Apple Software Renderer",
        "GL_VERSION": None,
    }
    values[1] = object()
    with pytest.raises(TypeError, match="bytes or None"):
        context.query_strings()
    context.free()


@pytest.mark.parametrize("fault", ["version", "bytes", "missing", "unreadable"])
def test_darwin_primary_gate_before_allocation(boundary, monkeypatch, tmp_path, fault):
    b = boundary
    primary = sys.modules[b.base.__module__]
    fake = SimpleNamespace(__version__="3.13.0", Renderer=b.base)
    monkeypatch.setitem(sys.modules, "mujoco", fake)
    monkeypatch.setattr(owned, "sys", SimpleNamespace(platform="darwin"))
    if fault == "version":
        fake.__version__ = "3.13.1"
    elif fault == "bytes":
        changed = tmp_path / "renderer.py"
        changed.write_bytes(Path(primary.__file__).read_bytes() + b"\n# changed\n")
        monkeypatch.setattr(primary, "__file__", str(changed))
    elif fault == "missing":
        monkeypatch.delattr(primary, "__file__")
    else:
        monkeypatch.setattr(primary, "__file__", str(tmp_path / "absent.py"))
    with pytest.raises(RuntimeError, match="3.13.0|primary source"):
        owned.create_renderer(b.model, 2, 3)
    assert b.api.events == []
    assert b.api.choices == [(0, 1, 11)]


@pytest.mark.parametrize("rc", [False, True, 0.0, None, ctypes.c_int(0)])
def test_checked_rejects_nonexact_integer_success(rc):
    with pytest.raises(RuntimeError, match="typed result"):
        owned._checked(rc, "typed result")
    owned._checked(0, "typed result")
    # In particular False must not reach context creation on any candidate.
    api = FakeCGL([(False, 1, 11)] * 3)
    with pytest.raises(RuntimeError, match="No owned CGL"):
        owned._OwnedCGLContext(3, 2, api=api)
    assert not any(e[0] == "create" for e in api.events)
    assert api.events.count(("release_pixel", 11)) == 3


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("cleanup_type", [RuntimeError, KeyboardInterrupt])
def test_context_initialization_preserves_failure_and_handles(
    monkeypatch, error_type, cleanup_type
):
    api = FakeCGL()
    original = error_type("choose interrupted after partial allocation")
    cleanup_error = cleanup_type("pixel release unconfirmed")

    def choose(attrs, pixel, count):
        api.choose_pixel_format(attrs, pixel, count)
        raise original

    def release(pixel):
        api.events.append(("release_failed", api.value(pixel)))
        raise cleanup_error

    monkeypatch.setattr(api, "CGLChoosePixelFormat", choose)
    monkeypatch.setattr(api, "CGLReleasePixelFormat", release)
    with pytest.raises(error_type) as caught:
        owned._OwnedCGLContext(3, 2, api=api)
    assert caught.value is original
    assert original.owned_cleanup_error is cleanup_error
    owner = original.owned_cleanup_owner
    assert (
        original.owned_cleanup_status
        == owner.ownership_status
        == {"context": None, "pixel_format": 11, "locked": False, "bound": False}
    )
    assert [e[0] for e in api.events] == ["choose", "release_failed"]
    monkeypatch.setattr(api, "CGLReleasePixelFormat", api.release_pixel_format)
    owner.free()
    assert owner.ownership_status["pixel_format"] is None
    # Successful cleanup of cancellation must still never select another format.
    api = FakeCGL()

    def cancelled_choose(attrs, pixel, count):
        api.choose_pixel_format(attrs, pixel, count)
        raise KeyboardInterrupt("cancel")

    monkeypatch.setattr(api, "CGLChoosePixelFormat", cancelled_choose)
    with pytest.raises(KeyboardInterrupt, match="cancel"):
        owned._OwnedCGLContext(3, 2, api=api)
    assert [e[0] for e in api.events] == ["choose", "release_pixel"]


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("cleanup_type", [RuntimeError, KeyboardInterrupt])
def test_renderer_initialization_preserves_failure_and_owner(
    boundary, monkeypatch, error_type, cleanup_type
):
    b = boundary
    original = error_type("buffer initialization interrupted")
    cleanup_error = cleanup_type("mjr release unconfirmed")

    def fail_buffer(buffer, context):
        raise original

    def fail_free():
        b.api.events.append(("mjr_free_failed",))
        raise cleanup_error

    def mjr_context(model, font):
        return SimpleNamespace(free=fail_free)

    monkeypatch.setattr(b.engine, "mjr_setBuffer", fail_buffer)
    monkeypatch.setattr(b.engine, "MjrContext", mjr_context)
    with pytest.raises(error_type) as caught:
        b.cls(b.model, 2, 3)
    assert caught.value is original
    assert original.owned_cleanup_error is cleanup_error
    renderer = original.owned_cleanup_owner
    assert (
        original.owned_cleanup_status
        == renderer.ownership_status
        == {
            "mjr_context": True,
            "gl_context": {"context": 22, "pixel_format": 11, "locked": True, "bound": True},
        }
    )
    assert not any(e[0].startswith("release_") for e in b.api.events)
    before = list(b.api.events)
    renderer.__del__()
    assert b.api.events == before
    # Explicit, same-thread repair only; do not free GL beneath an unfreed Mjr.
    renderer._mjr_context.free = lambda: b.api.events.append(("mjr_free",))
    renderer.close()
    assert renderer.ownership_status == {"mjr_context": False, "gl_context": None}
    assert b.api.events.index(("mjr_free",)) < b.api.events.index(("release_context", 22))
    assert b.api.events.count(("release_context", 22)) == 1


@pytest.mark.parametrize("hostile", ["attributes", "note", "both"])
def test_context_hostile_exception_hooks_preserve_identity(monkeypatch, hostile):
    diagnostic_error = BaseException("diagnostic rejected")
    attempts = []

    class HostileFailure(BaseException):
        def __setattr__(self, name, value):
            if name.startswith("owned_cleanup_"):
                attempts.append(name)
                if hostile in ("attributes", "both"):
                    raise diagnostic_error
            super().__setattr__(name, value)

        def add_note(self, note):
            attempts.append("add_note")
            if hostile in ("note", "both"):
                raise diagnostic_error
            super().add_note(note)

    api = FakeCGL()
    original = HostileFailure("choose interrupted")
    cleanup_error = BaseException("release unconfirmed")
    owners = []
    real_free = owned._OwnedCGLContext.free

    def choose(attrs, pixel, count):
        api.choose_pixel_format(attrs, pixel, count)
        raise original

    def release(pixel):
        api.events.append(("release_failed", api.value(pixel)))
        raise cleanup_error

    def free(owner):
        owners.append(owner)
        real_free(owner)

    monkeypatch.setattr(api, "CGLChoosePixelFormat", choose)
    monkeypatch.setattr(api, "CGLReleasePixelFormat", release)
    monkeypatch.setattr(owned._OwnedCGLContext, "free", free)
    with pytest.raises(HostileFailure) as caught:
        owned._OwnedCGLContext(3, 2, api=api)
    assert caught.value is original
    assert attempts == [
        "owned_cleanup_owner",
        "owned_cleanup_error",
        "owned_cleanup_status",
        "add_note",
    ]
    assert [e[0] for e in api.events] == ["choose", "release_failed"]
    assert len(owners) == 1
    owner = owners[0]
    assert owner.ownership_status["pixel_format"] == 11
    if hostile in ("attributes", "both"):
        assert not hasattr(original, "owned_cleanup_owner")
        assert not hasattr(original, "owned_cleanup_error")
        assert not hasattr(original, "owned_cleanup_status")
    else:
        assert original.owned_cleanup_owner is owner
        assert original.owned_cleanup_error is cleanup_error
        assert original.owned_cleanup_status == owner.ownership_status
    if hostile == "attributes":
        assert "attachment unconfirmed" in original.__notes__[0]
    else:
        assert not hasattr(original, "__notes__")
    monkeypatch.setattr(api, "CGLReleasePixelFormat", api.release_pixel_format)
    owner.free()
    assert owner.ownership_status["pixel_format"] is None
    assert api.events.count(("release_pixel", 11)) == 1


@pytest.mark.parametrize("hostile", ["status", "status_and_note", "status_repr"])
def test_renderer_hostile_status_preserves_identity(boundary, monkeypatch, hostile):
    b = boundary
    diagnostic_error = BaseException("diagnostic interrupted")
    attempts = []

    class HostileFailure(BaseException):
        def __setattr__(self, name, value):
            if name == "owned_cleanup_owner":
                attempts.append(name)
                raise diagnostic_error
            super().__setattr__(name, value)

        def add_note(self, note):
            attempts.append("add_note")
            if hostile == "status_and_note":
                raise diagnostic_error
            super().add_note(note)

    class HostileStatus:
        def __repr__(self):
            raise diagnostic_error

    original = HostileFailure("buffer interrupted")
    cleanup_error = BaseException("mjr release unconfirmed")
    owners = []
    real_close = b.cls.close
    real_status = b.cls.ownership_status

    def fail_buffer(buffer, context):
        raise original

    def fail_free():
        b.api.events.append(("mjr_free_failed",))
        raise cleanup_error

    def mjr_context(model, font):
        return SimpleNamespace(free=fail_free)

    def close(owner):
        owners.append(owner)
        real_close(owner)

    def status(owner):
        attempts.append("ownership_status")
        if hostile != "status_repr":
            raise diagnostic_error
        return HostileStatus()

    monkeypatch.setattr(b.engine, "mjr_setBuffer", fail_buffer)
    monkeypatch.setattr(b.engine, "MjrContext", mjr_context)
    monkeypatch.setattr(b.cls, "close", close)
    monkeypatch.setattr(b.cls, "ownership_status", property(status))
    with pytest.raises(HostileFailure) as caught:
        b.cls(b.model, 2, 3)
    assert caught.value is original
    assert attempts == ["owned_cleanup_owner", "ownership_status", "add_note"]
    assert original.owned_cleanup_error is cleanup_error
    assert not hasattr(original, "owned_cleanup_owner")
    assert len(owners) == 1
    renderer = owners[0]
    assert real_status.fget(renderer) == {
        "mjr_context": True,
        "gl_context": {"context": 22, "pixel_format": 11, "locked": True, "bound": True},
    }
    if hostile == "status_and_note":
        assert not hasattr(original, "__notes__")
    else:
        assert "owned_cleanup_owner attachment unconfirmed" in original.__notes__[0]
        assert "unavailable" in original.__notes__[0]
    if hostile != "status_repr":
        assert "UNKNOWN" in original.owned_cleanup_status
    assert not any(e[0].startswith("release_") for e in b.api.events)
    before = list(b.api.events)
    renderer.__del__()
    assert b.api.events == before
    assert len(owners) == 1
    renderer._mjr_context.free = lambda: b.api.events.append(("mjr_free",))
    renderer.close()
    assert real_status.fget(renderer) == {"mjr_context": False, "gl_context": None}
    assert b.api.events.index(("mjr_free",)) < b.api.events.index(("release_context", 22))
    assert b.api.events.count(("release_context", 22)) == 1
