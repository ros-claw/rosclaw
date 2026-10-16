"""Pure boundary tests ONLY. Authored, not executed in this source phase.

These fakes test telemetry/ownership, not a CGL implementation or camera oracle.
No test constructs _NativeBoundary or imports a graphics/provider package.
"""

import builtins
import importlib.util
import json
import sys
import threading
import types
from pathlib import Path

import pytest


@pytest.fixture
def probe_module():
    path = Path(__file__).resolve().parents[2] / "tools/diagnostics/macos_cgl_probe_native.py"
    spec = importlib.util.spec_from_file_location("pure_cgl_probe_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Boundary:
    def __init__(self, mode="ok"):
        self.mode = mode
        self.events = []
        self.requested = []
        self.local = threading.local()
        self.index = -1

    def choose(self, attrs, handles):
        self.index += 1
        self.requested.append(list(attrs))
        self.events.append(("choose", self.index))
        handles.update(pix=f"p{self.index}", npix=1)
        if self.mode == "choose_exception":
            raise ValueError("partial pixel allocation")
        if self.mode == "choose_fail":
            return 10002
        if self.mode == "null":
            handles["pix"] = None
        if self.mode == "npix":
            handles["npix"] = 0
        return 0

    def create(self, handles):
        handles["ctx"] = f"c{self.index}"
        self.events.append(("create", self.index))
        return 10000 if self.mode == "create_fail" else 0

    def set_current(self, handle):
        self.events.append(("set", handle))
        if self.mode == "make_fail" and handle is not None:
            return 10004
        if self.mode == "detach_fail" and handle is None:
            return 10004
        self.local.current = handle
        return 0

    def query(self, name):
        assert getattr(self.local, "current", None) is not None
        self.events.append(("query", name))
        if self.mode == "query_fail" and name == "GL_VENDOR":
            raise ValueError("injected query failure")
        if self.mode == "unavailable":
            return None
        return "injected-" + name  # Not an expected native capability value.

    def release_context(self, handle):
        self.events.append(("release_context", handle))

    def release_pixel_format(self, handle):
        self.events.append(("release_pixel_format", handle))
        if self.mode == "release_fail":
            raise ValueError("release outcome unknown; do not retry")


def run(module, mode="ok"):
    boundary = Boundary(mode)
    report = module.probe(system="Darwin", boundary_factory=lambda metadata: boundary)
    return report, boundary


def test_exact_official_array_is_first(probe_module):
    report, boundary = run(probe_module)
    assert boundary.requested[0] == [99, 4096, 8, 24, 11, 8, 12, 24, 13, 8, 59, 55, 1, 56, 4, 73, 0]
    first = report["candidates"][0]
    assert first["symbols"] == [
        "CGLPFAOpenGLProfile",
        "CGLOGLPVersion_Legacy",
        "CGLPFAColorSize",
        "24",
        "CGLPFAAlphaSize",
        "8",
        "CGLPFADepthSize",
        "24",
        "CGLPFAStencilSize",
        "8",
        "CGLPFAMultisample",
        "CGLPFASampleBuffers",
        "1",
        "CGLPFASample",
        "4",
        "CGLPFAAccelerated",
        "TERMINATOR",
    ]


def test_finite_predeclared_no_implicit_fallback(probe_module):
    before = probe_module.candidate_records()
    assert 1 <= len(before) <= 4
    report, boundary = run(probe_module, "choose_fail")
    assert report["candidates"] == before == probe_module.candidate_records()
    assert boundary.requested == [c["numeric"] for c in before]
    assert report["attempted_count"] == len(before)
    assert report["succeeded_count"] == report["created_count"] == 0
    assert len({tuple(a) for a in boundary.requested}) == len(before)


def test_numeric_choose_null_and_npix_are_distinct(probe_module):
    for mode in ("choose_fail", "null", "npix"):
        report, boundary = run(probe_module, mode)
        first = report["attempts"][0]
        assert first["choose"]["rc"] == (10002 if mode == "choose_fail" else 0)
        assert first["choose"]["pixel_format_null"] == (mode == "null")
        assert first["choose"]["npix"] == (0 if mode == "npix" else 1)
        assert first["creation"]["status"] == "NOT_RUN"
        assert not any(e[0] in ("create", "query") for e in boundary.events)
        expected = 0 if mode == "null" else report["attempted_count"]
        assert sum(e[0] == "release_pixel_format" for e in boundary.events) == expected


def test_partial_create_failure_releases_both_once(probe_module):
    report, boundary = run(probe_module, "create_fail")
    assert report["created_count"] == report["current_count"] == 0
    for i, row in enumerate(report["attempts"]):
        assert row["creation"]["rc"] == 10000
        assert row["creation"]["context_null"] is False
        assert row["owned"] == {"pixel_format_acquired": True, "context_acquired": True}
        assert boundary.events.count(("release_context", f"c{i}")) == 1
        assert boundary.events.count(("release_pixel_format", f"p{i}")) == 1
    assert not any(e[0] in ("set", "query") for e in boundary.events)


def test_make_current_failure_restores_and_never_queries(probe_module):
    report, boundary = run(probe_module, "make_fail")
    assert report["created_count"] == report["attempted_count"]
    assert report["current_count"] == report["succeeded_count"] == 0
    assert not any(e[0] == "query" for e in boundary.events)
    for row in report["attempts"]:
        assert row["make_current"]["rc"] == 10004
        assert row["cleanup"]["restore_prior_null"]["rc"] == 0
        assert all(v["value"] == "UNKNOWN" for v in row["query"].values())


def test_query_error_keeps_unknown_without_hiding_other_fields(probe_module):
    report, boundary = run(probe_module, "query_fail")
    assert report["current_count"] == report["attempted_count"]
    assert report["succeeded_count"] == 0
    for row in report["attempts"]:
        assert row["query"]["GL_VENDOR"] == {
            "status": "ERROR",
            "value": "UNKNOWN",
            "error_type": "ValueError",
        }
        assert row["query"]["GL_VERSION"]["status"] == "OK"
        assert row["cleanup"]["release_context"]["status"] == "OK"
    assert sum(e[0] == "query" for e in boundary.events) == 3 * report["attempted_count"]


def test_choose_exception_still_cleans_partial_allocation(probe_module):
    report, boundary = run(probe_module, "choose_exception")
    for i, row in enumerate(report["attempts"]):
        assert row["choose"]["rc"] == "UNKNOWN"
        assert row["choose"]["status"] == "ERROR"
        assert row["choose"]["pixel_format_null"] is False
        assert row["choose"]["npix"] == 1
        assert boundary.events.count(("release_pixel_format", f"p{i}")) == 1
    assert not any(e[0] == "create" for e in boundary.events)


def test_release_exception_is_not_retried_and_stops(probe_module):
    report, boundary = run(probe_module, "release_fail")
    assert report["status"] == "STOPPED_CLEANUP_FAILURE"
    assert report["attempted_count"] == 1
    assert report["succeeded_count"] == 0
    assert boundary.events.count(("release_context", "c0")) == 1
    assert boundary.events.count(("release_pixel_format", "p0")) == 1
    release = report["attempts"][0]["cleanup"]["release_pixel_format"]
    assert release["status"] == "ERROR" and release["rc"] == "UNKNOWN"


def test_non_darwin_never_imports_graphics_or_constructs_boundary(probe_module, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in ("OpenGL", "mujoco")
        return original(name, *args, **kwargs)

    def forbidden(metadata):
        raise AssertionError("native boundary was constructed")

    monkeypatch.setattr(builtins, "__import__", guarded)
    report = probe_module.probe(system="Linux", boundary_factory=forbidden)
    assert report["status"] == "NOT_APPLICABLE"
    assert report["attempts"] == [] and report["attempted_count"] == 0


def test_caller_thread_context_untouched_and_detach_precedes_release(probe_module):
    boundary = Boundary()
    boundary.local.current = "caller-sentinel"
    caller = threading.get_ident()
    worker_ids = []

    def factory(metadata):
        worker_ids.append(threading.get_ident())
        assert getattr(boundary.local, "current", None) is None
        return boundary

    report = probe_module.probe(system="Darwin", boundary_factory=factory)
    assert worker_ids[0] != caller
    assert boundary.local.current == "caller-sentinel"
    assert report["succeeded_count"] == report["attempted_count"]
    for i in range(report["attempted_count"]):
        release = boundary.events.index(("release_context", f"c{i}"))
        assert boundary.events[release - 1] == ("set", None)
        assert boundary.events[release + 1] == ("release_pixel_format", f"p{i}")


def test_unavailable_strings_are_unknown_and_json_is_bounded(probe_module):
    report, _ = run(probe_module, "unavailable")
    assert report["current_count"] > 0 and report["succeeded_count"] == 0
    for row in report["attempts"]:
        assert all(
            v == {"status": "UNAVAILABLE", "value": "UNKNOWN"} for v in row["query"].values()
        )
    encoded = json.dumps(report)
    assert len(encoded) < 32768
    assert json.loads(encoded)["attempted_count"] == len(report["attempts"])


def test_detach_failure_retains_context_and_stops(probe_module):
    report, boundary = run(probe_module, "detach_fail")
    assert report["status"] == "STOPPED_CLEANUP_FAILURE"
    assert report["attempted_count"] == 1 and report["succeeded_count"] == 0
    cleanup = report["attempts"][0]["cleanup"]
    assert cleanup["restore_prior_null"]["rc"] == 10004
    assert cleanup["release_context"]["status"] == "RETAINED_UNSAFE_TO_RELEASE"
    assert not any(e[0] == "release_context" for e in boundary.events)
    assert boundary.events.count(("release_pixel_format", "p0")) == 1


def test_invoke_rejects_boolean_return_codes(probe_module):
    for value in (False, True):
        record = probe_module._stage()
        calls = []

        def call(calls=calls, value=value):
            calls.append(value)
            return value

        assert probe_module._invoke(record, call) is False
        assert calls == [value]
        assert record == {"status": "FAIL", "rc": "UNKNOWN"}
    for value in (0, 10002):
        record = probe_module._stage()
        assert probe_module._invoke(record, lambda value=value: value) is (value == 0)
        assert type(record["rc"]) is int
        assert record["rc"] == value


def test_invoke_rejects_float_return_codes_and_preserves_exception(probe_module):
    for value in (0.0, -0.0, 1.0, float("nan"), float("inf"), -float("inf")):
        record = probe_module._stage()
        assert probe_module._invoke(record, lambda value=value: value) is False
        assert record == {"status": "FAIL", "rc": "UNKNOWN"}

    def raises():
        raise ValueError("injected boundary error")

    record = probe_module._stage()
    assert probe_module._invoke(record, raises) is False
    assert record == {"status": "ERROR", "rc": "UNKNOWN", "error_type": "ValueError"}


def test_main_cleanup_failure_overrides_previous_success(probe_module, monkeypatch, capsys):
    boundary = Boundary()
    release = boundary.release_pixel_format

    def release_second_fails(handle):
        release(handle)
        if handle == "p1":
            raise ValueError("second candidate cleanup failed")

    boundary.release_pixel_format = release_second_fails
    report = probe_module.probe(system="Darwin", boundary_factory=lambda metadata: boundary)
    assert report["status"] == "STOPPED_CLEANUP_FAILURE"
    assert report["succeeded_count"] == 1 and report["attempted_count"] == 2
    assert report["attempts"][0]["succeeded"] is True
    assert report["attempts"][1]["cleanup"]["release_pixel_format"]["status"] == "ERROR"
    assert boundary.events.count(("release_pixel_format", "p1")) == 1
    monkeypatch.setattr(probe_module, "probe", lambda: report)
    assert probe_module.main() == 1
    assert json.loads(capsys.readouterr().out) == report

    for status, count, expected in (
        ("STOPPED_CLEANUP_FAILURE", 0, 1),
        ("NOT_APPLICABLE", 0, 0),
        ("OBSERVED", 1, 0),
        ("OBSERVED", 0, 1),
        ("BOUNDARY_ERROR", 0, 1),
    ):
        result = {"status": status, "succeeded_count": count}
        monkeypatch.setattr(probe_module, "probe", lambda result=result: result)
        assert probe_module.main() == expected
        assert json.loads(capsys.readouterr().out) == result


def test_actual_query_public_bytes_none_pointer_and_source_gates(probe_module, monkeypatch):
    # Synthetic modules only: execute the real query method, never __init__.
    names = (
        "OpenGL",
        "OpenGL.GL",
        "OpenGL.GL.glget",
        "OpenGL.GL.VERSION",
        "OpenGL.GL.VERSION.GL_1_1",
        "OpenGL.arrays",
        "OpenGL.arrays.arraydatatype",
        "OpenGL.raw",
        "OpenGL.raw.GL",
        "OpenGL.raw.GL.VERSION",
        "OpenGL.raw.GL.VERSION.GL_1_0",
    )
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        module.__file__ = "fake:" + name
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
    gl = modules["OpenGL.GL"]
    glget = modules["OpenGL.GL.glget"]
    version = modules["OpenGL.GL.VERSION.GL_1_1"]
    raw = modules["OpenGL.raw.GL.VERSION.GL_1_0"]
    raw._p = types.SimpleNamespace(
        PLATFORM=types.SimpleNamespace(GL=types.SimpleNamespace(_name=probe_module.FRAMEWORK))
    )
    response = [b"fake-vendor"]
    requests = []

    def public_get_string(token):
        requests.append(token)
        return response[0]

    def forbidden_raw(token):
        raise AssertionError("raw glGetString must not be called")

    gl.glGetString = glget.glGetString = version.glGetString = public_get_string
    glget._simple = version
    raw.glGetString = forbidden_raw
    for index, name in enumerate(probe_module.GL_FIELDS):
        setattr(gl, name, index + 1)
    pins = {
        "fake:OpenGL.GL": "83028a306ab5e9f25b8a3018b7922beb0013612dea37aafdeaacb658758f149d",
        "fake:OpenGL.GL.glget": "8450ad54ba0a814585cc8d52660b7686861fe1c26181c885ff19cdb982ec824e",
        "fake:OpenGL.GL.VERSION.GL_1_1": "9e576704e0493d6868e7c0c396f3654a76eabe3630572d05d45bd1530fa02873",
        "fake:OpenGL.arrays.arraydatatype": "b101de94b04b53854f359a381004315af7dca033d27deb152ac6ca96fa47c77f",
        "fake:OpenGL.raw.GL.VERSION.GL_1_0": probe_module.GL_SHA256,
    }
    hashed = []

    def fake_hash(path):
        hashed.append(path)
        return {"sha256": pins.get(path, "UNKNOWN"), "status": "FAKE_METADATA"}

    monkeypatch.setattr(probe_module, "_hash_file", fake_hash)

    class Sink:
        def __init__(self):
            self.lines = []
            self.flushes = 0

        def write(self, line):
            self.lines.append(line)

        def flush(self):
            self.flushes += 1

    sink = Sink()
    monkeypatch.setattr(probe_module.sys, "stderr", sink)
    boundary = Boundary()
    boundary.metadata = {}
    boundary.query_verified = False
    boundary.query = types.MethodType(probe_module._NativeBoundary.query, boundary)
    report = probe_module.probe(system="Darwin", boundary_factory=lambda metadata: boundary)
    assert report["succeeded_count"] == 3
    assert requests == [1, 2, 3] * 3
    assert set(pins).issubset(hashed)
    assert boundary.metadata["query_sources"]["public_glget"]["sha256"] == pins[glget.__file__]
    assert "c_char_p" in boundary.metadata["query_api"]
    events = [json.loads(line) for line in sink.lines]
    pairs = [(e["stage"], e["phase"]) for e in events]
    for stage in (
        "boundary",
        "choose",
        "create",
        "current",
        "GL_VENDOR",
        "GL_RENDERER",
        "GL_VERSION",
        "query",
        "detach",
        "release_context",
        "release_pixel_format",
    ):
        assert (stage, "pre") in pairs and (stage, "post") in pairs
        assert pairs.index((stage, "pre")) < pairs.index((stage, "post"))
    assert pairs.index(("detach", "post")) < pairs.index(("release_context", "pre"))
    assert pairs.index(("release_context", "post")) < pairs.index(("release_pixel_format", "pre"))
    assert sink.flushes == len(sink.lines)

    for value, expected in (
        (None, None),
        (b"", ""),
        (b"abc\0ignored", "abc"),
        (b"x" * 600 + b"\0", "x" * 512),
        (b"\xff", "\ufffd"),
    ):
        response[0] = value
        assert boundary.query("GL_VENDOR") == expected

    class PoisonPointer:
        def trap(self, *args):
            raise AssertionError("pointer inspected")

        __bool__ = __getitem__ = __str__ = __repr__ = __int__ = __iter__ = trap

    for value in (PoisonPointer(), 0, "not bytes", bytearray(b"not bytes")):
        response[0] = value
        with pytest.raises(TypeError, match="PUBLIC_GL_STRING_NOT_BYTES_OR_NONE"):
            boundary.query("GL_VENDOR")
    response[0] = b"safe"
    for path in tuple(pins):
        expected_pin = pins[path]
        pins[path] = "0" * 64
        boundary.query_verified = False
        before = len(requests)
        with pytest.raises(RuntimeError, match="UNVERIFIED_PUBLIC_QUERY_SOURCE"):
            boundary.query("GL_VENDOR")
        assert len(requests) == before
        pins[path] = expected_pin
    gl.glGetString = forbidden_raw
    boundary.query_verified = False
    with pytest.raises(RuntimeError, match="UNVERIFIED_PUBLIC_QUERY_CHAIN"):
        boundary.query("GL_VENDOR")
    gl.glGetString = public_get_string
    raw._p.PLATFORM.GL._name = "not-the-framework"
    with pytest.raises(RuntimeError, match="UNVERIFIED_QUERY_LIBRARY"):
        boundary.query("GL_VENDOR")


def test_stage_serializer_adversarial_budget_and_flush(probe_module, monkeypatch):
    class Poison:
        def trap(self, *args):
            raise AssertionError("arbitrary object inspected")

        __bool__ = __getitem__ = __str__ = __repr__ = __int__ = __iter__ = trap

    class Sink:
        def __init__(self):
            self.lines = []
            self.flushes = 0

        def write(self, line):
            self.lines.append(line)

        def flush(self):
            self.flushes += 1

    sink = Sink()
    monkeypatch.setattr(probe_module.sys, "stderr", sink)
    secret = "address-or-secret-must-not-appear" * 10000
    poison = Poison()
    for _ in range(200):
        probe_module._cgl_stage_event(
            poison, secret, {"status": poison, "rc": poison, "secret": secret}
        )
    assert len(sink.lines) == sink.flushes == 128
    assert all(json.loads(line) == {"stage": "UNKNOWN", "phase": "UNKNOWN"} for line in sink.lines)
    assert all(len(line.encode("ascii")) <= 1024 for line in sink.lines)
    assert sum(len(line.encode("ascii")) for line in sink.lines) <= 65536
    probe_module._CGL_STAGE_STATE[:] = [0, 65530]
    probe_module._cgl_stage_event("choose", "pre")
    assert len(sink.lines) == 128
    probe_module._CGL_STAGE_STATE[:] = [0, 0]
    probe_module._cgl_stage_event("choose", "post", {"rc": True, "status": secret})
    assert json.loads(sink.lines[-1]) == {"stage": "choose", "phase": "post"}
    probe_module._cgl_stage_event("choose", "post", {"rc": 10002, "status": "FAIL"})
    assert json.loads(sink.lines[-1])["rc"] == 10002

    def broken_write(line):
        raise OSError("sink unavailable")

    monkeypatch.setattr(sink, "write", broken_write)
    probe_module._cgl_stage_event("detach", "pre")  # Must not interrupt cleanup.


def test_stage_error_post_and_cleanup_order_remain_observable(probe_module, monkeypatch):
    writes = []
    flushed = []

    class Sink:
        def write(self, line):
            writes.append(json.loads(line))

        def flush(self):
            flushed.append(len(writes))

    monkeypatch.setattr(probe_module.sys, "stderr", Sink())
    report, boundary = run(probe_module, "query_fail")
    assert report["succeeded_count"] == 0
    assert len(writes) == len(flushed)
    vendor = [e for e in writes if e["stage"] == "GL_VENDOR"]
    assert vendor == [
        event
        for _ in range(3)
        for event in (
            {"stage": "GL_VENDOR", "phase": "pre"},
            {"stage": "GL_VENDOR", "phase": "post", "status": "ERROR"},
        )
    ]
    for i in range(3):
        detach = boundary.events.index(("set", None), i * 9)
        assert boundary.events[detach + 1] == ("release_context", f"c{i}")
        assert boundary.events[detach + 2] == ("release_pixel_format", f"p{i}")
    writes.clear()
    flushed.clear()
    probe_module._CGL_STAGE_STATE[:] = [0, 0]
    report, _ = run(probe_module, "detach_fail")
    assert report["status"] == "STOPPED_CLEANUP_FAILURE"
    assert {"stage": "detach", "phase": "post", "status": "FAIL", "rc": 10004} in writes
    assert not any(e["stage"] == "release_context" for e in writes)
    assert writes[-1] == {"stage": "release_pixel_format", "phase": "post", "status": "OK"}
    assert len(writes) == len(flushed)
