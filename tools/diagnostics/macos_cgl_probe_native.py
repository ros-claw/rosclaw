"""Finite, source-only CGL diagnostic; not a MuJoCo rendering backend.

No native bindings are imported at module import. Production work is confined
inside a fresh thread, whose initial current CGL context is NULL. No unverified
CGLGetCurrentContext prototype is introduced. See docs/MACOS_CGL_PROBE_NATIVE.md.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import threading
from pathlib import Path

UNKNOWN = "UNKNOWN"
CGL_SHA256 = "b836c7148846cd0d6ea1f461c3ad60eade74e2687b10d9c68a7de06db50467c0"
INIT_SHA256 = "075193793fb0743385eb37d0ca46ac3949a6eb1db6ad5bf41091bc588d96d41b"
GL_SHA256 = "7a5572ab18e9a0aa2265ef6fc6d11019c61b2acffcd38d49a912fb38116d38ef"
PUBLIC_GL_GET_SHA256 = "8450ad54ba0a814585cc8d52660b7686861fe1c26181c885ff19cdb982ec824e"
PUBLIC_GL_INIT_SHA256 = "83028a306ab5e9f25b8a3018b7922beb0013612dea37aafdeaacb658758f149d"
PUBLIC_GL_VERSION_SHA256 = "9e576704e0493d6868e7c0c396f3654a76eabe3630572d05d45bd1530fa02873"
PUBLIC_GL_ARRAY_SHA256 = "b101de94b04b53854f359a381004315af7dca033d27deb152ac6ca96fa47c77f"
FRAMEWORK = "/System/Library/Frameworks/OpenGL.framework/OpenGL"
# (Exact symbolic token, exact numeric token). Zero is the array terminator.
_BASE = (
    ("CGLPFAOpenGLProfile", 99),
    ("CGLOGLPVersion_Legacy", 0x1000),
    ("CGLPFAColorSize", 8),
    ("24", 24),
    ("CGLPFAAlphaSize", 11),
    ("8", 8),
    ("CGLPFADepthSize", 12),
    ("24", 24),
    ("CGLPFAStencilSize", 13),
    ("8", 8),
)
_ACCEL = (("CGLPFAAccelerated", 73),)
_END = (("TERMINATOR", 0),)
# All experiments exist BEFORE any attempt. These are not production fallbacks.
CANDIDATES = (
    (
        "mujoco_3_13_default",
        _BASE
        + (
            ("CGLPFAMultisample", 59),
            ("CGLPFASampleBuffers", 55),
            ("1", 1),
            ("CGLPFASample", 56),
            ("4", 4),
        )
        + _ACCEL
        + _END,
    ),
    ("legacy_no_sampling_accelerated", _BASE + _ACCEL + _END),
    ("legacy_no_sampling_unrestricted_acceleration", _BASE + _END),
)
GL_FIELDS = ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")


_CGL_STAGE_STATE = [0, 0]


def _cgl_stage_event(stage, phase, detail=None):
    """Process-wide progress budget; exact-type allowlist never inspects handles.

    Counters reserve space before writing. Sink errors cannot change ownership
    flow. No arbitrary value, exception message, address or path is serialized.
    """
    _state = _CGL_STAGE_STATE
    if _state[0] >= 128 or _state[1] >= 65536:
        return
    stages = (
        "boundary",
        "binding",
        "choose",
        "create",
        "current",
        "query",
        "GL_VENDOR",
        "GL_RENDERER",
        "GL_VERSION",
        "detach",
        "release_context",
        "release_pixel_format",
    )
    event = {
        "stage": stage if type(stage) is str and stage in stages else "UNKNOWN",
        "phase": phase if type(phase) is str and phase in ("pre", "post") else "UNKNOWN",
    }
    if type(detail) is dict:
        status = detail.get("status")
        rc = detail.get("rc")
        if type(status) is str and status in (
            "NOT_RUN",
            "ATTEMPTED",
            "OK",
            "FAIL",
            "ERROR",
            "UNAVAILABLE",
            "RETAINED_UNSAFE_TO_RELEASE",
        ):
            event["status"] = status
        if type(rc) is int and -2147483648 <= rc <= 2147483647:
            event["rc"] = rc
    line = json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n"
    size = len(line.encode("ascii"))
    if size > 1024 or _state[1] + size > 65536:
        return
    _state[0] += 1
    _state[1] += size
    try:
        sys.stderr.write(line)
        sys.stderr.flush()
    except Exception:
        return  # Progress delivery failure is not a native success or a retry.


def candidate_records():
    return [
        {"name": name, "symbols": [s for s, _ in tokens], "numeric": [v for _, v in tokens]}
        for name, tokens in CANDIDATES
    ]


def _stage():
    return {"status": "NOT_RUN", "rc": UNKNOWN}


def _invoke(record, call):
    """Preserve raw numeric RC; never turn exceptions into fabricated RCs."""
    record["status"] = "ATTEMPTED"
    try:
        rc = call()
        record["rc"] = rc if type(rc) is int else UNKNOWN
        record["status"] = "OK" if type(rc) is int and rc == 0 else "FAIL"
    except Exception as exc:  # A boundary exception is evidence, not a retry.
        record["status"] = "ERROR"
        record["error_type"] = type(exc).__name__[:80]
    return record["status"] == "OK"


def _attempts(backend, report):
    """Injected handle boundary. Caller must supply an isolated fresh thread.

    Backend mutates owned handles even if a native call raises after allocation.
    No destructor retries releases. A failed detach quarantines the context and
    stops the finite experiment rather than releasing a possibly current handle.
    """
    for candidate in report["candidates"]:
        if report["status"] == "STOPPED_CLEANUP_FAILURE":
            break
        handles = {"pix": None, "ctx": None, "npix": UNKNOWN}
        row = {
            "candidate": candidate["name"],
            "choose": _stage(),
            "creation": _stage(),
            "make_current": _stage(),
            "query": {name: {"status": "NOT_RUN", "value": UNKNOWN} for name in GL_FIELDS},
            "cleanup": {
                name: _stage()
                for name in ("restore_prior_null", "release_context", "release_pixel_format")
            },
            "owned": {"pixel_format_acquired": False, "context_acquired": False},
            "succeeded": False,
        }
        report["attempts"].append(row)
        report["attempted_count"] += 1
        touched_current = False
        try:
            _cgl_stage_event("choose", "pre")
            chosen = _invoke(
                row["choose"],
                lambda candidate=candidate, handles=handles: backend.choose(
                    candidate["numeric"], handles
                ),
            )
            _cgl_stage_event("choose", "post", row["choose"])
            row["choose"]["npix"] = handles["npix"]
            row["choose"]["pixel_format_null"] = not bool(handles["pix"])
            if (
                not chosen
                or not handles["pix"]
                or not isinstance(handles["npix"], int)
                or handles["npix"] <= 0
            ):
                continue
            _cgl_stage_event("create", "pre")
            created = _invoke(row["creation"], lambda handles=handles: backend.create(handles))
            _cgl_stage_event("create", "post", row["creation"])
            row["creation"]["context_null"] = not bool(handles["ctx"])
            if not created or not handles["ctx"]:
                continue
            report["created_count"] += 1
            touched_current = True  # Also restore if SetCurrent fails or raises.
            _cgl_stage_event("current", "pre")
            if not _invoke(
                row["make_current"], lambda handles=handles: backend.set_current(handles["ctx"])
            ):
                _cgl_stage_event("current", "post", row["make_current"])
                continue
            _cgl_stage_event("current", "post", row["make_current"])
            report["current_count"] += 1
            for name in GL_FIELDS:
                field = row["query"][name]
                _cgl_stage_event(name, "pre")
                try:
                    value = backend.query(name)
                    if value:
                        field.update(status="OK", value=value[:512])
                    else:
                        field.update(status="UNAVAILABLE")
                except Exception as exc:
                    field.update(status="ERROR", error_type=type(exc).__name__[:80])
                _cgl_stage_event(name, "post", field)
            row["succeeded"] = all(v["status"] == "OK" for v in row["query"].values())
        finally:
            row["owned"] = {
                "pixel_format_acquired": bool(handles["pix"]),
                "context_acquired": bool(handles["ctx"]),
            }
            detached = True
            if touched_current:
                _cgl_stage_event("detach", "pre")
                detached = _invoke(
                    row["cleanup"]["restore_prior_null"], lambda: backend.set_current(None)
                )
                _cgl_stage_event("detach", "post", row["cleanup"]["restore_prior_null"])
            for key, method, label in (
                ("ctx", backend.release_context, "release_context"),
                ("pix", backend.release_pixel_format, "release_pixel_format"),
            ):
                if handles[key]:
                    if key == "ctx" and not detached:
                        row["cleanup"][label]["status"] = "RETAINED_UNSAFE_TO_RELEASE"
                        continue
                    owned = handles[key]
                    handles[key] = None  # Transfer once, even when release raises.
                    stage = row["cleanup"][label]
                    stage["status"] = "ATTEMPTED"
                    _cgl_stage_event(label, "pre")
                    try:
                        method(owned)  # Upstream release restype is None, not a CGL RC.
                        stage["status"] = "OK"
                    except Exception as exc:
                        stage.update(status="ERROR", error_type=type(exc).__name__[:80])
                    _cgl_stage_event(label, "post", stage)
            cleanup_ok = all(v["status"] in ("OK", "NOT_RUN") for v in row["cleanup"].values())
            row["succeeded"] = row["succeeded"] and cleanup_ok
            report["succeeded_count"] += int(row["succeeded"])
            if not cleanup_ok:
                report["status"] = "STOPPED_CLEANUP_FAILURE"
        if report["status"] == "STOPPED_CLEANUP_FAILURE":
            break


def _hash_file(path):
    """Bounded input identity; dyld shared-cache-only files can be unavailable."""
    try:
        digest = hashlib.sha256()
        size = 0
        with Path(path).open("rb") as stream:
            while chunk := stream.read(65536):
                size += len(chunk)
                if size > 128 * 1024 * 1024:
                    return {"sha256": UNKNOWN, "status": "OVER_HASH_LIMIT"}
                digest.update(chunk)
        return {"sha256": digest.hexdigest(), "status": "FILE_HASHED", "bytes": size}
    except OSError:
        return {"sha256": UNKNOWN, "status": "FILE_UNAVAILABLE"}


class _NativeBoundary:
    """Use installed upstream declarations unchanged, never checked wrappers."""

    def __init__(self, metadata):
        _cgl_stage_event("binding", "pre")
        import ctypes

        self.ctypes = ctypes
        self.metadata = metadata
        self.query_verified = False
        dist = importlib.metadata.distribution("mujoco")
        gl_dist = importlib.metadata.distribution("PyOpenGL")
        metadata["mujoco_version"] = dist.version
        metadata["pyopengl_version"] = gl_dist.version
        cgl_path = dist.locate_file("mujoco/cgl/cgl.py")
        init_path = dist.locate_file("mujoco/cgl/__init__.py")
        gl_path = gl_dist.locate_file("OpenGL/raw/GL/VERSION/GL_1_0.py")
        pins = (
            ("cgl_source", cgl_path, CGL_SHA256),
            ("context_source", init_path, INIT_SHA256),
            ("gl_1_0_source", gl_path, GL_SHA256),
        )
        for label, path, _ in pins:
            metadata[label] = _hash_file(path)
        metadata["system_opengl"] = {"requested_library": FRAMEWORK, **_hash_file(FRAMEWORK)}
        if dist.version != "3.13.0" or any(
            metadata[label]["sha256"] != pin for label, _, pin in pins
        ):
            raise RuntimeError("UNVERIFIED_INSTALLED_SOURCE")
        # Load cgl.py alone. Importing mujoco.cgl would initialize the parent
        # package and is unnecessary. Hash gate precedes this native load.
        spec = importlib.util.spec_from_file_location("_diagnostic_upstream_cgl", cgl_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("SOURCE_LOADER_UNAVAILABLE")
        self.cgl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cgl)
        metadata["native_binding"] = "installed_hash_pinned_upstream_raw_symbols"
        _cgl_stage_event("binding", "post", {"status": "OK"})

    def choose(self, attrs, handles):
        c, b = self.ctypes, self.cgl
        pix, npix = b.CGLPixelFormatObj(), b.GLint()
        values = (c.c_int * len(attrs))(*attrs)
        try:
            return b._CGLChoosePixelFormat(values, c.byref(pix), c.byref(npix))
        finally:
            handles.update(pix=pix.value, npix=npix.value)

    def create(self, handles):
        c, b = self.ctypes, self.cgl
        ctx = b.CGLContextObj()
        try:
            # The upstream argtypes are (void*, int, void*). Cast the output
            # address to that exact declaration, without replacing the prototype.
            return b._CGLCreateContext(
                b.CGLPixelFormatObj(handles["pix"]), 0, c.cast(c.byref(ctx), b.CGLContextObj)
            )
        finally:
            handles["ctx"] = ctx.value

    def set_current(self, handle):
        return self.cgl._CGLSetCurrentContext(self.cgl.CGLContextObj(handle))

    def release_context(self, handle):
        self.cgl._CGLReleaseContext(self.cgl.CGLContextObj(handle))

    def release_pixel_format(self, handle):
        self.cgl._CGLReleasePixelFormat(self.cgl.CGLPixelFormatObj(handle))

    def query(self, name):
        # Only after SetCurrent. Public glget owns the c_char_p declaration;
        # do not alter a restype locally or dereference a raw GLubyteArray.
        _cgl_stage_event("query", "pre")
        try:
            from OpenGL import GL
            from OpenGL.arrays import arraydatatype
            from OpenGL.GL import glget
            from OpenGL.GL.VERSION import GL_1_1
            from OpenGL.raw.GL.VERSION import GL_1_0

            if name not in GL_FIELDS:
                raise ValueError("UNSUPPORTED_QUERY_FIELD")
            if not self.query_verified:
                sources = (
                    ("public_gl", GL, PUBLIC_GL_INIT_SHA256),
                    ("public_glget", glget, PUBLIC_GL_GET_SHA256),
                    ("public_gl_1_1", GL_1_1, PUBLIC_GL_VERSION_SHA256),
                    ("public_arraydatatype", arraydatatype, PUBLIC_GL_ARRAY_SHA256),
                    ("raw_gl_1_0", GL_1_0, GL_SHA256),
                )
                provenance = {label: _hash_file(module.__file__) for label, module, _ in sources}
                self.metadata["query_sources"] = provenance
                library = getattr(GL_1_0._p.PLATFORM.GL, "_name", None)
                if type(library) is not str or library != FRAMEWORK:
                    raise RuntimeError("UNVERIFIED_QUERY_LIBRARY")
                self.metadata["query_library"] = {
                    "requested_library": library,
                    **_hash_file(library),
                }
                if any(provenance[label]["sha256"] != pin for label, _, pin in sources):
                    raise RuntimeError("UNVERIFIED_PUBLIC_QUERY_SOURCE")
                if (
                    GL.glGetString is not glget.glGetString
                    or glget._simple is not GL_1_1
                    or glget.glGetString is not GL_1_1.glGetString
                ):
                    raise RuntimeError("UNVERIFIED_PUBLIC_QUERY_CHAIN")
                self.metadata["query_api"] = "OpenGL.GL.glGetString; public glget c_char_p"
                self.query_verified = True
            value = GL.glGetString(getattr(GL, name))
            if value is None:
                _cgl_stage_event("query", "post", {"status": "UNAVAILABLE"})
                return None
            if type(value) is not bytes:
                raise TypeError("PUBLIC_GL_STRING_NOT_BYTES_OR_NONE")
            # c_char_p relies on the API's NUL-terminated C string contract.
            # This is a Python processing cap, NOT a hard bound on the C read.
            result = value[:512].split(b"\0", 1)[0].decode("utf-8", "replace")
        except Exception:
            _cgl_stage_event("query", "post", {"status": "ERROR"})
            raise
        _cgl_stage_event("query", "post", {"status": "OK"})
        return result


def probe(system=None, boundary_factory=None):
    """Run at most three independent requests, never a render or model load."""
    system = platform.system() if system is None else system
    report = {
        "schema_version": 1,
        "status": "NOT_APPLICABLE" if system != "Darwin" else "OBSERVED",
        "candidates": candidate_records(),
        "attempts": [],
        "attempted_count": 0,
        "succeeded_count": 0,
        "created_count": 0,
        "current_count": 0,
        "metadata": {
            "system": system,
            "release": platform.release()[:128],
            "machine": platform.machine()[:128],
            "macos": platform.mac_ver()[0][:128],
            "python": platform.python_version(),
            "probe_source": _hash_file(__file__),
        },
        "prior_context": "NULL_ON_FRESH_WORKER_THREAD",
        "scope": "CGL negotiation only; no pixels, camera, model, physics or global absence proof",
    }
    if system != "Darwin":
        return report  # No CGL/OpenGL import and no boundary construction.

    def worker():
        try:
            _cgl_stage_event("boundary", "pre")
            backend = (boundary_factory or _NativeBoundary)(report["metadata"])
            _cgl_stage_event("boundary", "post", {"status": "OK"})
            _attempts(backend, report)
        except Exception as exc:
            _cgl_stage_event("boundary", "post", {"status": "ERROR"})
            report.update(status="BOUNDARY_ERROR", error_type=type(exc).__name__[:80])

    # Never touch the invoking thread's current context. CGL current contexts
    # are per-thread; a newly created thread has none. No thread-pool reuse.
    thread = threading.Thread(target=worker, name="isolated-cgl-diagnostic")
    thread.start()
    thread.join()
    return report


def main():
    result = probe()
    print(json.dumps(result, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
    if result["status"] == "STOPPED_CLEANUP_FAILURE":
        return 1
    return 0 if result["status"] == "NOT_APPLICABLE" or result["succeeded_count"] else 1


if __name__ == "__main__":
    sys.exit(main())
