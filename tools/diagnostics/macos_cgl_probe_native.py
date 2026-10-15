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
            chosen = _invoke(
                row["choose"],
                lambda candidate=candidate, handles=handles: backend.choose(
                    candidate["numeric"], handles
                ),
            )
            row["choose"]["npix"] = handles["npix"]
            row["choose"]["pixel_format_null"] = not bool(handles["pix"])
            if (
                not chosen
                or not handles["pix"]
                or not isinstance(handles["npix"], int)
                or handles["npix"] <= 0
            ):
                continue
            created = _invoke(row["creation"], lambda handles=handles: backend.create(handles))
            row["creation"]["context_null"] = not bool(handles["ctx"])
            if not created or not handles["ctx"]:
                continue
            report["created_count"] += 1
            touched_current = True  # Also restore if SetCurrent fails or raises.
            if not _invoke(
                row["make_current"], lambda handles=handles: backend.set_current(handles["ctx"])
            ):
                continue
            report["current_count"] += 1
            for name in GL_FIELDS:
                field = row["query"][name]
                try:
                    value = backend.query(name)
                    if value:
                        field.update(status="OK", value=value[:512])
                    else:
                        field.update(status="UNAVAILABLE")
                except Exception as exc:
                    field.update(status="ERROR", error_type=type(exc).__name__[:80])
            row["succeeded"] = all(v["status"] == "OK" for v in row["query"].values())
        finally:
            row["owned"] = {
                "pixel_format_acquired": bool(handles["pix"]),
                "context_acquired": bool(handles["ctx"]),
            }
            detached = True
            if touched_current:
                detached = _invoke(
                    row["cleanup"]["restore_prior_null"], lambda: backend.set_current(None)
                )
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
                    try:
                        method(owned)  # Upstream release restype is None, not a CGL RC.
                        stage["status"] = "OK"
                    except Exception as exc:
                        stage.update(status="ERROR", error_type=type(exc).__name__[:80])
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
        # Import only AFTER successful SetCurrent. Signature and constants come
        # from hash-pinned generated GL1.0 source, not local C declarations.
        from OpenGL.raw.GL.VERSION import GL_1_0

        if not self.query_verified:
            actual_source = _hash_file(GL_1_0.__file__)
            library = str(getattr(GL_1_0._p.PLATFORM.GL, "_name", UNKNOWN))[:512]
            self.metadata["loaded_gl_1_0_source"] = actual_source
            self.metadata["query_library"] = {"requested_library": library, **_hash_file(library)}
            if actual_source["sha256"] != GL_SHA256 or library != FRAMEWORK:
                raise RuntimeError("UNVERIFIED_QUERY_BINDING")
            self.query_verified = True
        ptr = GL_1_0.glGetString(getattr(GL_1_0, name))
        if ptr is None:
            return None
        if isinstance(ptr, bytes):
            return ptr.split(b"\0", 1)[0][:512].decode("utf-8", "replace")
        if not bool(ptr):
            return None
        # Bounded scan avoids an unbounded string_at read. GL owns this pointer.
        output = bytearray()
        for i in range(512):
            byte = ptr[i]
            if not byte:
                break
            output.append(byte)
        return output.decode("utf-8", "replace")


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
            backend = (boundary_factory or _NativeBoundary)(report["metadata"])
            _attempts(backend, report)
        except Exception as exc:
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
