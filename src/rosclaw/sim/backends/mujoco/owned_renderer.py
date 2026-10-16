# Copyright 2022 DeepMind Technologies Limited
# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Owned Darwin lifecycle; upstream rendering mathematics remain inherited.

Constructor adaptation: installed MuJoCo 3.13.0 classic/renderer.py, SHA256
c193df6a8b8cc1659819abd0ded5c17dab0e3e64edb7af6a1e8d249bb4a4a548.
CGL lifecycle/bindings references: cgl/__init__.py SHA256
075193793fb0743385eb37d0ca46ac3949a6eb1db6ad5bf41091bc588d96d41b,
cgl/cgl.py SHA256
b836c7148846cd0d6ea1f461c3ad60eade74e2687b10d9c68a7de06db50467c0.
These are installed-byte pins, not upstream commit identifiers. Local changes:
finite format selection, checked native results, thread ownership, balanced
locking, partial-failure cleanup, Mjr-before-CGL destruction. No upstream writes.
Importing this module does not import MuJoCo, OpenGL or load CGL.
"""

import contextlib
import ctypes
import hashlib
import inspect
import sys
import threading
from pathlib import Path

# Exact predeclared candidates: legacy default, no-MSAA accelerated, then
# no-MSAA unrestricted acceleration. The last is NOT an acceleration guarantee.
_FORMATS = (
    (99, 4096, 8, 24, 11, 8, 12, 24, 13, 8, 59, 55, 1, 56, 4, 73, 0),
    (99, 4096, 8, 24, 11, 8, 12, 24, 13, 8, 73, 0),
    (99, 4096, 8, 24, 11, 8, 12, 24, 13, 8, 0),
)


class _CGLAPI:
    """Private typed bindings, loaded only by the Darwin factory at call time.

    Release functions have void results: no invented release return code.
    No use of the upstream checked wrappers or CGLErrorString native reads.
    """

    def __init__(self):
        self._library = ctypes.CDLL("/System/Library/Frameworks/OpenGL.framework/OpenGL")
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        signatures = {
            "CGLChoosePixelFormat": (
                [ctypes.POINTER(integer), ctypes.POINTER(pointer), ctypes.POINTER(integer)],
                integer,
            ),
            "CGLCreateContext": (
                [pointer, pointer, ctypes.POINTER(pointer)],
                integer,
            ),
            "CGLGetCurrentContext": ([], pointer),
            "CGLSetCurrentContext": ([pointer], integer),
            "CGLLockContext": ([pointer], integer),
            "CGLUnlockContext": ([pointer], integer),
            "CGLReleaseContext": ([pointer], None),
            "CGLReleasePixelFormat": ([pointer], None),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self._library, name)
            function.argtypes = arguments
            function.restype = result
            setattr(self, name, function)


def _checked(rc, operation):
    if type(rc) is not int or rc != 0:
        raise RuntimeError(f"{operation} failed: rc={rc!r}")


def _validate_primary(base, mujoco):
    """Fail closed on the installed version and full inherited source bytes.

    No cached approval: each owned type selection checks the file belonging to
    the actual base's module, not a bundled reference copy or a version range.
    This is a compatibility gate, not protection against hostile Python mutation.
    """
    if getattr(mujoco, "__version__", None) != "3.13.0":
        raise RuntimeError("Owned Darwin renderer requires MuJoCo 3.13.0")
    module = inspect.getmodule(base)
    filename = getattr(module, "__file__", None)
    if not filename or getattr(module, "Renderer", None) is not base:
        raise RuntimeError("Owned Darwin Renderer primary source is unavailable")
    try:
        digest = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError("Owned Darwin Renderer primary source is unreadable") from error
    if digest != "c193df6a8b8cc1659819abd0ded5c17dab0e3e64edb7af6a1e8d249bb4a4a548":
        raise RuntimeError("Owned Darwin Renderer primary source SHA256 mismatch")


def _cleanup_after_failure(owner, original, cleanup):
    """Never replace an initialization failure, including cancellation.

    Best-effort attach the owner and unresolved ownership diagnostics. Hostile
    exception hooks can reject all attachments; the caller still re-raises the
    very same original object. No failed cleanup is reported as successful.
    """
    try:
        cleanup()
    except BaseException as cleanup_error:
        # Each diagnostic is independent: exception hooks and status access
        # are arbitrary Python and must not replace the caller's failure.
        limitations = []
        for name, value in (
            ("owned_cleanup_owner", owner),
            ("owned_cleanup_error", cleanup_error),
        ):
            try:
                setattr(original, name, value)
            except BaseException:
                limitations.append(f"{name} attachment unconfirmed")
        try:
            status = owner.ownership_status
        except BaseException:
            status = "UNKNOWN: ownership_status access failed; cleanup unconfirmed"
            limitations.append("ownership_status unavailable")
        try:
            original.owned_cleanup_status = status
        except BaseException:
            limitations.append("owned_cleanup_status attachment unconfirmed")
        try:
            status_text = repr(status)
        except BaseException:
            status_text = "UNKNOWN: ownership_status representation failed"
            limitations.append("ownership_status representation unavailable")
        # If attributes and notes are both rejected, no exception-carried
        # diagnostic is guaranteed. Never claim release or retry implicitly.
        with contextlib.suppress(BaseException):
            original.add_note(
                f"Owned cleanup failed; remaining ownership: {status_text}"
                + ("; diagnostic limitations: " + "; ".join(limitations) if limitations else "")
            )
        return False
    return True


class _OwnedCGLContext:
    """One context, one owning thread, at most three format attempts.

    Keep the borrowed prior context alive until free; callers must not switch
    or destroy it independently during this lifecycle. No cross-thread cleanup.
    """

    def __init__(self, width, height, *, api=None):
        del width, height  # Like upstream CGL, dimensions belong to MjrContext.
        self._thread = threading.get_ident()
        self._api = api if api is not None else _CGLAPI()
        self._pix = ctypes.c_void_p()
        self._context = ctypes.c_void_p()
        self._locked = False
        self._bound = False
        self._prior = self._api.CGLGetCurrentContext()
        self.attempts = []
        for index, values in enumerate(_FORMATS):
            npix = ctypes.c_int()
            attributes = (ctypes.c_int * len(values))(*values)
            try:
                rc = self._api.CGLChoosePixelFormat(
                    attributes, ctypes.byref(self._pix), ctypes.byref(npix)
                )
                self.attempts.append((index, rc, npix.value, bool(self._pix)))
                _checked(rc, "CGLChoosePixelFormat")
                if npix.value <= 0 or not self._pix:
                    raise RuntimeError("CGLChoosePixelFormat returned no format")
                rc = self._api.CGLCreateContext(self._pix, None, ctypes.byref(self._context))
                _checked(rc, "CGLCreateContext")
                if not self._context:
                    raise RuntimeError("CGLCreateContext returned NULL")
                self.candidate_index = index
                return
            except BaseException as original:
                # Any partial pointer returned on an error still belongs to us.
                # No next candidate after cancellation or uncertain cleanup.
                cleaned = _cleanup_after_failure(self, original, self.free)
                if not cleaned or not isinstance(original, Exception):
                    raise
        raise RuntimeError(f"No owned CGL format succeeded: {self.attempts!r}")

    def _check_thread(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError("Owned CGL context used from a different thread")

    def make_current(self):
        self._check_thread()
        if not self._context:
            raise RuntimeError("Owned CGL context is closed")
        # Even an error may have changed thread-local binding: restore on free.
        self._bound = True
        _checked(self._api.CGLSetCurrentContext(self._context), "CGLSetCurrentContext")
        if not self._locked:
            _checked(self._api.CGLLockContext(self._context), "CGLLockContext")
            self._locked = True

    def query_strings(self):
        """Optional public PyOpenGL bytes/None query; not a safety probe.

        512 bounds Python processing only, NOT native c_char_p reads inside
        PyOpenGL. Never index a returned pointer or claim native read bounds.
        """
        self.make_current()
        from OpenGL import GL

        result = {}
        for name in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION"):
            value = GL.glGetString(getattr(GL, name))
            if value is not None and not isinstance(value, bytes):
                raise TypeError("Pinned glGetString contract is bytes or None")
            result[name] = None if value is None else value[:512].decode("utf-8", errors="replace")
        return result

    @property
    def ownership_status(self):
        """Nonzero handles mean owned or release not confirmed, never success."""
        return {
            "context": self._context.value,
            "pixel_format": self._pix.value,
            "locked": self._locked,
            "bound": self._bound,
        }

    def free(self):
        self._check_thread()
        if self._locked:
            _checked(self._api.CGLUnlockContext(self._context), "CGLUnlockContext")
            self._locked = False
        if self._bound:
            _checked(self._api.CGLSetCurrentContext(self._prior), "restore prior CGL")
            self._bound = False
        if self._context:
            self._api.CGLReleaseContext(self._context)
            self._context = ctypes.c_void_p()
        if self._pix:
            self._api.CGLReleasePixelFormat(self._pix)
            self._pix = ctypes.c_void_p()


def _renderer_type(base, mujoco):
    """Build an owned subclass without touching base or its module globals."""
    _validate_primary(base, mujoco)

    class OwnedDarwinRenderer(base):
        # render/update_scene/flags/properties/context manager are inherited,
        # not copied, wrapped, or replaced. In particular, never call base init.
        def __init__(
            self,
            model,
            height=240,
            width=320,
            max_geom=10000,
            font_scale=mujoco.mjtFontScale.mjFONTSCALE_150,
        ):
            self._gl_context = None
            self._mjr_context = None
            self._owner_thread = threading.get_ident()

            buffer_width = model.vis.global_.offwidth
            buffer_height = model.vis.global_.offheight
            if width > buffer_width:
                raise ValueError(
                    f"""
Image width {width} > framebuffer width {buffer_width}. Either reduce the image
width or specify a larger offscreen framebuffer in the model XML using the
clause:
<visual>
  <global offwidth="my_width"/>
</visual>""".lstrip()
                )
            if height > buffer_height:
                raise ValueError(
                    f"""
Image height {height} > framebuffer height {buffer_height}. Either reduce the
image height or specify a larger offscreen framebuffer in the model XML using
the clause:
<visual>
  <global offheight="my_height"/>
</visual>""".lstrip()
                )

            self._width = width
            self._height = height
            self._model = model
            self._scene = mujoco.MjvScene(model=model, maxgeom=max_geom)
            self._scene_option = mujoco.MjvOption()
            self._rect = mujoco.MjrRect(0, 0, self._width, self._height)
            try:
                self._gl_context = _OwnedCGLContext(width, height)
                self._gl_context.make_current()
                self._mjr_context = mujoco.MjrContext(model, font_scale.value)
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._mjr_context)
                self._mjr_context.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
                self._depth_rendering = False
                self._segmentation_rendering = False
            except BaseException as original:
                # Keep failed initialization out of destructor-driven retries.
                self._initialization_failed = True
                _cleanup_after_failure(self, original, self.close)
                raise

        @property
        def ownership_status(self):
            return {
                "mjr_context": self._mjr_context is not None,
                "gl_context": (
                    None if self._gl_context is None else self._gl_context.ownership_status
                ),
            }

        def close(self):
            if threading.get_ident() != self._owner_thread:
                raise RuntimeError("Owned renderer closed from a different thread")
            # Mjr must be freed while its selected GL context is alive/current.
            # Retain ownership on cleanup failure; explicit close may retry.
            if self._mjr_context is not None:
                self._gl_context.make_current()
                self._mjr_context.free()
                self._mjr_context = None
            if self._gl_context is not None:
                self._gl_context.free()
                self._gl_context = None

        def __del__(self):
            # Never invoke native cleanup from the wrong thread. Explicit close
            # is required; destructor is only a best-effort same-thread fallback.
            if not getattr(self, "_initialization_failed", False):
                with contextlib.suppress(BaseException):
                    self.close()

    return OwnedDarwinRenderer


def create_renderer(*args, **kwargs):
    """Preserve all original positional/keyword arguments; Darwin only adapts.

    Call with (model, height, width, max_geom, font_scale) as with Renderer.
    No model compilation, physics stepping, or GL work occurs until called.
    """
    import mujoco

    if sys.platform != "darwin":
        return mujoco.Renderer(*args, **kwargs)
    from mujoco.rendering.classic.renderer import Renderer

    return _renderer_type(Renderer, mujoco)(*args, **kwargs)
