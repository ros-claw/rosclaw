"""Build-time compatibility installer for the reference Humble image only.

No ROS imports, networking, package installation or runtime monkeypatching.
The CLI has no target/version overrides. install(image_root) is also usable on
an owned offline image fixture; it must never be used on a live ROS host.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import tempfile
from pathlib import Path

TARGET = Path("opt/ros/humble/local/lib/python3.10/dist-packages/rosapi/proxy.py")
STATUS = Path("var/lib/dpkg/status")
VERSIONS = {
    "ros-humble-rosapi": "2.0.8-1jammy.20260915.220619",
    "ros-humble-rclpy": "3.3.21-1jammy.20260724.022150",
}
ORIGINAL_SHA256 = "7042b78b804ec0b69a597dda48007d679448ce3b85c89d87c4a6c7b4bca208e7"
OLD_IMPORT = b"from ros2action.api import get_action_names_and_types\n"
NEW_IMPORT = b"from rclpy.action import get_action_names_and_types\n"
CALL = b"    names_and_types = get_action_names_and_types(node=_node)\n"


class CompatibilityError(RuntimeError):
    """Unsupported image or unsafe target; no patch should be attempted."""


def _regular_file(root: Path, relative: Path) -> Path:
    path = root
    for part in relative.parts:
        path = path / part
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise CompatibilityError(f"symlink not supported: {path}")
        if path != root / relative and not stat.S_ISDIR(info.st_mode):
            raise CompatibilityError(f"non-directory parent: {path}")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CompatibilityError(f"expected single-link regular file: {path}")
    return path


def _verify_versions(status: bytes) -> None:
    try:
        paragraphs = status.decode("utf-8").split("\n\n")
    except UnicodeDecodeError as exc:
        raise CompatibilityError("invalid dpkg status encoding") from exc
    found = {}
    for paragraph in paragraphs:
        fields = {}
        for line in paragraph.splitlines():
            if line and not line[0].isspace() and ": " in line:
                key, value = line.split(": ", 1)
                if key in fields:
                    raise CompatibilityError("duplicate dpkg field")
                fields[key] = value
        package = fields.get("Package")
        if package in VERSIONS:
            if package in found:
                raise CompatibilityError(f"duplicate package: {package}")
            found[package] = fields
    for package, version in VERSIONS.items():
        fields = found.get(package, {})
        if fields.get("Status") != "install ok installed" or fields.get("Version") != version:
            raise CompatibilityError(f"unsupported installed package/version: {package}")


def patched_bytes(source: bytes) -> bytes:
    """Accept only the pinned original or its exact one-line transformation."""
    if source.count(NEW_IMPORT) == 1 and OLD_IMPORT not in source:
        original = source.replace(NEW_IMPORT, OLD_IMPORT, 1)
        already_applied = True
    elif source.count(OLD_IMPORT) == 1 and NEW_IMPORT not in source:
        original = source
        already_applied = False
    else:
        raise CompatibilityError("unsupported action import layout")
    if original.count(CALL) != 1:
        raise CompatibilityError("unsupported action callback layout")
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
        raise CompatibilityError("unsupported full proxy SHA256")
    return source if already_applied else original.replace(OLD_IMPORT, NEW_IMPORT, 1)


def install(image_root: Path) -> str:
    """Apply to a private build-image root, or an owned temporary test root.

    All rejection checks precede writes. Replacement is atomic and preserves
    target mode/ownership; unrelated bytes (including the license) are retained.
    This is not a concurrency/security boundary for a hostile mutable root.
    """
    root = Path(image_root)
    if root.is_symlink() or not root.is_absolute() or root.resolve() != root:
        raise CompatibilityError("image root must be an absolute canonical non-link path")
    try:
        status_path = _regular_file(root, STATUS)
        _verify_versions(status_path.read_bytes())
        target = _regular_file(root, TARGET)
        before = target.stat()
        original = target.read_bytes()
        replacement = patched_bytes(original)
        if replacement == original:
            return "already-applied"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".rosapi-compat-", delete=False
            ) as out:
                temporary = Path(out.name)
                out.write(replacement)
                out.flush()
                os.fchown(out.fileno(), before.st_uid, before.st_gid)
                os.fchmod(out.fileno(), stat.S_IMODE(before.st_mode))
                os.fsync(out.fileno())
            current = _regular_file(root, TARGET).stat()
            if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
                raise CompatibilityError("target changed during install")
            if target.read_bytes() != original:
                raise CompatibilityError("target bytes changed during install")
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return "applied"
    except OSError as exc:
        raise CompatibilityError(f"image file operation failed: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args:
        print("rosapi Humble compatibility: no arguments supported", file=sys.stderr)
        return 2
    try:
        result = install(Path("/"))
    except CompatibilityError as exc:
        print(f"rosapi Humble compatibility refused: {exc}", file=sys.stderr)
        return 1
    print(f"rosapi Humble compatibility: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
