"""Durable namespace directory registration and atomic mutable-file writes.

Owners establish their boundary before creating nested paths. Existing immutable
intent bytes are validated and never reconstructed. Consumers choose whether a
particular existing payload may be updated; this helper never repairs corruption.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import tempfile
from pathlib import Path

from rosclaw.contracts.common import canonical_json


class DurableNamespace:
    def __init__(self, namespace: Path | str, *, kind: str = "sim") -> None:
        if kind not in ("sim", "practice", "plan", "legacy-sim"):
            raise ValueError("DIRECTORY_NAMESPACE_KIND_INVALID")
        self._task_root = Path(namespace)
        self._kind = kind

    def _directory_intent_anchor(self) -> Path:
        """Register an immutable namespace boundary before creating directories.

        The intent lives in the first already existing ancestor. New instances
        recover that exact boundary even when every new directory is visible
        following a failed sync. The intent itself must be synced first.
        """
        namespace = self._task_root.resolve()
        name = (
            f".rosclaw-{self._kind}-directory-"
            + hashlib.sha256(str(namespace).encode()).hexdigest()
            + ".json"
        )

        def find_intents() -> list[Path]:
            matches = [
                parent
                for parent in (namespace, *namespace.parents)
                if (parent / name).exists() or (parent / name).is_symlink()
            ]
            if len(matches) > 1:
                raise ValueError("STORE_DIRECTORY_INTENT_INVALID: multiple namespace anchors")
            return matches

        while True:
            found = find_intents()
            anchor = found[0] if found else namespace
            if not found:
                while not anchor.exists():
                    anchor = anchor.parent
            directory_fd = os.open(anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                fcntl.flock(directory_fd, fcntl.LOCK_EX)
                # A writer may have published the original ancestor intent and
                # created the namespace after our scan. Never register a second
                # anchor under a different lock based on that stale scan.
                current = find_intents()
                if current and current[0] != anchor:
                    continue  # finally releases this lock before choosing again
                if found and not current:
                    raise ValueError(
                        "STORE_DIRECTORY_INTENT_INVALID: registered intent disappeared"
                    )
                expected = canonical_json(
                    {
                        "schema": f"rosclaw.{self._kind}.directory_intent.v1",
                        "namespace": str(namespace),
                        "anchor": str(anchor),
                    }
                ).encode()
                intent = anchor / name
                if intent.exists() or intent.is_symlink():
                    if intent.is_symlink() or intent.read_bytes() != expected:
                        raise ValueError("STORE_DIRECTORY_INTENT_INVALID: namespace intent differs")
                    fd = os.open(intent, os.O_RDONLY | os.O_NOFOLLOW)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                else:
                    fd, temporary = tempfile.mkstemp(dir=anchor, prefix=".tmp_sim_directory_")
                    try:
                        with os.fdopen(fd, "wb") as output:
                            output.write(expected)
                            output.flush()
                            os.fsync(output.fileno())
                        os.replace(temporary, intent)
                    except BaseException:
                        Path(temporary).unlink(missing_ok=True)
                        raise
                os.fsync(directory_fd)
                return anchor
            finally:
                os.close(directory_fd)

    def ensure_directory(self, folder: Path) -> None:
        """Sync the registered chain before acknowledging any object ref."""
        absolute_folder = folder.resolve()
        namespace = self._task_root.resolve()
        if absolute_folder != namespace and namespace not in absolute_folder.parents:
            raise ValueError("DURABLE_PATH_ESCAPE: directory is outside owner namespace")
        anchor = self._directory_intent_anchor()
        if absolute_folder != anchor and anchor not in absolute_folder.parents:
            raise ValueError("STORE_DIRECTORY_INTENT_INVALID: partition outside registered anchor")
        chain: list[Path] = []
        cursor = absolute_folder
        while cursor != anchor:
            chain.append(cursor)
            cursor = cursor.parent
        for directory in reversed(chain):
            directory.mkdir(exist_ok=True)
        # Repeat all parent links, including after constructing a new SimStore.
        # Scope never extends above the original registered existing ancestor.
        for directory in reversed(chain):
            parent_fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)

    def assert_owned(self, path: Path) -> None:
        namespace = self._task_root.resolve()
        if path.resolve() != namespace and namespace not in path.resolve().parents:
            raise ValueError("DURABLE_PATH_ESCAPE: path is outside owner namespace")
        if path.is_symlink():
            raise ValueError("DURABLE_PATH_ESCAPE: destination is a symlink")

    def sync_file(self, path: Path) -> None:
        self.assert_owned(path)
        self.ensure_directory(path.parent)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self._sync_directory(path.parent)

    @staticmethod
    def _sync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def atomic_replace(self, path: Path, data: bytes) -> None:
        self.assert_owned(path)
        self.ensure_directory(path.parent)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp_durable_")
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            self._sync_directory(path.parent)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
