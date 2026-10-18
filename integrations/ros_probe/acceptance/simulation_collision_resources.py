"""Read exact prepared original resource bytes; no ROS, motion or authority."""

import hashlib
import os
import stat
from pathlib import Path

from rosclaw.connectors.ros.context.collision_assets import (
    REFERENCE_SCHEMA,
    register_prepared_collision_originals,
    staged_collision_resource_pins,
)


def _identity(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_at(root_fd, name, limit):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError("bounded regular original collision resource required")
        chunks, count = [], 0
        while True:
            chunk = os.read(fd, min(65536, limit + 1 - count))
            if not chunk:
                break
            count += len(chunk)
            if count > limit:
                raise ValueError("original collision resource size budget exceeded")
            chunks.append(chunk)
        after = os.fstat(fd)
        if _identity(before) != _identity(after) or count != before.st_size:
            raise ValueError("original collision resource changed during read")
        return b"".join(chunks), _identity(after)
    finally:
        os.close(fd)


def read_prepared_collision_observation(path, expected_sha256, *, reference_only=False):
    """SDK probe opt-in is one fixed readonly source path, not arbitrary file I/O."""
    if type(reference_only) is not bool:
        raise ValueError("explicit boolean collision reference mode required")
    if str(path) != "/assets/collision-asset-originals.json":
        raise ValueError("closed prepared collision manifest path required")
    root_fd = os.open("/assets", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        raw, _ = _read_at(root_fd, "collision-asset-originals.json", 5_000_000)
    finally:
        os.close(root_fd)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("original prepared collision manifest SHA differs")
    if reference_only:
        reference = register_prepared_collision_originals(raw, expected_sha256)
        staged_collision_resource_pins({"prepared_collision_assets": reference})
        return reference
    envelope = {
        "schema_version": "rosclaw.prepared_collision_asset_observation.v1",
        "source": "/assets/collision-asset-originals.json",
        "sha256": expected_sha256,
        "document": raw.decode("utf-8", errors="strict"),
    }
    staged_collision_resource_pins({"prepared_collision_assets": envelope})
    return envelope


class CollisionResourceContinuity:
    """Latch on changed manifest, bytes, file identity or directory substitution."""

    def __init__(self, assets_directory, observations):
        self.root = Path(assets_directory)
        self.observation = observations.get("prepared_collision_assets")
        if (
            isinstance(self.observation, dict)
            and self.observation.get("schema_version") == REFERENCE_SCHEMA
        ):
            root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                raw, _ = _read_at(root_fd, "collision-asset-originals.json", 5_000_000)
            finally:
                os.close(root_fd)
            if (
                register_prepared_collision_originals(raw, self.observation.get("sha256"))
                != self.observation
            ):
                raise ValueError("actual original collision document differs from reference")
        self.pins = staged_collision_resource_pins(observations)
        self.identities = {}
        self.root_identity = self.resources_identity = None
        self.failed = None
        self.verify()

    def verify(self, observations=None):
        if self.failed is not None:
            raise ValueError("collision resource continuity permanently latched: " + self.failed)
        try:
            if (
                observations is not None
                and observations.get("prepared_collision_assets") != self.observation
            ):
                raise ValueError(
                    "current prepared collision observation differs from immutable Body source"
                )
            if not self.pins:
                return {}
            root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                info = os.fstat(root_fd)
                identity = info.st_dev, info.st_ino
                if self.root_identity is not None and identity != self.root_identity:
                    raise ValueError("original collision asset directory replaced")
                self.root_identity = identity
                resource_fd = os.open(
                    "collision-resources",
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=root_fd,
                )
                try:
                    info = os.fstat(resource_fd)
                    identity = info.st_dev, info.st_ino
                    if self.resources_identity is not None and identity != self.resources_identity:
                        raise ValueError("original collision resource directory replaced")
                    self.resources_identity = identity
                    for path, digest in self.pins.items():
                        resource = path.startswith("collision-resources/")
                        raw, identity = _read_at(
                            resource_fd if resource else root_fd,
                            path.rsplit("/", 1)[-1],
                            8_000_000 if resource else 5_000_000,
                        )
                        if hashlib.sha256(raw).hexdigest() != digest:
                            raise ValueError("original collision resource bytes changed: " + path)
                        if path in self.identities and identity != self.identities[path]:
                            raise ValueError(
                                "original collision resource file identity changed: " + path
                            )
                        self.identities[path] = identity
                        if (
                            path == "collision-asset-originals.json"
                            and isinstance(self.observation, dict)
                            and self.observation.get("schema_version") == REFERENCE_SCHEMA
                            and register_prepared_collision_originals(raw, digest)
                            != self.observation
                        ):
                            raise ValueError(
                                "current collision reference differs from actual original"
                            )
                finally:
                    os.close(resource_fd)
            finally:
                os.close(root_fd)
            return dict(self.pins)
        except (OSError, ValueError, TypeError) as exc:
            self.failed = str(exc)
            raise ValueError("collision resource continuity rejected: " + self.failed) from exc


def prewarm_prepared_collision_facts():
    """Parse immutable offline bytes before admission, never reuse live facts.

    Fresh resource continuity and current source checks are still constructed
    after enrollment. This does not issue observations, acknowledge startup,
    create a Body, or open a daemon/ROS/control connection.
    """
    if not Path("/.dockerenv").is_file() or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise ValueError("owned isolated SDK required for passive byte preparation")
    if not Path("/assets/collision-asset-originals.json").exists():
        return None
    root_fd = os.open("/assets", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        raw, _ = _read_at(root_fd, "collision-asset-originals.json", 5_000_000)
    finally:
        os.close(root_fd)
    reference = register_prepared_collision_originals(raw, hashlib.sha256(raw).hexdigest())
    staged_collision_resource_pins({"prepared_collision_assets": reference})
    return {"scope": "OFFLINE_BYTE_FACTS_ONLY", "execution_authority": "NONE"}
