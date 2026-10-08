"""Read-only correspondence of frozen source and one owned Gazebo process.

This inspects Linux process identity, argv, the declared private GZ partition
and actual ELF mappings. It starts no World, Node, service or actuator. Loaded
source correspondence cannot establish backend cache behavior, physical stop,
cleaning success or robot authorization.
"""

import hashlib
import os
import re
import shlex
import stat
from pathlib import Path
from xml.etree import ElementTree as ET

from backend_probe_world import bounded_source
from probe_controller_ipc import process_identity
from probe_scene_geometry import decode_scene_json

from experiments import gazebo_arguments


def file_identity(path):
    path = Path(path)
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or path.is_symlink():
        raise ValueError("regular frozen world source file required")
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def mapped_file_rows(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 16_000_000:
        raise ValueError("bounded original owned process maps required")
    rows = {}
    for line in raw.decode("utf-8", errors="strict").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        if fields[5].endswith(" (deleted)"):
            # Unrelated unlinked shared-memory mappings are normal. A deleted
            # required ELF cannot match its exact declared path below.
            continue
        try:
            major, minor = [int(x, 16) for x in fields[3].split(":")]
            identity = (os.makedev(major, minor), int(fields[4]))
        except (ValueError, OverflowError) as exc:
            raise ValueError("original owned World ELF map identity malformed") from exc
        old = rows.setdefault(fields[5], identity)
        if old != identity:
            raise ValueError("owned World mapped source path has ambiguous identities")
    return rows


def bounded_process_file(path, limit):
    # procfs reports size zero even when cmdline/maps/environ have bytes.
    file_identity(path)
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit:
        raise ValueError("bounded original owned process source required")
    return raw


class WorldSourceOwner:
    """Pin a launcher-owned PID and reopen the exclusive final source bundle."""

    def __init__(
        self,
        bundle_directory,
        *,
        world_pid,
        world_uid,
        partition,
        seed,
        physics_plugin,
        physics_plugin_sha256,
        contact_plugin_sha256,
    ):
        self.directory = Path(bundle_directory).resolve()
        if type(partition) is not str or not re.fullmatch(
            r"rosclaw_backend_[a-f0-9]{32}", partition
        ):
            raise ValueError("explicit fresh owned GZ partition required")
        self.partition = partition
        self.peer = process_identity(world_pid, world_uid)
        self.proc = Path(f"/proc/{world_pid}")
        manifest_path = self.directory / "backend-world-bundle.json"
        raw = bounded_source(manifest_path)
        manifest = decode_scene_json(raw)
        if (
            manifest.get("schema_version") != "rosclaw.backend_world_source_bundle.v1"
            or manifest.get("source_contact_library_sha256") != contact_plugin_sha256
            or type(manifest.get("output_hashes")) is not dict
            or not 1 <= len(manifest["output_hashes"]) <= 100
            or not {
                "world-source/world.sdf",
                "world-source/librosclaw_passive_contacts.so",
            }.issubset(manifest["output_hashes"])
        ):
            raise ValueError("complete final owned source bundle manifest required")
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.files = {}
        self.hashes = {}
        self._pin(manifest_path, self.manifest_sha256)
        for name, sha in manifest["output_hashes"].items():
            if (
                type(name) is not str
                or type(sha) is not str
                or not re.fullmatch(r"[a-f0-9]{64}", sha)
            ):
                raise ValueError("bounded exact source bundle hash entries required")
            p = self.directory / name
            if not p.resolve().is_relative_to(self.directory):
                raise ValueError("world source bundle path escapes exclusive directory")
            self._pin(p, sha)
        self.world = self.directory / "world-source/world.sdf"
        self.contact = self.directory / "world-source/librosclaw_passive_contacts.so"
        self.physics = Path(physics_plugin).absolute()
        self._pin(self.physics, physics_plugin_sha256)
        self._pin(self.contact, contact_plugin_sha256)
        tree = ET.fromstring(bounded_source(self.world))
        physics = [
            p for p in tree.findall("world/plugin") if p.get("name") == "rosclaw::PassivePhysics"
        ]
        contacts = [
            p for p in tree.findall("world/plugin") if p.get("name") == "rosclaw::PassiveContacts"
        ]
        if (
            len(physics) != 1
            or physics[0].get("filename") != str(self.physics)
            or len(contacts) != 2
            or any(p.get("filename") != self.contact.name for p in contacts)
        ):
            raise ValueError("exact declared World plugin source paths required")
        self.argv = gazebo_arguments(self.world, seed)
        self.fault = None

    def _pin(self, path, sha):
        path = Path(path)
        before = file_identity(path)
        raw = bounded_source(path, 100_000_000)
        if hashlib.sha256(raw).hexdigest() != sha or file_identity(path) != before:
            raise ValueError("frozen actual world source SHA or file identity differs")
        self.files[path] = before
        self.hashes[path] = sha

    def check(self):
        """Fast immutable-file and actual-process check; any rejection latches."""
        if self.fault:
            raise ValueError("owned world source rejection remains latched")
        try:
            if process_identity(*self.peer[:2]) != self.peer:
                raise ValueError("owned world source process identity changed")
            for p, identity in self.files.items():
                if file_identity(p) != identity:
                    raise ValueError("frozen owned World source file changed")
            argv_raw = bounded_process_file(self.proc / "cmdline", 65536)
            args = [x.decode("utf-8") for x in argv_raw.split(b"\0") if x]
            actual = shlex.split(args[0]) if len(args) == 1 else args
            if actual != self.argv:
                raise ValueError(
                    "actual owned Gazebo process argv differs from frozen World launch"
                )
            environment = bounded_process_file(self.proc / "environ", 2_000_000)
            partitions = [
                v.split(b"=", 1)[1]
                for v in environment.split(b"\0")
                if v.startswith(b"GZ_PARTITION=")
            ]
            if partitions != [self.partition.encode("ascii")]:
                raise ValueError("actual owned Gazebo process GZ partition differs")
            rows = mapped_file_rows(bounded_process_file(self.proc / "maps", 16_000_000))
            required = [self.physics, self.contact]
            sdk = [Path(p) for p in rows if Path(p).name == "libgz-sim8.so.8.15.0"]
            commands = [
                Path(p) for p in rows if Path(p).name == "libgz-sim8-user-commands-system.so.8.15.0"
            ]
            if len(sdk) != 1 or len(commands) != 1:
                raise ValueError(
                    "actual pinned Gazebo8.15 core and owned scene command system must be mapped"
                )
            required += sdk + commands
            mapped = []
            for p in required:
                identity = file_identity(p)
                if rows.get(str(p)) != identity[:2]:
                    raise ValueError(
                        "actual owned Gazebo process lacks the exact mapped source ELF"
                    )
                if p not in self.files:
                    source = bounded_source(p, 100_000_000)
                    if not source.startswith(b"\x7fELF"):
                        raise ValueError("actual mapped source is not an ELF library")
                    self._pin(p, hashlib.sha256(source).hexdigest())
                mapped.append(
                    {
                        "path": str(p),
                        "sha256": self.hashes[p],
                        "device": identity[0],
                        "inode": identity[1],
                    }
                )
            if process_identity(*self.peer[:2]) != self.peer:
                raise ValueError("owned Gazebo process changed during original source check")
            return {
                "evidence_role": "actual_frozen_source_and_mapped_owned_world_process_correspondence",
                "pid": self.peer[0],
                "uid": self.peer[1],
                "process_starttime": self.peer[2],
                "gazebo_argv_sha256": hashlib.sha256(argv_raw).hexdigest(),
                "GZ_partition": self.partition,
                "bundle_manifest_sha256": self.manifest_sha256,
                "world_source_sha256": self.hashes[self.world],
                "actual_mapped_libraries": mapped,
                "loaded_source_correspondence": True,
                "backend_health_admitted": False,
                "physical_acceptance": "NOT_VERIFIED",
                "authorization": False,
            }
        except (ValueError, OSError, UnicodeError) as exc:
            self.fault = str(exc)
            raise ValueError(f"owned world source rejected: {exc}") from exc
