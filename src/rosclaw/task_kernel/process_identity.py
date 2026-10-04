"""Linux process ownership evidence for explicitly managed independent sessions.

No argv/environment plaintext is persisted; unknown identities are never adopted.
"""
from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import signal
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int
    pgid: int
    sid: int
    uid: int
    boot_id: str
    cwd: str
    command_sha256: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def parse(cls, raw: str) -> ProcessIdentity | None:
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
                return None
            if any(type(data[k]) is not int for k in ('pid', 'start_ticks', 'pgid', 'sid', 'uid')):
                return None
            if any(not isinstance(data[k], str) or not data[k] for k in ('boot_id', 'cwd', 'command_sha256')):
                return None
            identity = cls(**data)
            if (identity.pid <= 0 or identity.start_ticks <= 0 or identity.uid < 0
                    or identity.pgid != identity.pid or identity.sid != identity.pid):
                return None
            return identity
        except (TypeError, ValueError):
            return None

    @classmethod
    def capture(cls, pid: int) -> ProcessIdentity | None:
        try:
            path = Path('/proc') / str(pid)
            fields = (path / 'stat').read_text().rpartition(')')[2].split()
            if fields[0] in ('Z', 'X'):
                return None
            return cls(pid=pid, start_ticks=int(fields[19]), pgid=int(fields[2]),
                       sid=int(fields[3]), uid=path.stat().st_uid,
                       boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                       cwd=os.readlink(path / 'cwd'),
                       command_sha256=hashlib.sha256((path / 'cmdline').read_bytes()).hexdigest())
        except (OSError, ValueError, IndexError):
            return None

    def matches(self) -> bool:
        return self.capture(self.pid) == self

    def same_birth(self, other: ProcessIdentity | None) -> bool:
        # Children may exec or change cwd after a proved group snapshot. Their
        # kernel birth/group identity still proves ownership; PID reuse does not.
        return other is not None and all(
            getattr(self, key) == getattr(other, key)
            for key in ('pid', 'start_ticks', 'pgid', 'sid', 'uid', 'boot_id')
        )


def group_members(identity: ProcessIdentity) -> list[ProcessIdentity]:
    """Snapshot live members of this proved group/session; zombies are not live.

    Called only after leader proof. Failure to inspect a relevant member is
    unresolved, not evidence that the group is empty.
    """
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / 'stat').read_text().rpartition(')')[2].split()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError('process membership inspection unavailable') from exc
        try:
            state, pgid, sid = fields[0], int(fields[2]), int(fields[3])
        except (IndexError, ValueError) as exc:
            raise ValueError('process membership inspection malformed') from exc
        if pgid != identity.pgid or sid != identity.sid or state in ('Z', 'X'):
            continue
        member = ProcessIdentity.capture(int(path.name))
        if member is None:
            # It may have disappeared between stat and capture. Confirm that
            # before treating it as absent; otherwise cleanup remains unknown.
            if path.exists():
                try:
                    state = (path / 'stat').read_text().rpartition(')')[2].split()[0]
                except FileNotFoundError:
                    continue
                if state not in ('Z', 'X'):
                    raise ValueError('live group member identity unavailable')
            continue
        if (member.uid != identity.uid or member.boot_id != identity.boot_id
                or member.start_ticks < identity.start_ticks):
            raise ValueError('group ownership changed')
        members.append(member)
    return members


def _pidfd_open(pid: int) -> int:
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # This machine's uv CPython lacks the Python bindings, while glibc and
    # Linux support the public pidfd APIs. No architecture syscall numbers.
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, 'pidfd_open', None)
    if function is None:
        raise NotImplementedError('pidfd API unavailable')
    function.argtypes = [ctypes.c_int, ctypes.c_uint]
    function.restype = ctypes.c_int
    fd = function(pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_open failed')
    return int(fd)


def _pidfd_signal(fd: int, sig: int) -> None:
    if hasattr(signal, 'pidfd_send_signal'):
        signal.pidfd_send_signal(fd, sig)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, 'pidfd_send_signal', None)
    if function is None:
        raise NotImplementedError('pidfd API unavailable')
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(fd, sig, None, 0) < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_send_signal failed')


def signal_owned_members(members: list[ProcessIdentity], sig: int) -> bool:
    """Signal exact live kernel process handles, never a reused numeric PID/PGID."""
    for member in members:
        try:
            fd = _pidfd_open(member.pid)
        except ProcessLookupError:
            continue
        try:
            if not member.same_birth(ProcessIdentity.capture(member.pid)):
                # A vanished/zombie member needs no signal; a changed live
                # identity is unresolved, even if it uses the same number.
                if ProcessIdentity.capture(member.pid) is not None:
                    return False
                continue
            with contextlib.suppress(ProcessLookupError):
                _pidfd_signal(fd, sig)
        finally:
            os.close(fd)
    return True
