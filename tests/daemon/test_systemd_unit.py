"""Static least-privilege contracts for the reference rosclawd unit."""

from __future__ import annotations

import configparser
from pathlib import Path


def _service_contract() -> configparser.SectionProxy:
    unit = Path(__file__).resolve().parents[2] / "deploy" / "systemd" / "rosclawd.service"
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    with unit.open(encoding="utf-8") as stream:
        parser.read_file(stream)
    return parser["Service"]


def test_systemd_unit_uses_dedicated_identity_and_private_state() -> None:
    service = _service_contract()

    assert service["User"] == "rosclaw-hw"
    assert service["Group"] == "rosclaw-agent"
    assert service["SupplementaryGroups"] == "dialout"
    assert service["RuntimeDirectory"] == "rosclaw"
    assert service["RuntimeDirectoryMode"] == "0750"
    assert service["StateDirectory"] == "rosclaw"
    assert service["StateDirectoryMode"] == "0700"
    assert service["UMask"] == "0077"
    assert "--socket-mode 0660" in service["ExecStart"]
    assert "--socket-group rosclaw-agent" in service["ExecStart"]


def test_systemd_unit_fails_closed_and_drops_process_privileges() -> None:
    service = _service_contract()
    required = {
        "DevicePolicy": "closed",
        "NoNewPrivileges": "true",
        "PrivateTmp": "true",
        "ProtectSystem": "strict",
        "ProtectHome": "true",
        "ProtectControlGroups": "true",
        "ProtectClock": "true",
        "ProtectHostname": "true",
        "ProtectKernelLogs": "true",
        "ProtectKernelModules": "true",
        "ProtectKernelTunables": "true",
        "KeyringMode": "private",
        "LockPersonality": "true",
        "MemoryDenyWriteExecute": "true",
        "RestrictAddressFamilies": "AF_UNIX AF_INET AF_INET6 AF_NETLINK",
        "RestrictNamespaces": "true",
        "RestrictRealtime": "true",
        "RestrictSUIDSGID": "true",
        "RemoveIPC": "true",
        "SystemCallArchitectures": "native",
        "CapabilityBoundingSet": "",
        "AmbientCapabilities": "",
    }

    assert {key: service.get(key) for key in required} == required
    assert service.get("DeviceAllow") is None
