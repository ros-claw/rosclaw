"""SDK owned-process fault contracts use a synthetic proc tree; no real signals."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("simulation_owned_process_fault")
    proc = tmp_path / "proc"
    proc.mkdir()
    root = tmp_path / "evidence"
    root.mkdir()

    def stat(pid, parent, start):
        return f"{pid} (synthetic) " + " ".join(
            ["S", str(parent)] + ["0"] * 17 + [str(start)] + ["0"] * 5
        )

    for pid, parent in [(101, 1), (102, 101), (999, 1)]:
        p = proc / str(pid)
        p.mkdir()
        (p / "stat").write_text(stat(pid, parent, pid * 100))
        (p / "cmdline").write_bytes(b"ros2\0run\0ros_gz_bridge\0parameter_bridge\0")
    (root / "owned-stack-processes.json").write_text(
        json.dumps(
            {
                "scan_bridge": {
                    "pid": 101,
                    "argv": ["ros2", "run", "ros_gz_bridge", "parameter_bridge"],
                }
            }
        )
    )
    original_path = Path
    monkeypatch.setattr(module, "Path", lambda x: proc if str(x) == "/proc" else original_path(x))
    signals = []
    monkeypatch.setattr(module.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    return module, root, proc, signals, stat


def test_only_owned_root_and_descendants_are_paused_and_resumed(fixture):
    module, root, _, signals, _ = fixture
    record = module.pause_owned_process(root, "scan_bridge")
    assert signals == [(102, module.signal.SIGSTOP), (101, module.signal.SIGSTOP)]
    module.resume_owned_process(record)
    assert signals[-2:] == [(101, module.signal.SIGCONT), (102, module.signal.SIGCONT)]
    assert all(pid != 999 for pid, _ in signals)


def test_pid_reuse_rejects_all_resume_signals(fixture):
    module, root, proc, signals, stat = fixture
    record = module.pause_owned_process(root, "scan_bridge")
    signals.clear()
    (proc / "102/stat").write_text(stat(102, 101, 123456))
    with pytest.raises(ValueError, match="identity changed"):
        module.resume_owned_process(record)
    assert signals == []


def test_changed_root_cmdline_is_not_an_owned_fault_target(fixture):
    module, root, proc, signals, _ = fixture
    (proc / "101/cmdline").write_bytes(b"unrelated\0")
    with pytest.raises(ValueError, match="argv or UID changed"):
        module.pause_owned_process(root, "scan_bridge")
    assert signals == []


def test_physics_server_is_never_a_bridge_fault_target(fixture):
    module, root, _, signals, _ = fixture
    with pytest.raises(ValueError, match="nonphysics"):
        module.pause_owned_process(root, "gazebo")
    assert signals == []
