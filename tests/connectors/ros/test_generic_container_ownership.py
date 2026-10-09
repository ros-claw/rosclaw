"""Ownership refusal and immutable outer deadline boundaries; no Docker access."""

import importlib
import time
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    ownership = importlib.import_module("generic_container_ownership")
    host = importlib.import_module("generic_bootstrap_host")
    return ownership, host


@pytest.mark.parametrize(
    "container,owner",
    [
        ("name", "a" * 32),
        ("a" * 63, "a" * 32),
        ("a" * 64, "owner"),
        ("a" * 64, "A" * 32),
        (None, "a" * 32),
        ("a" * 64, None),
    ],
)
def test_unknown_or_name_only_ownership_never_reaches_docker(
    modules, monkeypatch, container, owner
):
    ownership, _ = modules

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid ownership must refuse before Docker")

    monkeypatch.setattr(ownership.subprocess, "run", forbidden)
    with pytest.raises(ValueError):
        ownership.inspect_owned_container(container, owner)


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), True, "later"])
def test_invalid_absolute_deadline_refuses_without_source_or_docker(modules, deadline):
    ownership, _ = modules

    def forbidden():
        raise AssertionError("invalid deadline must refuse before source callbacks")

    with pytest.raises(ValueError):
        ownership.wait_owned_container(
            "a" * 64, "b" * 32, deadline=deadline, check_source=forbidden
        )


def test_expensive_source_check_cannot_renew_deadline_or_start_another_docker_request(
    modules, monkeypatch
):
    ownership, _ = modules

    def forbidden(*args, **kwargs):
        raise AssertionError("no Docker request after the immutable deadline")

    monkeypatch.setattr(ownership, "inspect_owned_container", forbidden)
    deadline = time.monotonic() + 0.03
    assert (
        ownership.wait_owned_container(
            "a" * 64, "b" * 32, deadline=deadline, check_source=lambda: time.sleep(0.04)
        )
        == "IMMUTABLE_DEADLINE_REACHED"
    )


def test_bind_paths_and_declarations_cannot_become_shell_code(modules):
    _, host = modules
    malicious = Path("/fixture/$(touch injected); quote' space")
    argv = host.bootstrap_container_command(
        malicious,
        Path("/known/workspace"),
        Path("/known/output"),
        malicious,
        owner="b" * 32,
        domain=86,
        seconds=30,
    )
    shell = argv[argv.index("-c") + 1]
    assert str(malicious) not in shell
    assert 'exec python3 "$@"' in shell
    assert "type=bind,src=" + str(malicious) + ",dst=/workspace,readonly" in argv
    assert "type=bind,src=" + str(malicious) + ",dst=/bootstrap-declaration.json,readonly" in argv
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--pull") + 1] == "never"
    assert host.IMAGE_ID in argv


def test_docker_inspect_timeout_at_outer_deadline_is_deadline_completion(modules, monkeypatch):
    ownership, _ = modules
    stamps = iter([100.0, 100.0, 101.0])
    monkeypatch.setattr(ownership.time, "monotonic", lambda: next(stamps))

    def expired(*args, **kwargs):
        raise ownership.subprocess.TimeoutExpired(["docker", "inspect"], kwargs["timeout"])

    monkeypatch.setattr(ownership, "inspect_owned_container", expired)
    assert (
        ownership.wait_owned_container(
            "a" * 64, "b" * 32, deadline=101.0, check_source=lambda: None
        )
        == "IMMUTABLE_DEADLINE_REACHED"
    )


@pytest.mark.parametrize(
    "output", ["/readonly", "/readonly/source/writable", "/sealed", "/declarations"]
)
def test_writable_mount_cannot_expose_readonly_source_through_an_alias(modules, output):
    _, host = modules
    with pytest.raises(ValueError, match="alias"):
        host.bootstrap_container_command(
            Path("/readonly/source"),
            Path("/sealed/workspace"),
            Path(output),
            Path("/declarations/immutable.json"),
            owner="b" * 32,
            domain=86,
            seconds=30,
        )
