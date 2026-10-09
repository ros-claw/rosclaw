"""Owned host outcome survives diagnostic export failure; no Docker or ROS."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def host_fixture(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    host = importlib.import_module("generic_bootstrap_host")
    source = tmp_path / "source"
    relative = "integrations/ros_probe/acceptance/generic_bootstrap_host.py"
    launcher = source / relative
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(Path(host.__file__).read_bytes())
    monkeypatch.setattr(host, "__file__", str(launcher))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = b'{"fixture":"mock-workspace"}'
    (workspace / "generic-stack-source-manifest.json").write_bytes(manifest)
    declaration = tmp_path / "declaration.json"
    declaration.write_text("{}")
    output = tmp_path / "output"
    calls = []
    monkeypatch.setattr(host, "bootstrap_launch_plan", lambda *args: {"fixture": "plan"})
    monkeypatch.setattr(
        host,
        "read_prepared_generic_stack",
        lambda *args: {
            "captured_files": {},
            "manifest_sha256": host.hashlib.sha256(manifest).hexdigest(),
        },
    )

    def check_output(argv, **kwargs):
        assert argv[0] == "git"
        if argv[1:3] == ["rev-parse", "HEAD"]:
            return "f" * 40
        if argv[1:3] == ["status", "--porcelain"]:
            return b""
        assert argv[1:3] == ["ls-files", "-z"]
        return relative.encode() + b"\0"

    def run(argv, **kwargs):
        assert argv[0] == "docker"
        calls.append(argv[1])
        if argv[1] == "create":
            return SimpleNamespace(stdout=("a" * 64).encode(), returncode=0)
        assert argv[1] in {"start", "logs"}
        if argv[1] == "logs":
            kwargs["stdout"].write(b"original stdout")
            kwargs["stderr"].write(b"original stderr")
        return SimpleNamespace(returncode=0)

    def stop(*args):
        calls.append("verified-stop")
        return {"container_stopped": True}

    def wait(*args, **kwargs):
        kwargs["check_source"]()
        calls.append("wait")
        return "IMMUTABLE_DEADLINE_REACHED"

    monkeypatch.setattr(host.subprocess, "check_output", check_output)
    monkeypatch.setattr(host.subprocess, "run", run)
    monkeypatch.setattr(host, "inspect_owned_container", lambda *args: {})
    monkeypatch.setattr(host, "stop_owned_container", stop)
    monkeypatch.setattr(host, "wait_owned_container", wait)
    return host, source, workspace, declaration, output, calls


@pytest.mark.parametrize("original_failure", [False, True])
@pytest.mark.parametrize("diagnostic_failure", [False, True])
def test_original_outcome_and_cleanup_survive_sdk_export_error(
    host_fixture, monkeypatch, original_failure, diagnostic_failure
):
    host, source, workspace, declaration, output, calls = host_fixture

    def export(container_id, owner, target, *, expected_image):
        assert container_id == "a" * 64 and expected_image == host.IMAGE_ID
        assert target == output / "sdk-logs"
        assert calls[-2:] == ["verified-stop", "logs"]
        calls.append("sdk-export")
        if diagnostic_failure:
            raise TimeoutError("fixture diagnostic timeout")
        return {"status": "CAPTURED"}

    monkeypatch.setattr(host, "copy_owned_sdk_logs", export)
    if original_failure:

        def fail(*args, **kwargs):
            raise ValueError("original source failure")

        monkeypatch.setattr(host, "wait_owned_container", fail)
        with pytest.raises(ValueError, match="original source failure"):
            host.run_owned_bootstrap(source, workspace, declaration, output, seconds=10, domain=80)
    else:
        result = host.run_owned_bootstrap(
            source, workspace, declaration, output, seconds=10, domain=80
        )
        assert result == "IMMUTABLE_DEADLINE_REACHED"
    recorded = json.loads((output / "host-result.json").read_text())
    assert recorded["outcome"] == (
        "SOURCE_OR_CONTAINER_FAILURE" if original_failure else "IMMUTABLE_DEADLINE_REACHED"
    )
    assert recorded["cleanup_verified"] is True and recorded["log_capture"] == "CAPTURED"
    assert recorded["sdk_log_capture"]["status"] == ("FAILED" if diagnostic_failure else "CAPTURED")
    assert recorded["physical_acceptance"] == "NOT_EVALUATED"
    assert recorded["controller_activation"] is False
    assert (output / "container.original-stdout").read_bytes() == b"original stdout"
    assert calls[-1] == "sdk-export"


def test_unverified_cleanup_prevents_sdk_copy(host_fixture, monkeypatch):
    host, source, workspace, declaration, output, calls = host_fixture

    def refuse(*args):
        calls.append("refused-stop")
        raise ValueError("foreign container fixture")

    monkeypatch.setattr(host, "stop_owned_container", refuse)
    monkeypatch.setattr(
        host, "copy_owned_sdk_logs", lambda *args, **kwargs: pytest.fail("copy after refusal")
    )
    with pytest.raises(ValueError, match="foreign container fixture"):
        host.run_owned_bootstrap(source, workspace, declaration, output, seconds=10, domain=80)
    recorded = json.loads((output / "host-result.json").read_text())
    assert recorded["cleanup_verified"] is False
    assert recorded["sdk_log_capture"]["status"] == "NOT_ATTEMPTED"
    assert calls[-1] == "refused-stop"
