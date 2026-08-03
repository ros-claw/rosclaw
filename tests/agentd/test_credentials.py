"""Owner-only persisted credential tests for AgentD CLI."""

import io
import json
import os
import stat

import pytest

from rosclaw.agentd.cli import main as agentd_main
from rosclaw.agentd.credentials import AgentCredentialStore, CredentialStoreError


def test_store_round_trip_and_environment_precedence(tmp_path, monkeypatch):
    store = AgentCredentialStore(tmp_path)
    store.set("ROSCLAW_KIMI_API_KEY", "stored-value")
    monkeypatch.setenv("ROSCLAW_KIMI_API_KEY", "explicit-value")

    assert store.inject() == ("ROSCLAW_KIMI_API_KEY",)
    assert os.environ["ROSCLAW_KIMI_API_KEY"] == "explicit-value"
    assert store.status("ROSCLAW_KIMI_API_KEY")["stored"] is True
    assert "stored-value" not in json.dumps(store.status("ROSCLAW_KIMI_API_KEY"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission semantics")
def test_store_uses_owner_only_permissions(tmp_path):
    store = AgentCredentialStore(tmp_path)
    store.set("ROSCLAW_KIMI_API_KEY", "secret")

    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_store_rejects_symlink(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("do not overwrite", encoding="utf-8")
    credential_path = tmp_path / "agentd" / "credentials.json"
    credential_path.parent.mkdir(parents=True)
    credential_path.symlink_to(outside)

    with pytest.raises(CredentialStoreError, match="regular file"):
        AgentCredentialStore(tmp_path)
    assert outside.read_text(encoding="utf-8") == "do not overwrite"


def test_cli_set_status_delete_never_prints_secret(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("top-secret-value\n"))
    rc = agentd_main(["--home", str(tmp_path), "credential", "set", "--provider", "kimi-code"])
    assert rc == 0
    output = capsys.readouterr().out
    assert "top-secret-value" not in output
    assert json.loads(output)["stored"] is True

    rc = agentd_main(["--home", str(tmp_path), "credential", "status", "--provider", "kimi-code"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["stored"] is True

    rc = agentd_main(["--home", str(tmp_path), "credential", "delete", "--provider", "kimi-code"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["stored"] is False


def test_doctor_cli_automatically_loads_stored_credential(tmp_path, monkeypatch, capsys):
    store = AgentCredentialStore(tmp_path)
    store.set("ROSCLAW_KIMI_API_KEY", "persisted-value")
    monkeypatch.delenv("ROSCLAW_KIMI_API_KEY", raising=False)

    def fake_doctor(home):
        assert home == tmp_path
        assert os.environ["ROSCLAW_KIMI_API_KEY"] == "persisted-value"
        return {"status": "READY"}

    monkeypatch.setattr("rosclaw.agentd.cli.doctor", fake_doctor)
    rc = agentd_main(["--home", str(tmp_path), "doctor"])

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {"status": "READY"}
