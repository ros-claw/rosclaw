"""NA-FIX-7（规格 §22.4/P1-4）：ModelCredentialBroker。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rosclaw.agentd.credentials import ModelCredentialBroker


def _seed_legacy(home: Path, values: dict[str, str]) -> None:
    path = home / "agentd"
    path.mkdir(parents=True, exist_ok=True)
    (path / "credentials.json").write_text(json.dumps({"environment": values}), encoding="utf-8")


class TestBroker:
    def test_legacy_read_once_and_pi_key_bridge(self, tmp_path: Path, monkeypatch) -> None:
        _seed_legacy(tmp_path, {"ROSCLAW_KIMI_API_KEY": "sk-legacy-dummy"})
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("ROSCLAW_KIMI_API_KEY", raising=False)
        broker = ModelCredentialBroker(tmp_path)
        injected = broker.migrate_legacy_once()
        assert "ROSCLAW_KIMI_API_KEY" in injected
        # legacy → Pi env 键桥接（进程内，不落地）。
        assert os.environ["KIMI_API_KEY"] == "sk-legacy-dummy"
        # 让 monkeypatch 负责清理 broker 写进 environ 的键（防跨测试泄漏）。
        monkeypatch.setenv("KIMI_API_KEY", "sk-legacy-dummy")
        monkeypatch.setenv("ROSCLAW_KIMI_API_KEY", "sk-legacy-dummy")
        # 一次性：第二次不再注入。
        assert broker.migrate_legacy_once() == ()

    def test_env_precedence_and_source_report(self, tmp_path: Path, monkeypatch) -> None:
        _seed_legacy(tmp_path, {"ROSCLAW_KIMI_API_KEY": "sk-legacy-dummy"})
        monkeypatch.setenv("KIMI_API_KEY", "sk-explicit")
        monkeypatch.delenv("ROSCLAW_KIMI_API_KEY", raising=False)
        broker = ModelCredentialBroker(tmp_path)
        broker.migrate_legacy_once()
        # 显式 env 优先——不被 legacy 覆盖。
        assert os.environ["KIMI_API_KEY"] == "sk-explicit"
        report = {r["provider"]: r for r in broker.source_report() if "provider" in r}
        assert report["kimi-code"]["source"] == "env"
        assert report["kimi-code"]["fingerprint"]
        assert "sk-explicit" not in json.dumps(report)

    def test_no_credential_reports_none(self, tmp_path: Path, monkeypatch) -> None:
        for key in ("KIMI_API_KEY", "ROSCLAW_KIMI_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        broker = ModelCredentialBroker(tmp_path)
        assert broker.source_for("kimi-code")["source"] == "none"
