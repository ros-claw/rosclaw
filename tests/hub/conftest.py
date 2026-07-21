"""Hub test trust configuration kept separate from packaged runtime trust."""

from __future__ import annotations

from pathlib import Path

import pytest

HUB_KEYS = Path(__file__).parent.parent / "fixtures" / "hub_keys"


@pytest.fixture(autouse=True)
def _hub_fixture_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROSCLAW_HUB_TRUST_STORE", str(HUB_KEYS / "trust.json"))
    monkeypatch.setenv("ROSCLAW_HUB_SIGNING_KEY", str(HUB_KEYS / "fixture-private.pem"))
    monkeypatch.setenv("ROSCLAW_HUB_SIGNING_KEY_ID", "rosclaw-hub-fixture-v1")
