from __future__ import annotations

import pytest

from rosclaw.knowledge.contracts import ReferenceContextV2
from rosclaw.knowledge.facade import KnowledgeFacade
from rosclaw.knowledge.know_client import KnowUnavailableError
from rosclaw.knowledge.service_manager import KnowledgeServiceConfig, KnowledgeServiceManager


def test_disabled_reference_pack_fails_explicitly():
    facade = KnowledgeFacade(KnowledgeServiceManager(KnowledgeServiceConfig(mode="disabled")))
    with pytest.raises(KnowUnavailableError, match="disabled"):
        facade.reference_pack(query="anything", context=ReferenceContextV2(task="test"))


def test_missing_optional_packages_do_not_crash_core(monkeypatch, tmp_path):
    real_import = __import__

    def blocked(name, *args, **kwargs):
        if name.startswith(("rosclaw_know", "rosclaw_how")):
            raise ImportError("fixture missing optional package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    manager = KnowledgeServiceManager(
        KnowledgeServiceConfig(
            mode="inprocess", know_store_mode="embedded", know_store_path=str(tmp_path / "know")
        )
    )
    assert manager.health()["status"] == "degraded"
