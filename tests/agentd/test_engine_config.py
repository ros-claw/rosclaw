"""PNA-11（规格 §5/§31）：engine 配置与优先级。"""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.agentd.config import load_agent_config


def test_default_engine_is_legacy(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("agent:\n  enabled: true\n", encoding="utf-8")
    assert load_agent_config(tmp_path / "config.yaml").engine == "legacy"


def test_engine_pi_explicit(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "agent:\n  enabled: true\n  engine: pi\n", encoding="utf-8"
    )
    assert load_agent_config(tmp_path / "config.yaml").engine == "pi"


def test_engine_invalid_rejected(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "agent:\n  enabled: true\n  engine: turbo\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="pi|legacy"):
        load_agent_config(tmp_path / "config.yaml")
