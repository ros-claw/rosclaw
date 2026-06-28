"""Tests for Cosmos-style unstructured provider output fallback."""

from __future__ import annotations

import pytest

from rosclaw.provider.normalizer import ProviderResultNormalizer


class TestCosmosSchemaFallback:
    """Cover the safe fallback path for Cosmos/VLM free-text outputs."""

    def test_unstructured_cosmos_risk_text_returns_guarded_fallback(self):
        raw = (
            "After reviewing the image, the workspace contains several loose cables and a "
            "stool in the path. This represents a high risk of tripping or collision. "
            "I recommend clearing the area before any motion."
        )
        result = ProviderResultNormalizer.normalize(raw, capability="vlm.risk_assessment")
        assert result.schema_valid is False
        assert result.fallback_parse is True
        assert result.requires_guard is True
        assert result.risk_score == pytest.approx(0.8)
        assert "loose cables" in result.scene.lower()

    def test_cosmos_markdown_json_is_validated(self):
        raw = """```json
{
  "scene": "lab bench",
  "objects": [{"label": "screwdriver"}],
  "physical_risks": [{"description": "sharp object", "severity": "medium"}],
  "risk_score": 0.5,
  "executable": true,
  "requires_guard": false,
  "reasoning": "Looks okay"
}
```"""
        result = ProviderResultNormalizer.normalize(raw, capability="vlm.risk_assessment")
        assert result.schema_valid is True
        assert result.fallback_parse is False
        assert result.risk_score == pytest.approx(0.5)
        assert result.executable is True

    def test_cosmos_partial_json_enforces_guard(self):
        raw = '{"scene": "bench", "risk_score": 0.2}'
        result = ProviderResultNormalizer.normalize(raw, capability="vlm.risk_assessment")
        assert result.schema_valid is False
        assert result.fallback_parse is True
        assert result.requires_guard is True
        assert result.risk_score == pytest.approx(0.2)

    def test_cosmos_low_risk_text(self):
        raw = "The scene is clear and organized, presenting only a low risk."
        result = ProviderResultNormalizer.normalize(raw)
        assert result.risk_score == pytest.approx(0.2)
        assert result.requires_guard is True
