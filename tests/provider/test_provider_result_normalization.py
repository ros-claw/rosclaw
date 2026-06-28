"""Tests for provider result normalization."""

from __future__ import annotations

import json

import pytest

from rosclaw.provider.normalizer import ProviderResultNormalizer


class TestProviderResultNormalizer:
    """Cover the normalization contract for Cosmos-style risk outputs."""

    def test_valid_schema_is_accepted(self):
        raw = json.dumps({
            "scene": "lab bench",
            "objects": [{"label": "screwdriver", "bbox": [0, 0, 10, 10]}],
            "physical_risks": [{"description": "loose cable", "severity": "medium"}],
            "risk_score": 0.4,
            "executable": True,
            "requires_guard": False,
            "reasoning": "Looks safe",
        })
        result = ProviderResultNormalizer.normalize(raw, capability="vlm.risk_assessment")
        assert result.schema_valid is True
        assert result.fallback_parse is False
        assert result.risk_score == pytest.approx(0.4)
        assert result.requires_guard is False
        assert result.executable is True

    def test_nested_normalized_key(self):
        raw = json.dumps({
            "normalized": {
                "scene": "bench",
                "objects": [],
                "physical_risks": [],
                "risk_score": 0.1,
                "executable": False,
                "requires_guard": True,
            }
        })
        result = ProviderResultNormalizer.normalize(raw)
        assert result.schema_valid is True
        assert result.risk_score == pytest.approx(0.1)

    def test_missing_keys_becomes_fallback(self):
        raw = json.dumps({"scene": "bench", "risk_score": 0.2})
        result = ProviderResultNormalizer.normalize(raw)
        assert result.schema_valid is False
        assert result.fallback_parse is True
        assert result.requires_guard is True

    def test_unstructured_text_uses_heuristic(self):
        raw = "The scene has a high risk of collision."
        result = ProviderResultNormalizer.normalize(raw)
        assert result.schema_valid is False
        assert result.fallback_parse is True
        assert result.requires_guard is True
        # "high risk" heuristic should map to 0.8.
        assert result.risk_score == pytest.approx(0.8)

    def test_markdown_fenced_json_is_parsed(self):
        raw = '''```json
        {
            "scene": "bench",
            "objects": [],
            "physical_risks": [],
            "risk_score": 0.0,
            "executable": false,
            "requires_guard": false
        }
        ```'''
        result = ProviderResultNormalizer.normalize(raw)
        assert result.schema_valid is True
        assert result.risk_score == pytest.approx(0.0)

    def test_empty_input_is_guarded(self):
        result = ProviderResultNormalizer.normalize("")
        assert result.schema_valid is False
        assert result.requires_guard is True
        assert result.risk_score is None
