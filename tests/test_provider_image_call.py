"""Tests for provider image invocation and normalized output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from rosclaw.cli import cmd_provider_invoke


class TestProviderImageCall:
    """Cover ``rosclaw provider invoke --image ... --output ...``."""

    @pytest.fixture
    def dummy_image(self, tmp_path: Path) -> Path:
        img = tmp_path / "scene.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"" * 16)
        return img

    def test_provider_image_call_writes_safe_normalized_output(self, dummy_image, tmp_path, capsys):
        """When no provider runtime or endpoint is available, the CLI returns a safe structured fallback."""
        output = tmp_path / "provider_result.json"
        args = argparse.Namespace(
            provider_id="cosmos-reason2-lan",
            provider_id_opt=None,
            input="{}",
            capability="vlm.risk_assessment",
            question="Is this scene safe?",
            image_path=str(dummy_image),
            output_path=str(output),
            json=True,
            trace_id=None,
        )
        assert cmd_provider_invoke(args) == 0

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["provider_id"] == "cosmos-reason2-lan"
        assert data["image"] is True
        assert data["capability"] == "vlm.risk_assessment"
        normalized = data["normalized"]
        assert normalized["requires_guard"] is True
        assert normalized["executable"] is False
        assert normalized["schema_valid"] is True
        assert normalized["fallback_parse"] is False

    def test_provider_image_call_prints_json(self, dummy_image, tmp_path, capsys):
        output = tmp_path / "provider_result.json"
        args = argparse.Namespace(
            provider_id="cosmos-reason2-lan",
            provider_id_opt=None,
            input="{}",
            capability="vlm.risk_assessment",
            question=None,
            image_path=str(dummy_image),
            output_path=str(output),
            json=True,
            trace_id=None,
        )
        assert cmd_provider_invoke(args) == 0
        # The command prints human-readable prefixes plus JSON; read the saved file for a clean parse.
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["normalized"]["requires_guard"] is True

    def test_provider_image_call_missing_image_errors(self, tmp_path):
        args = argparse.Namespace(
            provider_id="cosmos-reason2-lan",
            provider_id_opt=None,
            input="{}",
            capability="vlm.risk_assessment",
            question=None,
            image_path=str(tmp_path / "missing.png"),
            output_path=None,
            json=True,
            trace_id=None,
        )
        assert cmd_provider_invoke(args) == 1
