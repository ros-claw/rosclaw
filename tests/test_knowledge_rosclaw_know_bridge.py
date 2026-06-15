"""Tests for optional rosclaw_know integration inside rosclaw.know.interface.

These tests mock the private ``rosclaw_know`` loader so they do not depend on
the compiled assets in ``rosclaw-know/data/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

from rosclaw.know.interface import KnowledgeInterface


@dataclass(frozen=True)
class FakeCuratedPattern:
    pattern_id: str
    safety_label: str
    standard_name: str
    domain: str
    matched_keywords: list[str]
    fix_pattern: str
    failed_attempt: str = ""
    before_code: str = ""
    after_code: str = ""
    cross_domain_hints: list[dict[str, str]] = field(default_factory=list)
    topic_group: str | None = None
    topic_tag: str | None = None
    robot_type: str | None = None
    status: str = "active"
    runtime_eligible: bool = True
    source_tier: str | None = "S_CURATED_VERIFIED"


class TestKnowledgeInterfaceRosclawKnowBridge:
    """Optional enrichment from private rosclaw_know package."""

    def test_enriches_from_registry_when_enabled(self, monkeypatch, tmp_path):
        fake = FakeCuratedPattern(
            pattern_id="anti_windup_pid",
            safety_label="Torque_Overflow",
            standard_name="PID anti-windup",
            domain="Control_Locomotion",
            matched_keywords=["torque", "saturation", "windup"],
            fix_pattern="Clamp the integral term",
            failed_attempt="Increase Ki indefinitely",
        )

        import rosclaw.know.interface as ki_mod

        monkeypatch.setattr(ki_mod, "_load_rosclaw_know_patterns", lambda: [fake])

        ki = KnowledgeInterface(
            robot_id="test",
            assets_path=str(tmp_path),
            use_rosclaw_know_registry=True,
        )
        ki._do_initialize()

        assert "anti_windup_pid" in ki._patterns
        pattern = ki._patterns["anti_windup_pid"]
        assert pattern["symptom"] == "PID anti-windup"
        assert pattern["domain"] == "Control_Locomotion"
        assert pattern["fix"] == "Clamp the integral term"
        assert pattern["anti_pattern"] == "Increase Ki indefinitely"
        assert pattern["keywords"] == ["torque", "saturation", "windup"]
        ki._do_stop()

    def test_no_registry_calls_by_default(self, monkeypatch, tmp_path):
        import rosclaw.know.interface as ki_mod

        calls = []
        monkeypatch.setattr(
            ki_mod,
            "_load_rosclaw_know_patterns",
            lambda: (calls.append(True), [])[1],
        )

        ki = KnowledgeInterface(robot_id="test", assets_path=str(tmp_path))
        ki._do_initialize()

        assert not calls
        ki._do_stop()

    def test_bridge_validation_logs_warnings(self, monkeypatch, tmp_path):
        import rosclaw.know.interface as ki_mod

        report = {"ok": False, "errors": ["missing schema_version"], "warnings": []}
        validator = MagicMock(return_value=report)
        monkeypatch.setattr(ki_mod, "_validate_bridge_index", validator)

        # Pre-create a bridge_index.json so the validator is invoked.
        bridge = tmp_path / "bridge_index.json"
        bridge.write_text('{"schema_version": 2, "symptom_clusters": {}}')

        ki = KnowledgeInterface(robot_id="test", assets_path=str(tmp_path))
        ki._do_initialize()

        validator.assert_called_once()
        ki._do_stop()

    def test_bridge_validation_swallowed_when_package_missing(self, monkeypatch, tmp_path):
        import rosclaw.know.interface as ki_mod

        # Simulate rosclaw_know not installed by making the validator raise ImportError.
        monkeypatch.setattr(
            ki_mod,
            "_validate_bridge_index",
            lambda _data, _cp: (_ for _ in ()).throw(ImportError("no module")),
        )

        ki = KnowledgeInterface(robot_id="test", assets_path=str(tmp_path))
        # Should not raise even though validation errored.
        ki._do_initialize()
        ki._do_stop()

    def test_registry_patterns_are_searchable(self, monkeypatch, tmp_path):
        fake = FakeCuratedPattern(
            pattern_id="gradient_clipping",
            safety_label="Numerical_Instability",
            standard_name="Gradient clipping",
            domain="Learning_Training",
            matched_keywords=["gradient", "explode"],
            fix_pattern="Clip gradients",
        )

        import rosclaw.know.interface as ki_mod

        monkeypatch.setattr(ki_mod, "_load_rosclaw_know_patterns", lambda: [fake])

        ki = KnowledgeInterface(
            robot_id="test",
            assets_path=str(tmp_path),
            use_rosclaw_know_registry=True,
        )
        ki._do_initialize()

        match = ki.match_symptom("gradient explode in loss")
        assert match is not None
        assert match["pattern_id"] == "gradient_clipping"
        ki._do_stop()
