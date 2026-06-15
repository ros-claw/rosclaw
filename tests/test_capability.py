"""Tests for provider capability taxonomy."""

import pytest

from rosclaw.provider.core.capability import (
    CAPABILITY_CATALOG,
    Capability,
    CapabilityDomain,
    is_valid_capability,
    list_capabilities,
)


class TestCapability:
    def test_str(self):
        cap = Capability(domain="vlm", name="object_grounding")
        assert str(cap) == "vlm.object_grounding"

    def test_parse_valid(self):
        cap = Capability.parse("skill.pick_and_place")
        assert cap.domain == "skill"
        assert cap.name == "pick_and_place"

    def test_parse_invalid_no_dot(self):
        with pytest.raises(ValueError, match="Invalid capability string"):
            Capability.parse("invalid")

    def test_parse_multiple_dots(self):
        cap = Capability.parse("a.b.c")
        assert cap.domain == "a"
        assert cap.name == "b.c"


class TestCapabilityDomain:
    def test_constants(self):
        assert CapabilityDomain.LLM == "llm"
        assert CapabilityDomain.VLM == "vlm"
        assert CapabilityDomain.VLA == "vla"
        assert CapabilityDomain.VLN == "vln"
        assert CapabilityDomain.WORLD == "world"
        assert CapabilityDomain.SKILL == "skill"
        assert CapabilityDomain.CRITIC == "critic"
        assert CapabilityDomain.EMBEDDING == "embedding"


class TestCatalog:
    def test_catalog_non_empty(self):
        assert len(CAPABILITY_CATALOG) >= 8

    def test_is_valid_capability_true(self):
        assert is_valid_capability("vlm.object_grounding") is True
        assert is_valid_capability("skill.pick_and_place") is True

    def test_is_valid_capability_false(self):
        assert is_valid_capability("vlm.nonexistent") is False
        assert is_valid_capability("invalid.no_dot") is False

    def test_is_valid_capability_bad_format(self):
        assert is_valid_capability("nodot") is False

    def test_list_capabilities_all(self):
        caps = list_capabilities()
        assert len(caps) > 0
        assert "vlm.object_grounding" in caps
        assert "skill.pick_and_place" in caps

    def test_list_capabilities_by_domain(self):
        vlm_caps = list_capabilities(domain="vlm")
        assert all(c.startswith("vlm.") for c in vlm_caps)
        assert "vlm.object_grounding" in vlm_caps

    def test_list_capabilities_empty_domain(self):
        assert list_capabilities(domain="nonexistent") == []
