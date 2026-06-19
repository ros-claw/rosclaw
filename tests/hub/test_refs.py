"""Tests for ROSClaw Hub asset references."""

from __future__ import annotations

import pytest

from rosclaw.hub.errors import HubError, HubErrorCode
from rosclaw.hub.refs import AssetRef, normalize_name, parse_ref, parse_ref_with_version


class TestParseRef:
    """Tests for parse_ref."""

    def test_parse_full_ref(self) -> None:
        ref = parse_ref("rosclaw://hardware_mcp/rosclaw/unitree-g1@1.0.0")
        assert ref == AssetRef(
            type="hardware_mcp",
            namespace="rosclaw",
            name="unitree-g1",
            version="1.0.0",
        )

    def test_parse_ref_without_version(self) -> None:
        ref = parse_ref("rosclaw://skill/rosclaw/g1-pick-place")
        assert ref.type == "skill"
        assert ref.namespace == "rosclaw"
        assert ref.name == "g1-pick-place"
        assert ref.version is None

    def test_parse_all_asset_types(self) -> None:
        for asset_type in ("skill", "provider", "hardware_mcp", "digital_twin", "cognitive_wiki"):
            ref = parse_ref(f"rosclaw://{asset_type}/rosclaw/demo-asset@0.1.0")
            assert ref.type == asset_type

    def test_parse_rejects_invalid_scheme(self) -> None:
        with pytest.raises(HubError) as exc_info:
            parse_ref("http://example.com/asset")
        assert exc_info.value.code == HubErrorCode.MANIFEST_INVALID

    def test_parse_rejects_invalid_type(self) -> None:
        with pytest.raises(HubError) as exc_info:
            parse_ref("rosclaw://unknown/rosclaw/demo@1.0.0")
        assert exc_info.value.code == HubErrorCode.MANIFEST_INVALID

    def test_parse_rejects_bad_namespace(self) -> None:
        with pytest.raises(HubError):
            parse_ref("rosclaw://skill/ROSClaw/demo@1.0.0")

    def test_parse_rejects_non_string(self) -> None:
        with pytest.raises(HubError) as exc_info:
            parse_ref(None)  # type: ignore[arg-type]
        assert exc_info.value.code == HubErrorCode.MANIFEST_INVALID


class TestAssetRef:
    """Tests for AssetRef helpers."""

    def test_str_with_version(self) -> None:
        ref = AssetRef(
            type="digital_twin",
            namespace="rosclaw",
            name="g1-mujoco-basic",
            version="0.5.0",
        )
        assert str(ref) == "rosclaw://digital_twin/rosclaw/g1-mujoco-basic@0.5.0"

    def test_str_without_version(self) -> None:
        ref = AssetRef(
            type="cognitive_wiki",
            namespace="rosclaw",
            name="humanoid-locomotion-patterns",
            version=None,
        )
        assert str(ref) == "rosclaw://cognitive_wiki/rosclaw/humanoid-locomotion-patterns"

    def test_identity_tuple(self) -> None:
        ref = parse_ref("rosclaw://provider/rosclaw/qwen3-vl-gr00t@0.3.1")
        assert ref.identity_tuple() == ("provider", "rosclaw", "qwen3-vl-gr00t", "0.3.1")

    def test_frozen_dataclass_is_hashable(self) -> None:
        ref = parse_ref("rosclaw://skill/rosclaw/g1-pick-place@1.2.0")
        assert hash(ref) == hash(ref.identity_tuple())


class TestParseRefWithVersion:
    """Tests for parse_ref_with_version."""

    def test_overrides_version(self) -> None:
        ref = parse_ref_with_version(
            "rosclaw://skill/rosclaw/g1-pick-place", version="2.0.0"
        )
        assert ref.version == "2.0.0"

    def test_preserves_existing_version_when_no_override(self) -> None:
        ref = parse_ref_with_version("rosclaw://skill/rosclaw/g1-pick-place@1.0.0")
        assert ref.version == "1.0.0"


class TestNormalizeName:
    """Tests for normalize_name."""

    def test_normalizes_underscores_and_dots(self) -> None:
        assert normalize_name("G1_Pick.Place") == "g1-pick-place"

    def test_already_normalized(self) -> None:
        assert normalize_name("g1-pick-place") == "g1-pick-place"
