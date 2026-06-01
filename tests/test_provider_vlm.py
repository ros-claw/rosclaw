"""Tests for MockVLMProvider."""

from pathlib import Path
from unittest.mock import patch

import pytest

try:
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from rosclaw.provider.builtins.vlm import MockVLMProvider, _ArtifactRef, _resolve_artifact
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.request import ProviderRequest


def _make_manifest(**kwargs):
    return ProviderManifest(
        name=kwargs.get("name", "vlm"),
        version="1.0.0",
        type="vlm",
        capabilities=kwargs.get("capabilities", ["vlm.object_grounding", "vlm.scene_understanding"]),
    )


class TestArtifactRef:
    def test_from_dict(self):
        ref = _ArtifactRef.from_dict({"type": "image", "uri": "artifact://ep1/img.png"})
        assert ref.type == "image"
        assert ref.uri == "artifact://ep1/img.png"

    def test_from_dict_defaults(self):
        ref = _ArtifactRef.from_dict({})
        assert ref.type == ""
        assert ref.uri == ""


class TestResolveArtifact:
    def test_artifact_uri(self):
        path = _resolve_artifact("artifact://episode1/image.png")
        assert "episode1" in str(path)
        assert "image.png" in str(path)

    def test_plain_path(self):
        path = _resolve_artifact("/tmp/image.png")
        assert str(path) == "/tmp/image.png"


class TestExtractColor:
    def test_red_cup(self):
        assert MockVLMProvider._extract_color("red cup") == "red"

    def test_blue_block(self):
        assert MockVLMProvider._extract_color("blue block") == "blue"

    def test_no_color(self):
        assert MockVLMProvider._extract_color("random object") is None

    def test_case_insensitive(self):
        assert MockVLMProvider._extract_color("RED CUP") == "red"


class TestResolveImagePath:
    @pytest.fixture
    def provider(self):
        return MockVLMProvider(_make_manifest())

    def test_string_path(self, provider):
        path = provider._resolve_image_path("/tmp/test.png")
        assert str(path) == "/tmp/test.png"

    def test_artifact_string(self, provider):
        path = provider._resolve_image_path("artifact://ep1/img.png")
        assert "ep1" in str(path)

    def test_file_uri(self, provider):
        path = provider._resolve_image_path("file:///tmp/test.png")
        assert str(path) == "/tmp/test.png"

    def test_dict_artifact(self, provider):
        path = provider._resolve_image_path({"type": "artifact", "uri": "artifact://ep1/img.png"})
        assert "ep1" in str(path)

    def test_dict_file(self, provider):
        path = provider._resolve_image_path({"type": "file", "value": "/tmp/test.png"})
        assert str(path) == "/tmp/test.png"

    def test_none(self, provider):
        assert provider._resolve_image_path(None) is None

    def test_unknown_dict(self, provider):
        assert provider._resolve_image_path({"type": "unknown"}) is None


class TestDetectByColor:
    @pytest.fixture
    def provider(self):
        return MockVLMProvider(_make_manifest())

    def test_no_image_fallback(self, provider):
        result = provider._detect_by_color(None, "red cup")
        assert result["method"] == "synthetic_fallback"
        assert result["confidence"] == 0.85
        assert "red" in result["id"]

    def test_no_color_in_query(self, provider):
        result = provider._detect_by_color(Path("/nonexistent.png"), "random thing")
        assert result["method"] == "synthetic_fallback"

    def test_with_pil_real_image(self, provider, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        # Create a red image
        img_path = tmp_path / "red.png"
        img = Image.new("RGB", (100, 100), color=(200, 30, 30))
        img.save(img_path)

        result = provider._detect_by_color(img_path, "red cup")
        assert result["method"] == "pil_color_detection"
        assert result["confidence"] > 0.5
        assert "red" in result["id"]
        assert "pixel_count" in result

    def test_with_pil_no_matching_color(self, provider, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        # Create a blue image, query for red
        img_path = tmp_path / "blue.png"
        img = Image.new("RGB", (100, 100), color=(30, 30, 200))
        img.save(img_path)

        result = provider._detect_by_color(img_path, "red cup")
        assert result["method"] == "color_not_found"
        assert "red" in result["id"]

    def test_pil_import_error(self, provider, tmp_path):
        # When PIL is unavailable AND image doesn't exist → synthetic_fallback
        with patch.dict("sys.modules", {"PIL": None}):
            result = provider._detect_by_color(tmp_path / "fake.png", "red cup")
            # Path doesn't exist so falls back before trying PIL
            assert result["method"] in ("synthetic_fallback", "synthetic_no_pil")

    @pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
    def test_pil_exception(self, provider, tmp_path):
        # Create an actual file so PIL is attempted, then mock it
        img_path = tmp_path / "fake.png"
        img_path.write_text("not an image")
        with patch("PIL.Image.open", side_effect=IOError("corrupt")):
            result = provider._detect_by_color(img_path, "red cup")
            assert result["method"] == "error"
            assert "corrupt" in result.get("error", "")


class TestInferObjectGrounding:
    @pytest.mark.asyncio
    async def test_grounding_no_image(self):
        provider = MockVLMProvider(_make_manifest())
        req = ProviderRequest(
            request_id="r1",
            capability="vlm.object_grounding",
            inputs={"query": "red cup", "image": None},
        )
        resp = await provider.infer(req)
        assert resp.status == "ok"
        assert len(resp.result["objects"]) == 1
        assert resp.result["objects"][0]["method"] == "synthetic_fallback"

    @pytest.mark.asyncio
    async def test_grounding_with_image(self, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        provider = MockVLMProvider(_make_manifest())
        img_path = tmp_path / "red.png"
        Image.new("RGB", (100, 100), color=(200, 30, 30)).save(img_path)

        req = ProviderRequest(
            request_id="r1",
            capability="vlm.object_grounding",
            inputs={"query": "red cup", "image": str(img_path)},
        )
        resp = await provider.infer(req)
        assert resp.status == "ok"
        assert len(resp.result["objects"]) == 1
        assert resp.result["objects"][0]["method"] == "pil_color_detection"


class TestInferSceneUnderstanding:
    @pytest.mark.asyncio
    async def test_scene_understanding_no_image(self):
        provider = MockVLMProvider(_make_manifest())
        req = ProviderRequest(
            request_id="r1",
            capability="vlm.scene_understanding",
            inputs={},
        )
        resp = await provider.infer(req)
        assert resp.status == "ok"
        assert resp.result["scene"] == "tabletop"
        assert resp.confidence == 0.8

    @pytest.mark.asyncio
    async def test_scene_understanding_with_image(self, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        provider = MockVLMProvider(_make_manifest())
        img_path = tmp_path / "bright.png"
        Image.new("RGB", (100, 100), color=(250, 250, 250)).save(img_path)

        req = ProviderRequest(
            request_id="r1",
            capability="vlm.scene_understanding",
            inputs={"image": str(img_path)},
        )
        resp = await provider.infer(req)
        assert resp.status == "ok"
        assert resp.result["scene"] == "bright_indoor"


class TestAnalyzeScene:
    def test_bright_scene(self, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        img_path = tmp_path / "bright.png"
        Image.new("RGB", (100, 100), color=(250, 250, 250)).save(img_path)
        assert MockVLMProvider._analyze_scene(img_path) == "bright_indoor"

    def test_dark_scene(self, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        img_path = tmp_path / "dark.png"
        Image.new("RGB", (100, 100), color=(10, 10, 10)).save(img_path)
        assert MockVLMProvider._analyze_scene(img_path) == "dark_scene"

    def test_mid_scene(self, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        img_path = tmp_path / "mid.png"
        Image.new("RGB", (100, 100), color=(150, 150, 150)).save(img_path)
        assert MockVLMProvider._analyze_scene(img_path) == "tabletop"

    def test_analyze_exception(self):
        assert MockVLMProvider._analyze_scene(Path("/nonexistent")) == "tabletop"


class TestVLMHealth:
    @pytest.mark.asyncio
    async def test_health(self):
        provider = MockVLMProvider(_make_manifest())
        health = await provider.health()
        assert health["ok"] is True
        assert health["backend"] == "pil_color_detection"


class TestUnsupportedCapability:
    @pytest.mark.asyncio
    async def test_unsupported_raises(self):
        provider = MockVLMProvider(_make_manifest(capabilities=["vlm.object_grounding"]))
        req = ProviderRequest(
            request_id="r1",
            capability="vlm.unknown",
            inputs={},
        )
        with pytest.raises(Exception):
            await provider.infer(req)
