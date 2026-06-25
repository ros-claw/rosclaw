"""Tests for FeedbackUploader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from rosclaw.feedback.installation import InstallationManager
from rosclaw.feedback.upload import FeedbackUploader


class TestFeedbackUploader:
    def test_upload_requires_redact(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        result = FeedbackUploader(home).upload(redact=False)
        assert result["ok"] is False
        assert result["error"] == "redaction_required"

    def test_dry_run_does_not_upload(self, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        result = FeedbackUploader(home).upload(redact=True, dry_run=True)
        assert result["ok"] is True
        assert result["dry_run"] is True
        assert "bundle_path" in result

    @patch("requests.post")
    def test_upload_posts_bundle(self, mock_post, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()

        mock_response = MagicMock()
        mock_response.text = '{"ok": true, "request_id": "fb_123"}'
        mock_response.json.return_value = {"ok": True, "request_id": "fb_123"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = FeedbackUploader(home).upload(redact=True, dry_run=False)
        assert result["ok"] is True
        assert result.get("request_id") == "fb_123"
        mock_post.assert_called_once()

        call_args = mock_post.call_args
        assert "files" in call_args.kwargs
        assert "bundle" in call_args.kwargs["files"]
        assert call_args.kwargs["data"]["anonymous_installation_id"].startswith("rci_")
        assert call_args.kwargs["data"]["redacted"] is True
