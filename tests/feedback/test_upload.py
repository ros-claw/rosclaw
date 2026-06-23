"""Tests for FeedbackUploader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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

    @patch("urllib.request.urlopen")
    def test_upload_posts_bundle(self, mock_urlopen, tmp_path: Path) -> None:
        home = tmp_path / ".rosclaw"
        InstallationManager(home).ensure_installation()
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b'{"ok": true, "request_id": "fb_123"}'

        result = FeedbackUploader(home).upload(redact=True, dry_run=False)
        assert result["ok"] is True
        mock_urlopen.assert_called_once()
