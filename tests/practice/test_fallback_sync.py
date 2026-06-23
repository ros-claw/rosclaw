"""Tests for FallbackSync."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from rosclaw.practice.storage.fallback_sync import FallbackSync


def test_sync_success_moves_to_archive():
    with tempfile.TemporaryDirectory() as tmp:
        fallback = Path(tmp)
        payload = {"practice_id": "prac_001"}
        (fallback / "prac_001.json").write_text(json.dumps(payload), encoding="utf-8")

        with patch("rosclaw.practice.storage.fallback_sync.requests.post") as mock_post:
            mock_post.return_value.raise_for_status = lambda: None
            sync = FallbackSync(seekdb_url="http://localhost:2881", fallback_dir=fallback)
            summary = sync.sync()

        assert summary["attempted"] == 1
        assert summary["success"] == 1
        assert summary["failed"] == 0
        assert not (fallback / "prac_001.json").exists()
        assert (fallback / "archived" / "prac_001.json").exists()


def test_sync_failure_keeps_file():
    with tempfile.TemporaryDirectory() as tmp:
        fallback = Path(tmp)
        (fallback / "prac_002.json").write_text(json.dumps({"practice_id": "prac_002"}), encoding="utf-8")

        with patch("rosclaw.practice.storage.fallback_sync.requests.post") as mock_post:
            mock_post.return_value.raise_for_status.side_effect = Exception("boom")
            sync = FallbackSync(seekdb_url="http://localhost:2881", fallback_dir=fallback)
            summary = sync.sync()

        assert summary["attempted"] == 1
        assert summary["success"] == 0
        assert summary["failed"] == 1
        assert (fallback / "prac_002.json").exists()


def test_list_pending():
    with tempfile.TemporaryDirectory() as tmp:
        fallback = Path(tmp)
        (fallback / "a.json").write_text("{}", encoding="utf-8")
        sync = FallbackSync(fallback_dir=fallback)
        assert len(sync.list_pending()) == 1
