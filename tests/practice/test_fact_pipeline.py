"""PR-DF-04: Practice fact pipeline — canonical names + session-close verify."""

import tempfile
import time
from pathlib import Path

from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator


def test_canonical_names_and_aliases():
    from rosclaw.practice.distiller import (
        DistillationResult,
        EpisodeFactBundle,
        EpisodeFactExtractor,
        PracticeDistiller,
    )
    from rosclaw.practice.seekdb_ingestor import PracticeFactIngestor, SeekDBIngestor

    assert PracticeDistiller is EpisodeFactExtractor
    assert DistillationResult is EpisodeFactBundle
    assert SeekDBIngestor is PracticeFactIngestor
    assert hasattr(EpisodeFactExtractor, "extract")


def test_extract_returns_fact_bundle():
    """The canonical verb produces the fact bundle from a closed session."""
    from rosclaw.practice.distiller import EpisodeFactBundle, EpisodeFactExtractor

    with tempfile.TemporaryDirectory() as tmp:
        cfg = PracticeConfig(
            robot_id="test_bot",
            task_name="pick cup",
            data_root=tmp,
            sources=SourceConfig(agent=True),
            mock=True,
            publish_to_event_bus=False,
        )
        coord = PracticeCoordinator(cfg)
        coord.initialize()
        coord.start()
        time.sleep(0.3)
        coord.stop()
        practice_id = coord.summary.practice_id

        bundle = EpisodeFactExtractor(tmp).extract(practice_id, write_artifacts=False)
        assert isinstance(bundle, EpisodeFactBundle)
        assert bundle.practice_id == practice_id
        assert bundle.session_id


def test_session_close_runs_fact_verify():
    """PR-DF-04 §19: session close = verify → extract → ingest; the verify
    outcome is recorded on the summary (observable) and never blocks close."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PracticeConfig(
            robot_id="test_bot",
            task_name="pick cup",
            data_root=tmp,
            sources=SourceConfig(agent=True, runtime=True),
            mock=True,
            publish_to_event_bus=False,
        )
        coord = PracticeCoordinator(cfg)
        coord.initialize()
        coord.start()
        time.sleep(0.3)
        coord.stop()

        summary = coord.summary
        assert summary is not None
        assert summary.fact_verify is not None
        assert summary.fact_verify["passed"] is True
        assert summary.fact_verify["errors"] == 0
        # the manifest was written after the pipeline and carries the record
        manifest = (Path(tmp) / "sessions" / summary.practice_id / "manifest.yaml").read_text()
        assert "fact_verify" in manifest


def test_fact_verify_failure_does_not_block_close():
    """A verifier exception degrades to None — the session still closes."""
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp:
        cfg = PracticeConfig(
            robot_id="test_bot",
            task_name="pick cup",
            data_root=tmp,
            sources=SourceConfig(agent=True),
            mock=True,
            publish_to_event_bus=False,
        )
        coord = PracticeCoordinator(cfg)
        coord.initialize()
        coord.start()
        time.sleep(0.2)
        with patch(
            "rosclaw.practice.verifier.PracticeVerifier.verify",
            side_effect=RuntimeError("verifier exploded"),
        ):
            coord.stop()
        summary = coord.summary
        assert summary is not None
        assert summary.fact_verify is None
        assert summary.outcome == "SUCCESS"
