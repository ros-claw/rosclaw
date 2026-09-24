from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw.dream.contracts import DreamBudget
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.research import (
    ExperimentFamily,
    ResearchBudget,
    ResearchCampaign,
    ResearchHypothesis,
)
from rosclaw.growth.sealed_use import SealedUseLedger


def h(value):
    return canonical_hash({"value": value})


def campaign():
    return ResearchCampaign(
        "fixture.skill",
        ResearchHypothesis("hypothesis", "claim", "prediction", h("conditions"), h("evaluation")),
        ExperimentFamily("family", h("mechanism")),
        ResearchBudget(DreamBudget(0.0, 64, 16, 3600.0, 0.0, 0.0), 3),
        h("train"),
        h("dev"),
        h("retention"),
        h("sealed"),
    )


def consume(ledger, value=None, **changes):
    return ledger.consume(
        value or campaign(),
        **{
            "candidate_hash": h("candidate"),
            "evaluation_hash": h("evaluation"),
            "development_receipt_hash": h("dev-receipt"),
            "retention_receipt_hash": h("retention-receipt"),
            **changes,
        },
    )


@pytest.mark.parametrize(
    "field",
    [
        "train_snapshot_hash",
        "development_snapshot_hash",
        "retention_snapshot_hash",
        "sealed_commitment",
    ],
)
def test_unbound_bank_rejected_before_consumption(tmp_path, field):
    with SealedUseLedger(tmp_path / "state", source_checkout=tmp_path / "source") as ledger:
        with pytest.raises(ValueError, match="bound"):
            consume(ledger, replace(campaign(), **{field: None}))
        assert consume(ledger)  # rejection did not consume a different or null identity


def test_crash_recovery_cannot_reopen_same_bank_for_another_candidate(tmp_path):
    root = tmp_path / "state"
    with SealedUseLedger(root, source_checkout=tmp_path / "source") as ledger:
        assert consume(ledger).startswith("sha256:")
    with SealedUseLedger(root, source_checkout=tmp_path / "source") as recovered:
        with pytest.raises(ValueError, match="consumed"):
            consume(recovered, candidate_hash=h("different-candidate"))
        with pytest.raises(ValueError, match="consumed"):
            consume(recovered, replace(campaign(), campaign_id="renamed"))


def test_changed_evaluator_and_fake_same_stage_receipt_rejected_before_consume(tmp_path):
    with SealedUseLedger(tmp_path / "state", source_checkout=tmp_path / "source") as ledger:
        with pytest.raises(ValueError, match="evaluation"):
            consume(ledger, evaluation_hash=h("changed"))
        with pytest.raises(ValueError, match="distinct"):
            consume(ledger, retention_receipt_hash=h("dev-receipt"))
        assert consume(ledger)


def test_second_writer_rejected_and_close_releases_lock(tmp_path):
    root = tmp_path / "state"
    first = SealedUseLedger(root, source_checkout=tmp_path / "source")
    with pytest.raises(BlockingIOError):
        SealedUseLedger(root, source_checkout=tmp_path / "source")
    first.close()
    with pytest.raises(RuntimeError):
        consume(first)
    with SealedUseLedger(root, source_checkout=tmp_path / "source") as second:
        assert consume(second)


def test_ambiguous_write_latches_and_recovery_keeps_committed_consumption(tmp_path, monkeypatch):
    root = tmp_path / "state"
    with SealedUseLedger(root, source_checkout=tmp_path / "source") as ledger:
        append = ledger._log.append

        def fail_after_commit(*args):
            append(*args)
            raise OSError("simulated error after durable commit")

        monkeypatch.setattr(ledger._log, "append", fail_after_commit)
        with pytest.raises(OSError):
            consume(ledger)
        with pytest.raises(RuntimeError, match="failed"):
            consume(ledger)
    with (
        SealedUseLedger(root, source_checkout=tmp_path / "source") as recovered,
        pytest.raises(ValueError, match="consumed"),
    ):
        consume(recovered)


def test_mutable_data_inside_checkout_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        SealedUseLedger(tmp_path / "source/state", source_checkout=tmp_path / "source")
    with pytest.raises(ValueError):
        SealedUseLedger(Path("/"), source_checkout=tmp_path / "source")


def test_tampered_journal_rejected(tmp_path):
    root = tmp_path / "state"
    with SealedUseLedger(root, source_checkout=tmp_path / "source") as ledger:
        consume(ledger)
    event = next((root / "events").glob("*.json"))
    event.write_text(event.read_text().replace(h("candidate"), h("tampered")))
    with pytest.raises(ValueError, match="hash"):
        SealedUseLedger(root, source_checkout=tmp_path / "source")
