"""Tests for ROSClaw dataset vocab encoding."""

from __future__ import annotations

from rosclaw.integrations.lerobot.dataset_vocab import (
    UNKNOWN_CODE,
    UNKNOWN_LABEL,
    build_rosclaw_vocab,
    decode,
    encode,
    write_vocab,
    read_vocab,
)


def test_encode_known_label() -> None:
    vocab = {"UNKNOWN": 0, "ALLOW": 1, "BLOCK": 2}
    assert encode("ALLOW", vocab) == 1
    assert encode("BLOCK", vocab) == 2


def test_encode_unknown_label() -> None:
    vocab = {"UNKNOWN": 0, "ALLOW": 1}
    assert encode("NOT_IN_VOCAB", vocab) == UNKNOWN_CODE


def test_encode_missing_label() -> None:
    vocab = {"UNKNOWN": 0, "ALLOW": 1}
    assert encode(None, vocab) == UNKNOWN_CODE


def test_encode_numeric_string_already_code() -> None:
    vocab = {"UNKNOWN": 0, "ALLOW": 1}
    assert encode("1", vocab) == 1


def test_decode_known_code() -> None:
    vocab = {"UNKNOWN": 0, "ALLOW": 1}
    assert decode(1, vocab) == "ALLOW"


def test_decode_unknown_code() -> None:
    vocab = {"UNKNOWN": 0, "ALLOW": 1}
    assert decode(99, vocab) == UNKNOWN_LABEL


def test_build_rosclaw_vocab_groups() -> None:
    vocab = build_rosclaw_vocab(["safety", "failure", "intervention", "action"])
    assert "rosclaw.sandbox.decision" in vocab.vocabularies
    assert "rosclaw.failure.code" in vocab.vocabularies
    assert "rosclaw.intervention.source" in vocab.vocabularies
    assert "rosclaw.action.source" in vocab.vocabularies


def test_write_and_read_vocab(tmp_path) -> None:
    vocab = build_rosclaw_vocab(["safety"])
    write_vocab(vocab, tmp_path)
    loaded = read_vocab(tmp_path)
    assert loaded is not None
    assert "rosclaw.sandbox.decision" in loaded.vocabularies
