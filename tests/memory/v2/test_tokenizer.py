"""Tests for the bilingual memory tokenizer (§6.4)."""

from __future__ import annotations

from rosclaw.memory.v2.tokenizer import token_set, tokenize


def test_chinese_segmentation() -> None:
    tokens = tokenize("手指过流")
    assert "手指" in tokens
    assert "过流" in tokens


def test_english_words() -> None:
    tokens = tokenize("finger overcurrent")
    assert "finger" in tokens
    assert "overcurrent" in tokens


def test_error_codes_preserved() -> None:
    tokens = tokenize("串口报错 USB_TIMEOUT, EIO -110")
    assert "USB_TIMEOUT" in tokens
    assert "EIO" in tokens
    assert "-110" in tokens


def test_device_names_preserved() -> None:
    tokens = tokenize("RH56 D435i Jetson CH340 FTDI")
    for expected in ("RH56", "D435i", "Jetson", "CH340", "FTDI"):
        assert expected in tokens


def test_code_symbols_preserved_with_parts() -> None:
    tokens = tokenize("rosclaw.practice.recorder 和 stress_closed_loop")
    assert "rosclaw.practice.recorder" in tokens
    assert "recorder" in tokens
    assert "stress_closed_loop" in tokens
    assert "stress" in tokens


def test_cross_lingual_failure_recall() -> None:
    """§6.4: these four phrasings must share retrieval-relevant tokens."""
    zh = token_set("手指过流")
    en = token_set("finger overcurrent")
    zh2 = token_set("电机电流异常")
    en2 = token_set("RH56 finger current spike")
    # With the cross-lingual bridge, zh and en phrasings of the same failure
    # family share expanded tokens.
    assert zh & en, "手指过流 and finger overcurrent must share expanded tokens"
    assert zh2 & en2, "电机电流异常 and RH56 finger current spike must share tokens"
    assert "finger" in en and "手指" in en
    assert "过流" in en and "overcurrent" in zh
    assert "RH56" in en2
    assert "过流" in zh
    assert "电流" in zh2


def test_cross_lingual_gesture_bridge() -> None:
    zh = token_set("剪刀 失败")
    assert "scissors" in zh
    assert "failed" in zh or "failure" in zh
    en = token_set("left_scissors failed")
    assert "剪刀" in en
    assert "失败" in en


def test_empty_and_noise() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []
