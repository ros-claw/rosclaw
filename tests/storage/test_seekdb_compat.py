"""PR-SDB-140-1: SeekDB compatibility matrix unit tests (no engine needed)."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from rosclaw.storage.seekdb_compat import (
    ALLOW_KNOWN_BAD_ENV,
    classify_sdk_version,
    detect_capabilities,
    log_capabilities,
    validate_sdk_version,
)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (None, "missing"),
        ("1.3.0", "validated"),
        ("1.4.0.post1", "candidate"),
        ("1.4.0", "known_bad"),
        ("1.3.1", "untested"),
        ("1.4.0.dev2", "untested"),
        ("2.0.0", "untested"),
    ],
)
def test_classify(version, expected):
    assert classify_sdk_version(version) == expected


def test_known_bad_raises():
    with pytest.raises(RuntimeError, match="KNOWN-INCOMPATIBLE"):
        validate_sdk_version("1.4.0")


def test_known_bad_override(monkeypatch, caplog):
    monkeypatch.setenv(ALLOW_KNOWN_BAD_ENV, "1")
    with caplog.at_level(logging.ERROR):
        assert validate_sdk_version("1.4.0") == "known_bad"
    assert any("overridden" in r.message for r in caplog.records)


def test_validated_silent(caplog):
    with caplog.at_level(logging.WARNING):
        assert validate_sdk_version("1.3.0") == "validated"
    assert not caplog.records


def test_untested_warns(caplog):
    with caplog.at_level(logging.WARNING):
        assert validate_sdk_version("9.9.9") == "untested"
    assert any("outside the validated version matrix" in r.message for r in caplog.records)


def test_detect_embedded_vs_server():
    embedded = detect_capabilities(path="/tmp/x", host=None)
    assert embedded.deployment == "legacy_embedded"
    assert embedded.native_rrf is False
    assert embedded.multi_instance is False
    assert embedded.explicit_refresh is True

    server = detect_capabilities(path=None, host="127.0.0.1")
    assert server.deployment == "server"
    assert server.native_rrf is True
    assert server.multi_instance is True


def test_detect_records_versions():
    with patch(
        "rosclaw.storage.seekdb_compat.installed_distribution_version",
        side_effect=lambda dist: {"pyseekdb": "1.3.0", "pylibseekdb": "1.3.0.post3"}.get(dist),
    ):
        caps = detect_capabilities(path="/tmp/x", host=None)
    assert caps.sdk_version == "1.3.0"
    assert caps.binding_version == "1.3.0.post3"
    assert caps.to_dict()["deployment"] == "legacy_embedded"


def test_log_capabilities_one_line(caplog):
    caps = detect_capabilities(path=None, host="db.local")
    with caplog.at_level(logging.INFO):
        log_capabilities(caps)
    assert len(caplog.records) == 1
    assert "deployment=server" in caplog.records[0].message
