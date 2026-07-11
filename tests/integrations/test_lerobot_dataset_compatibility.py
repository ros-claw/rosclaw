"""Tests for `rosclaw lerobot dataset-compatibility`."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

from rosclaw.cli import main


def test_dataset_compatibility_text(capsys) -> None:
    with patch.object(sys, "argv", ["rosclaw", "lerobot", "dataset-compatibility"]):
        rc = main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "Safety / sandbox metadata" in out
    assert "Physical telemetry" in out


def test_dataset_compatibility_json(capsys) -> None:
    with patch.object(sys, "argv", ["rosclaw", "lerobot", "dataset-compatibility", "--json"]):
        rc = main()
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    features = {row["feature"] for row in data["matrix"]}
    assert "Safety / sandbox metadata" in features
