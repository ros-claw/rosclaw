"""replay-patch CLI tests (PR-SAFE-2, v4 §8.5)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosclaw.how.choreography.cli import cmd_how_replay_patch


def _write_session(
    root: Path, practice_id: str, *, interval_s: float = 7.7, rounds: int = 20
) -> None:
    raw = root / "sessions" / practice_id / "raw"
    raw.mkdir(parents=True)
    t0 = 1_700_000_000.0
    with open(raw / "events.jsonl", "w", encoding="utf-8") as handle:
        for i in range(rounds):
            started = t0 + i * interval_s
            event = {
                "event_id": f"evt_{i}",
                "event_type": "rps.stress.round.resolved",
                "timestamp_ns": int(started * 1e9),
                "payload": {
                    "round": {
                        "round_id": f"r_{i}",
                        "started_at": started,
                        "ended_at": started + 4.0,
                        "result": "verified",
                    }
                },
            }
            handle.write(json.dumps(event) + "\n")


def _args(root: Path, patch_file: Path, practice_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        contract="configs/choreography/rh56_rps_v1.yaml",
        patch=str(patch_file),
        practice_id=practice_id,
        current_params=None,
        data_root=str(root),
    )


def test_replay_patch_blocks_run1_death_spiral(tmp_path: Path, capsys) -> None:
    """v4 §13 Replay: run1 death-spiral patches never reach the machine."""
    practice_id = "prac_test_replay"
    _write_session(tmp_path, practice_id)
    patch_file = tmp_path / "run1.json"
    patch_file.write_text(json.dumps({"servo_speed_scale": 0.6, "per_phase_delay_ms": 400}))

    assert cmd_how_replay_patch(_args(tmp_path, patch_file, practice_id)) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["allowed_for_real_experiment"] is False
    assert any("forbidden_parameter" in v for v in output["violations"])
    assert output["rounds_observed"] == 20


def test_replay_patch_allows_safe_cooldown(tmp_path: Path, capsys) -> None:
    practice_id = "prac_test_replay_ok"
    _write_session(tmp_path, practice_id)
    patch_file = tmp_path / "cooldown.json"
    patch_file.write_text(json.dumps({"inter_round_cooldown_sec": 5}))

    assert cmd_how_replay_patch(_args(tmp_path, patch_file, practice_id)) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["allowed_for_real_experiment"] is True
    assert (
        output["patched_timeline_ms"]["cooldown"] - output["original_timeline_ms"]["cooldown"]
        == 5000.0
    )
    assert output["expected_reveal_offset_ms"] == 3300.0


def test_replay_patch_blocks_budget_breaker(tmp_path: Path, capsys) -> None:
    practice_id = "prac_test_replay_budget"
    _write_session(tmp_path, practice_id)
    patch_file = tmp_path / "budget.json"
    patch_file.write_text(json.dumps({"inter_round_cooldown_sec": 25}))

    assert cmd_how_replay_patch(_args(tmp_path, patch_file, practice_id)) == 1
    output = json.loads(capsys.readouterr().out)
    assert any("round_budget" in v for v in output["violations"])
