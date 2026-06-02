"""Tests for practice CLI module."""

import json


from rosclaw.practice.cli import list_episodes, replay_episode, show_episode


class TestListEpisodes:
    def test_empty_directory(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            "rosclaw.practice.cli._artifact_dir", lambda: tmp_path
        )
        list_episodes()
        captured = capsys.readouterr()
        assert "No episodes recorded" in captured.out

    def test_with_episodes(self, tmp_path, monkeypatch, capsys):
        artifacts = tmp_path
        ep_dir = artifacts / "ep_001"
        ep_dir.mkdir(parents=True)
        meta = ep_dir / "metadata.json"
        meta.write_text(
            json.dumps({"robot_id": "ur5e", "status": "success", "reward": 0.95})
        )

        monkeypatch.setattr(
            "rosclaw.practice.cli._artifact_dir", lambda: artifacts
        )
        list_episodes()
        captured = capsys.readouterr()
        assert "ep_001" in captured.out
        assert "ur5e" in captured.out
        assert "success" in captured.out
        assert "0.95" in captured.out

    def test_with_non_ep_dirs(self, tmp_path, monkeypatch, capsys):
        artifacts = tmp_path
        (artifacts / "not_an_episode").mkdir()
        monkeypatch.setattr(
            "rosclaw.practice.cli._artifact_dir", lambda: artifacts
        )
        list_episodes()
        captured = capsys.readouterr()
        assert "No episodes recorded" in captured.out

    def test_missing_metadata(self, tmp_path, monkeypatch, capsys):
        artifacts = tmp_path
        (artifacts / "ep_002").mkdir()
        monkeypatch.setattr(
            "rosclaw.practice.cli._artifact_dir", lambda: artifacts
        )
        list_episodes()
        captured = capsys.readouterr()
        assert "No episodes recorded" in captured.out


class TestShowEpisode:
    def test_episode_found(self, tmp_path, monkeypatch, capsys):
        artifacts = tmp_path
        ep_dir = artifacts / "ep_003"
        ep_dir.mkdir()
        meta = ep_dir / "metadata.json"
        meta.write_text(json.dumps({"robot_id": "g1", "status": "failed"}))

        monkeypatch.setattr(
            "rosclaw.practice.cli._artifact_dir", lambda: artifacts
        )
        show_episode("ep_003")
        captured = capsys.readouterr()
        assert "g1" in captured.out
        assert "failed" in captured.out

    def test_episode_not_found(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            "rosclaw.practice.cli._artifact_dir", lambda: tmp_path
        )
        show_episode("ep_missing")
        captured = capsys.readouterr()
        assert "not found" in captured.out


class TestReplayEpisode:
    def test_full_replay(self, tmp_path, monkeypatch, capsys):
        artifacts = tmp_path
        ep_dir = artifacts / "ep_004"
        ep_dir.mkdir()

        meta = ep_dir / "metadata.json"
        meta.write_text(
            json.dumps(
                {
                    "robot_id": "ur5e",
                    "status": "success",
                    "reward": 0.9,
                    "received_events": ["start", "complete"],
                }
            )
        )

        traj = ep_dir / "trajectory.jsonl"
        traj.write_text(
            json.dumps({"phase": "approach", "skill_name": "reach"}) + "\n"
        )

        trace = ep_dir / "provider_trace.jsonl"
        trace.write_text(json.dumps({"status": "ok"}) + "\n")

        sandbox = ep_dir / "sandbox_replay.json"
        sandbox.write_text(json.dumps({"blocked": False, "block_reason": ""}))

        monkeypatch.setattr(
            "rosclaw.practice.cli._artifact_dir", lambda: artifacts
        )
        replay_episode("ep_004")
        captured = capsys.readouterr()
        assert "Replay: ep_004" in captured.out
        assert "ur5e" in captured.out
        assert "approach" in captured.out
        assert "Provider Traces" in captured.out
        assert "Sandbox Replay" in captured.out

    def test_replay_not_found(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            "rosclaw.practice.cli._artifact_dir", lambda: tmp_path
        )
        replay_episode("ep_missing")
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_replay_metadata_only(self, tmp_path, monkeypatch, capsys):
        artifacts = tmp_path
        ep_dir = artifacts / "ep_005"
        ep_dir.mkdir()

        meta = ep_dir / "metadata.json"
        meta.write_text(json.dumps({"robot_id": "go2", "status": "success"}))

        monkeypatch.setattr(
            "rosclaw.practice.cli._artifact_dir", lambda: artifacts
        )
        replay_episode("ep_005")
        captured = capsys.readouterr()
        assert "go2" in captured.out
        assert "Trajectory" not in captured.out
