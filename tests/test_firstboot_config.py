"""Tests for ROSClaw firstboot configuration utilities."""

from __future__ import annotations

import pytest

from rosclaw.firstboot.config import FirstbootConfig, generate_rosclaw_yaml, load_rosclaw_yaml, merge_config


class TestMergeConfig:
    def test_merge_adds_missing_top_level_keys(self):
        existing = {"a": 1}
        defaults = {"a": 100, "b": 2}
        merged = merge_config(existing, defaults)
        assert merged == {"a": 1, "b": 2}

    def test_merge_preserves_existing_top_level_keys(self):
        existing = {"runtime": {"robot_id": "custom_bot"}}
        defaults = {"runtime": {"robot_id": "sim_ur5e", "log_level": "INFO"}}
        merged = merge_config(existing, defaults)
        assert merged["runtime"]["robot_id"] == "custom_bot"
        assert merged["runtime"]["log_level"] == "INFO"

    def test_merge_recursively_preserves_nested_keys(self):
        existing = {"cloud": {"sync": {"configs": True}}}
        defaults = {"cloud": {"enabled": False, "sync": {"configs": False, "logs": False}}}
        merged = merge_config(existing, defaults)
        assert merged["cloud"]["enabled"] is False
        assert merged["cloud"]["sync"]["configs"] is True
        assert merged["cloud"]["sync"]["logs"] is False

    def test_merge_keeps_existing_non_dict_when_default_is_dict(self):
        existing = {"sandbox": "disabled"}
        defaults = {"sandbox": {"enabled": True}}
        merged = merge_config(existing, defaults)
        assert merged["sandbox"] == "disabled"

    def test_merge_keeps_existing_dict_when_default_is_non_dict(self):
        existing = {"log_level": {"level": "DEBUG"}}
        defaults = {"log_level": "INFO"}
        merged = merge_config(existing, defaults)
        assert merged["log_level"] == {"level": "DEBUG"}

    def test_merge_does_not_mutate_inputs(self):
        existing = {"a": {"b": 1}}
        defaults = {"a": {"c": 2}}
        merged = merge_config(existing, defaults)
        assert existing == {"a": {"b": 1}}
        assert defaults == {"a": {"c": 2}}
        assert merged == {"a": {"b": 1, "c": 2}}


class TestFirstbootConfig:
    def test_defaults_applied_without_overwriting_provided_values(self):
        config = FirstbootConfig(
            workspace={"home": "/tmp/rc", "profile": "cloud"},
            runtime={"robot_id": "turtlebot", "log_level": "DEBUG"},
        )
        assert config.workspace["home"] == "/tmp/rc"
        assert config.workspace["profile"] == "cloud"
        assert config.workspace["mode"] == "local"
        assert config.runtime["robot_id"] == "turtlebot"
        assert config.runtime["log_level"] == "DEBUG"
        assert config.runtime["safety_level"] == "strict"

    def test_apply_profile_offline_disables_cloud_and_telemetry(self):
        config = FirstbootConfig()
        config.apply_profile("offline")
        assert config.cloud["enabled"] is False
        assert config.telemetry["enabled"] is False
        assert config.provider["mode"] == "local"

    def test_apply_profile_cloud_enables_cloud_configs(self):
        config = FirstbootConfig()
        config.apply_profile("cloud")
        assert config.cloud["enabled"] is True
        assert config.cloud["sync"]["configs"] is True
        assert config.provider["mode"] == "hybrid"

    def test_to_dict_removes_none_values(self):
        config = FirstbootConfig(
            provider={"default_llm": None, "default_vlm": "foo"},
        )
        data = config.to_dict()
        assert "default_llm" not in data["provider"]
        assert data["provider"]["default_vlm"] == "foo"


class TestGenerateRosclawYaml:
    def test_generates_yaml_and_preserves_custom_key(self, tmp_path):
        home = tmp_path / ".rosclaw"
        config = FirstbootConfig(
            workspace={"home": str(home), "profile": "offline"},
            runtime={"robot_id": "sim_ur5e"},
        )
        config.apply_profile("offline")

        # Write an existing rosclaw.yaml with a custom key to verify preservation.
        cfg_path = home / "config" / "rosclaw.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text("custom_key: 42\n", encoding="utf-8")

        generate_rosclaw_yaml(home, config)
        loaded = load_rosclaw_yaml(home)
        assert loaded["custom_key"] == 42
        assert loaded["workspace"]["profile"] == "offline"
        assert loaded["telemetry"]["enabled"] is False

    def test_load_rosclaw_yaml_returns_empty_when_missing(self, tmp_path):
        home = tmp_path / ".rosclaw"
        assert load_rosclaw_yaml(home) == {}
