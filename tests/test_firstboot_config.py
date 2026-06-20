"""Tests for ROSClaw firstboot configuration utilities."""

from __future__ import annotations


class TestMergeConfig:
    def test_merge_config_preserves_existing_keys(self):
        from rosclaw.firstboot.config import merge_config

        existing = {"cloud": {"enabled": True}, "runtime": {"robot_id": "custom"}}
        defaults = {"cloud": {"enabled": False, "endpoint": "https://example.com"}, "runtime": {"robot_id": "sim_ur5e"}}
        merged = merge_config(existing, defaults)

        assert merged["cloud"]["enabled"] is True
        assert merged["cloud"]["endpoint"] == "https://example.com"
        assert merged["runtime"]["robot_id"] == "custom"

    def test_merge_config_overlays_missing_defaults(self):
        from rosclaw.firstboot.config import merge_config

        existing = {"runtime": {"log_level": "DEBUG"}}
        defaults = {"runtime": {"robot_id": "sim_ur5e", "log_level": "INFO"}, "sandbox": {"enabled": True}}
        merged = merge_config(existing, defaults)

        assert merged["runtime"]["robot_id"] == "sim_ur5e"
        assert merged["runtime"]["log_level"] == "DEBUG"
        assert merged["sandbox"]["enabled"] is True

    def test_merge_config_nested_merge(self):
        from rosclaw.firstboot.config import merge_config

        existing = {"cloud": {"sync": {"configs": True}}, "security": {"secrets_backend": "keyring"}}
        defaults = {
            "cloud": {"enabled": False, "sync": {"configs": False, "logs": False}},
            "security": {"never_execute_robot_without_confirmation": True},
        }
        merged = merge_config(existing, defaults)

        assert merged["cloud"]["enabled"] is False
        assert merged["cloud"]["sync"]["configs"] is True
        assert merged["cloud"]["sync"]["logs"] is False
        assert merged["security"]["secrets_backend"] == "keyring"
        assert merged["security"]["never_execute_robot_without_confirmation"] is True

    def test_merge_config_leaves_lists_untouched(self):
        from rosclaw.firstboot.config import merge_config

        existing = {"darwin": {"seeds": [10, 20]}}
        defaults = {"darwin": {"seeds": [0, 1, 2], "episodes": 50}}
        merged = merge_config(existing, defaults)

        assert merged["darwin"]["seeds"] == [10, 20]
        assert merged["darwin"]["episodes"] == 50

    def test_merge_config_empty_existing(self):
        from rosclaw.firstboot.config import merge_config

        defaults = {"runtime": {"robot_id": "sim_ur5e"}, "cloud": {"enabled": False}}
        merged = merge_config({}, defaults)

        assert merged == defaults
