"""Tests for AI Collaboration / DeepSeek integration."""

import pytest

from rosclaw.agent_runtime.ai_collaboration import DeepSeekConfig, DeepSeekClient


class TestDeepSeekConfig:
    def test_default_config(self):
        cfg = DeepSeekConfig()
        assert cfg.base_url == "https://api.deepseek.com"
        assert cfg.model == "deepseek-v4-pro"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096
        assert cfg.api_key == ""

    def test_custom_config(self):
        cfg = DeepSeekConfig(api_key="sk-test", base_url="http://localhost", model="m", temperature=0.5, max_tokens=100)
        assert cfg.api_key == "sk-test"
        assert cfg.base_url == "http://localhost"
        assert cfg.model == "m"
        assert cfg.temperature == 0.5
        assert cfg.max_tokens == 100


class TestDeepSeekClient:
    def test_init_defaults(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        client = DeepSeekClient()
        assert client.config.api_key == ""
        assert client.config.base_url == "https://api.deepseek.com"
        assert client._client is None

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://env.local")
        client = DeepSeekClient()
        assert client.config.api_key == "sk-env"
        assert client.config.base_url == "http://env.local"

    def test_init_with_config(self):
        cfg = DeepSeekConfig(api_key="sk-cfg")
        client = DeepSeekClient(config=cfg)
        assert client.config.api_key == "sk-cfg"

    def test_get_client_import_error(self, monkeypatch):
        """Simulate openai not installed by blocking the import."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        client = DeepSeekClient(DeepSeekConfig(api_key="k"))
        with pytest.raises(RuntimeError, match="openai package required"):
            client._get_client()

    def test_plan_task_error_path(self):
        """plan_task returns error dict when client fails."""
        client = DeepSeekClient(DeepSeekConfig(api_key=""))
        result = client.plan_task("pick cup", {"robot": "ur5e"})
        assert "error" in result
        assert result.get("task_name") == "failed"
        assert "steps" in result

    def test_analyze_failure_with_heuristic_hit(self):
        """analyze_failure uses heuristic_engine when provided and match found."""
        class FakeHeuristic:
            async def suggest_recovery(self, error_log):
                return {"action": "retry", "rule_id": "R1", "condition": "test"}

        client = DeepSeekClient(DeepSeekConfig(api_key=""))
        result = client.analyze_failure("task", "error", heuristic_engine=FakeHeuristic())
        assert result["root_cause"] == "matched_heuristic"
        assert result["recovery_strategy"] == "retry"
        assert result["source"] == "heuristic"
        assert "Rule R1" in result["preventive_measures"][0]

    def test_analyze_failure_with_heuristic_miss(self):
        """analyze_failure falls back to LLM when heuristic returns None."""
        class FakeHeuristic:
            async def suggest_recovery(self, error_log):
                return None

        client = DeepSeekClient(DeepSeekConfig(api_key=""))
        result = client.analyze_failure("task", "error", heuristic_engine=FakeHeuristic())
        # LLM path also fails because no API key -> error
        assert "error" in result or "root_cause" in result

    def test_analyze_failure_heuristic_exception_fallback(self):
        """analyze_failure catches heuristic exception and falls back."""
        class BadHeuristic:
            async def suggest_recovery(self, error_log):
                raise RuntimeError("boom")

        client = DeepSeekClient(DeepSeekConfig(api_key=""))
        result = client.analyze_failure("task", "error", heuristic_engine=BadHeuristic())
        assert "error" in result or "root_cause" in result

    def test_analyze_failure_no_heuristic(self):
        """analyze_failure without heuristic_engine falls back to LLM."""
        client = DeepSeekClient(DeepSeekConfig(api_key=""))
        result = client.analyze_failure("task", "error")
        assert "error" in result or "root_cause" in result

    def test_generate_skill_description_error_path(self):
        client = DeepSeekClient(DeepSeekConfig(api_key=""))
        result = client.generate_skill_description({"demo": "data"})
        assert "error" in result
        assert result.get("skill_name") == "unknown"
