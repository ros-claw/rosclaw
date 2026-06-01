"""Tests for LLM Provider abstraction layer."""

from unittest.mock import MagicMock, patch

import pytest

from rosclaw.agent_runtime.llm_provider import (
    LLMConfig,
    LLMProvider,
    DeepSeekProvider,
    OpenAIProvider,
    QwenProvider,
    get_provider,
    list_providers,
    register_provider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai_client():
    """Return a mock OpenAI-compatible client."""
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = '{"result": "ok"}'
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def mock_openai_module(mock_openai_client):
    """Patch the openai module import."""
    mock_module = MagicMock()
    mock_module.OpenAI.return_value = mock_openai_client
    with patch("builtins.__import__", lambda name, *args, **kwargs: mock_module if name == "openai" else __builtins__["__import__"](name, *args, **kwargs)):
        yield mock_module


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

def test_llm_config_defaults():
    cfg = LLMConfig()
    assert cfg.api_key == ""
    assert cfg.base_url == ""
    assert cfg.model == ""
    assert cfg.temperature == 0.7
    assert cfg.max_tokens == 4096
    assert cfg.timeout == 30.0


def test_llm_config_override():
    cfg = LLMConfig(api_key="sk-test", model="gpt-4", temperature=0.5)
    assert cfg.api_key == "sk-test"
    assert cfg.model == "gpt-4"
    assert cfg.temperature == 0.5


# ---------------------------------------------------------------------------
# Provider instantiation tests
# ---------------------------------------------------------------------------

def test_deepseek_provider_env_vars(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deep")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://custom.deepseek.com")

    provider = DeepSeekProvider()
    assert provider.config.api_key == "sk-deep"
    assert provider.config.base_url == "https://custom.deepseek.com"
    assert provider.config.model == "deepseek-v4-pro"
    assert provider.provider_name == "deepseek"


def test_deepseek_provider_explicit_config():
    cfg = LLMConfig(api_key="sk-explicit", model="deepseek-chat")
    provider = DeepSeekProvider(cfg)
    assert provider.config.api_key == "sk-explicit"
    assert provider.config.model == "deepseek-chat"


def test_openai_provider_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    provider = OpenAIProvider()
    assert provider.config.api_key == "sk-openai"
    assert provider.config.model == "gpt-4o"
    assert provider.provider_name == "openai"


def test_openai_provider_explicit_config():
    cfg = LLMConfig(api_key="sk-oai", model="gpt-4-turbo")
    provider = OpenAIProvider(cfg)
    assert provider.config.api_key == "sk-oai"
    assert provider.config.model == "gpt-4-turbo"


def test_qwen_provider_env_vars(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dash")
    provider = QwenProvider()
    assert provider.config.api_key == "sk-dash"
    assert provider.config.model == "qwen-max"
    assert provider.provider_name == "qwen"


def test_qwen_provider_explicit_config():
    cfg = LLMConfig(api_key="sk-qwen", model="qwen-plus")
    provider = QwenProvider(cfg)
    assert provider.config.api_key == "sk-qwen"
    assert provider.config.model == "qwen-plus"


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------

def test_get_provider_deepseek():
    provider = get_provider("deepseek", LLMConfig(api_key="sk-test"))
    assert isinstance(provider, DeepSeekProvider)
    assert provider.provider_name == "deepseek"


def test_get_provider_openai():
    provider = get_provider("openai", LLMConfig(api_key="sk-test"))
    assert isinstance(provider, OpenAIProvider)
    assert provider.provider_name == "openai"


def test_get_provider_qwen():
    provider = get_provider("qwen", LLMConfig(api_key="sk-test"))
    assert isinstance(provider, QwenProvider)
    assert provider.provider_name == "qwen"


def test_get_provider_case_insensitive():
    provider = get_provider("DeepSeek", LLMConfig(api_key="sk-test"))
    assert isinstance(provider, DeepSeekProvider)


def test_get_provider_unknown():
    with pytest.raises(ValueError, match="Unknown LLM provider 'unknown'"):
        get_provider("unknown")


def test_list_providers():
    providers = list_providers()
    assert "deepseek" in providers
    assert "openai" in providers
    assert "qwen" in providers


# ---------------------------------------------------------------------------
# Custom provider registration
# ---------------------------------------------------------------------------

def test_register_custom_provider():
    class CustomProvider(LLMProvider):
        @property
        def provider_name(self):
            return "custom"

        def _create_client(self):
            return MagicMock()

    register_provider("custom", CustomProvider)
    provider = get_provider("custom")
    assert isinstance(provider, CustomProvider)
    assert provider.provider_name == "custom"


def test_register_provider_must_subclass():
    with pytest.raises(TypeError, match="must subclass LLMProvider"):
        register_provider("bad", str)


# ---------------------------------------------------------------------------
# API consistency tests (all providers must support same interface)
# ---------------------------------------------------------------------------

def test_all_providers_have_same_methods():
    required_methods = [
        "plan_task",
        "analyze_failure",
        "generate_skill_description",
        "health_check",
        "provider_name",
    ]
    for cls in [DeepSeekProvider, OpenAIProvider, QwenProvider]:
        for method in required_methods:
            assert hasattr(cls, method), f"{cls.__name__} missing {method}"


# ---------------------------------------------------------------------------
# Functional tests with mocked client
# ---------------------------------------------------------------------------

def test_plan_task(mock_openai_client):
    provider = DeepSeekProvider(LLMConfig(api_key="sk-test"))
    provider._client = mock_openai_client

    result = provider.plan_task("move arm", {"joints": 6})
    assert result == {"result": "ok"}
    mock_openai_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "deepseek-v4-pro"
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_analyze_failure(mock_openai_client):
    provider = DeepSeekProvider(LLMConfig(api_key="sk-test"))
    provider._client = mock_openai_client

    result = provider.analyze_failure("grasp failed", "timeout error")
    assert result == {"result": "ok"}


def test_generate_skill_description(mock_openai_client):
    provider = DeepSeekProvider(LLMConfig(api_key="sk-test"))
    provider._client = mock_openai_client

    result = provider.generate_skill_description({"trajectory": [[0, 0]]})
    assert result == {"result": "ok"}


def test_plan_task_error_handling(mock_openai_client):
    provider = DeepSeekProvider(LLMConfig(api_key="sk-test"))
    mock_openai_client.chat.completions.create.side_effect = RuntimeError("API down")
    provider._client = mock_openai_client

    result = provider.plan_task("move arm", {"joints": 6})
    assert "error" in result
    assert result["task_name"] == "failed"


def test_health_check_success(mock_openai_client):
    provider = DeepSeekProvider(LLMConfig(api_key="sk-test"))
    provider._client = mock_openai_client

    result = provider.health_check()
    assert result["ok"] is True
    assert result["provider"] == "deepseek"


def test_health_check_failure(mock_openai_client):
    provider = DeepSeekProvider(LLMConfig(api_key="sk-test"))
    mock_openai_client.chat.completions.create.side_effect = RuntimeError("connection refused")
    provider._client = mock_openai_client

    result = provider.health_check()
    assert result["ok"] is False
    assert result["provider"] == "deepseek"
    assert "connection refused" in result["error"]


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------

def test_deepseek_client_alias():
    from rosclaw.agent_runtime import DeepSeekClient, DeepSeekConfig
    assert DeepSeekClient is DeepSeekProvider
    assert DeepSeekConfig is LLMConfig


def test_old_imports_still_work():
    from rosclaw import DeepSeekClient, DeepSeekConfig
    assert DeepSeekClient is DeepSeekProvider
    assert DeepSeekConfig is LLMConfig
