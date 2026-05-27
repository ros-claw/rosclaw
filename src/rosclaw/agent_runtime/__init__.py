"""
ROSClaw Agent Runtime - MCP Hub

The Agent Runtime is the bridge between LLMs and the physical world.
It provides:
- MCP (Model Context Protocol) server for tool exposure
- AgentContext: Session state and grounding context for LLMs
- LLMProvider abstraction: DeepSeek, OpenAI, Qwen support
- Integration with all grounding engines through the EventBus
"""

from rosclaw.agent_runtime.mcp_hub import MCPHub, AgentContext
from rosclaw.agent_runtime.llm_provider import (
    LLMProvider,
    LLMConfig,
    DeepSeekProvider,
    OpenAIProvider,
    QwenProvider,
    get_provider,
    list_providers,
    register_provider,
)

# Backward-compatible aliases for documentation
AgentRuntime = AgentContext

# Backward-compatible aliases for old DeepSeek API
DeepSeekClient = DeepSeekProvider
DeepSeekConfig = LLMConfig

__all__ = [
    "MCPHub",
    "AgentContext",
    "AgentRuntime",
    # LLM Provider abstraction
    "LLMProvider",
    "LLMConfig",
    "DeepSeekProvider",
    "OpenAIProvider",
    "QwenProvider",
    "get_provider",
    "list_providers",
    "register_provider",
    # Backward-compatible aliases
    "DeepSeekClient",
    "DeepSeekConfig",
]
