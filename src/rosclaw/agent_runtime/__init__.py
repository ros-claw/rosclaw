"""
ROSClaw Agent Runtime - MCP Hub

The Agent Runtime is the bridge between LLMs and the physical world.
It provides:
- MCP (Model Context Protocol) server for tool exposure
- AgentContext: Session state and grounding context for LLMs
- Integration with all grounding engines through the EventBus
"""

from rosclaw.agent_runtime.mcp_hub import MCPHub, AgentContext
from rosclaw.agent_runtime.ai_collaboration import DeepSeekClient, DeepSeekConfig

# Backward-compatible alias for documentation
AgentRuntime = AgentContext

__all__ = ["MCPHub", "AgentContext", "AgentRuntime", "DeepSeekClient", "DeepSeekConfig"]
