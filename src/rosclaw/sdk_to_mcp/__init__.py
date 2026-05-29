"""SDK to MCP Asset Compiler — Compile ROSClaw SDK assets to MCP format.

Converts:
- Skill definitions → MCP tool schemas
- Provider manifests → MCP resource templates
- Robot configs → MCP prompts
- e-URDF profiles → MCP tool input schemas
"""

from .compiler import AssetCompiler
from .manifest import MCPManifestBuilder

__all__ = ["AssetCompiler", "MCPManifestBuilder"]
