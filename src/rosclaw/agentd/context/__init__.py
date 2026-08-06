"""Embodied context compilation: sources, compiler, prompt registry."""

from rosclaw.agentd.context.compiler import (
    COMPILER_VERSION,
    CompilationError,
    ContextCompiler,
    StaleSourceError,
    wrap_untrusted,
)
from rosclaw.agentd.context.prompt_registry import PromptInfo, list_prompts, load_prompt
from rosclaw.agentd.context.sources import (
    BodyFacts,
    CapabilityInfo,
    ConsentFacts,
    ConversationMessage,
    EvidenceClass,
    MemoryItem,
    OrgFacts,
    SelfFacts,
    SourceBundle,
)
from rosclaw.agentd.context.tokens import estimate_tokens

__all__ = [
    "COMPILER_VERSION",
    "BodyFacts",
    "CapabilityInfo",
    "CompilationError",
    "ConsentFacts",
    "ContextCompiler",
    "ConversationMessage",
    "EvidenceClass",
    "MemoryItem",
    "OrgFacts",
    "PromptInfo",
    "SelfFacts",
    "SourceBundle",
    "StaleSourceError",
    "estimate_tokens",
    "list_prompts",
    "load_prompt",
    "wrap_untrusted",
]
