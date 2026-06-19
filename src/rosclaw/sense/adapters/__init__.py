"""Adapter stubs for rosclaw.sense integration with other modules."""

from rosclaw.sense.adapters.auto_context import AutoContextAdapter
from rosclaw.sense.adapters.how_context import HowContextAdapter
from rosclaw.sense.adapters.memory_writer import MemoryWriterAdapter
from rosclaw.sense.adapters.practice_writer import PracticeWriterAdapter
from rosclaw.sense.adapters.sandbox_context import SandboxContextAdapter
from rosclaw.sense.adapters.skill_requirements import SkillRequirementsAdapter

__all__ = [
    "AutoContextAdapter",
    "HowContextAdapter",
    "MemoryWriterAdapter",
    "PracticeWriterAdapter",
    "SandboxContextAdapter",
    "SkillRequirementsAdapter",
]
