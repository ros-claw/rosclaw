"""ROSClaw agent onboarding package.

Provides project-level file generation, validation, and diagnostics for
integrating Claude Code (and future agents) with a ROSClaw runtime.
"""

from __future__ import annotations

from rosclaw.agent.doctor import cmd_agent_doctor_claude_code
from rosclaw.agent.init_claude_code import cmd_agent_init_claude_code
from rosclaw.agent.test_claude_code import cmd_agent_test_claude_code

__all__ = [
    "cmd_agent_init_claude_code",
    "cmd_agent_doctor_claude_code",
    "cmd_agent_test_claude_code",
]
