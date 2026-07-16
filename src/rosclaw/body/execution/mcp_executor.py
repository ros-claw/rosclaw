"""MCP-based body executor (stub).

The ``inspire-rh56-mcp`` server drives the RH56 over CAN 2.0B (11 joints,
0-65535).  This adapter will route ``ActionExecutionRequest`` through that
MCP server's tools (``set_finger_position`` / ``set_all_fingers`` /
``emergency_stop``) when the CAN transport profile is in use.

Until the CAN hardware is available this executor is fail-closed: any
construction attempt raises :class:`ExecutorCommunicationError`.  The binding
gate already prevents RS485 profiles from reaching this class.
"""

from __future__ import annotations

from rosclaw.body.execution.interface import ExecutorCommunicationError


class MCPExecutor:
    """Fail-closed placeholder for the inspire-rh56-mcp CAN executor."""

    def __init__(self, *args, **kwargs):
        raise ExecutorCommunicationError(
            "mcp_executor_unavailable: inspire-rh56-mcp CAN executor is pending "
            "hardware bring-up; use RH56Executor with the RS485 mock transport"
        )
