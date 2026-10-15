"""Startup diagnostics remain readable; unexpected native tools fail closed."""

import pytest

from benchmarks.ros_expert_ablation.host_client import native_tool_items


def test_successful_structured_turn_with_startup_diagnostic_is_allowed():
    events = [
        {"type": "thread.started", "thread_id": "example"},
        {
            "type": "item.completed",
            "item": {
                "type": "error",
                "message": "Code Mode is unavailable because code-mode host is disabled.",
            },
        },
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": '{"operation":"finish"}'},
        },
        {"type": "turn.completed", "usage": {"input_tokens": 10684, "output_tokens": 16}},
    ]
    assert native_tool_items(events) == []


@pytest.mark.parametrize("phase", ["item.started", "item.updated", "item.completed"])
@pytest.mark.parametrize(
    "kind", ["command_execution", "mcp_tool_call", "file_change", "future_unknown_tool", None]
)
def test_unexpected_execution_is_detected_even_without_completion(phase, kind):
    assert native_tool_items([{"type": phase, "item": {"type": kind}}]) == [kind]
