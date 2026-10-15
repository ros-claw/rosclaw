"""CLI diagnostics are retained without being mistaken for native execution."""



def test_diagnostic_item_is_not_a_native_tool():
    events = [
        {"type": "item.completed", "item": {"type": "error", "message": "Code mode disabled"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
    ]
    allowed = {"context", "agent_message", "reasoning", "error", "warning"}
    assert not [e for e in events if e["item"]["type"] not in allowed]
