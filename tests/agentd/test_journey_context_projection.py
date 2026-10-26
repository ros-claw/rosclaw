"""Fixture model decisions must see tool results even after context-only refresh."""

import pytest

from tests.agentd.test_h4_verifier_live import _VerifierFake
from tests.agentd.test_hf5_4_action_result_ui import _LieTwiceModel
from tests.agentd.test_product_journey import _FakeModel


@pytest.mark.parametrize(
    "model,text,call_id,expected",
    [
        (_FakeModel, "位姿移动测试", "call_star_action", "位姿移动测试完成"),
        (_LieTwiceModel, "执行一个假动作", "call_lie_1", "动作已执行"),
        (_VerifierFake, "写入并交付hello.txt", "call_deliver", "hello.txt 已写入并交付"),
    ],
)
def test_context_only_message_does_not_hide_last_tool_result(model, text, call_id, expected):
    body = {
        "stream": True,
        "messages": [
            {"role": "user", "content": text},
            {"role": "tool", "tool_call_id": call_id, "content": "COMPLETED"},
            {"role": "user", "content": "<ROSCLAW_TRUSTED_CONTEXT> fresh observation only"},
        ],
    }
    response = model().answer(body).decode()
    assert "tool_calls" not in response
    assert expected in response


def test_latest_real_user_message_still_starts_a_new_decision():
    body = {
        "stream": True,
        "messages": [
            {"role": "tool", "tool_call_id": "call_star_action", "content": "COMPLETED"},
            {"role": "user", "content": "<ROSCLAW_TRUSTED_CONTEXT> fresh"},
            {"role": "user", "content": "回到零点"},
        ],
    }
    response = _FakeModel().answer(body).decode()
    assert "call_home_action" in response
