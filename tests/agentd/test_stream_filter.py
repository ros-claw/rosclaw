"""DecisionBlockFilter tests: protocol blocks hidden, prose preserved."""

from __future__ import annotations

from rosclaw.agentd.stream_filter import DecisionBlockFilter

DECISION = (
    '```json\n{"schema_version": "rosclaw.decision.v1", "decision_id": "d", '
    '"mission_id": "m", "context_id": "c", "context_revision": 1, '
    '"next_intent": "ANSWER", "summary": "x", "evidence_refs": []}\n```'
)


def _run(pieces: list[str]) -> str:
    out: list[str] = []
    f = DecisionBlockFilter(out.append)
    for p in pieces:
        f.feed(p)
    f.flush()
    return "".join(out)


class TestFilter:
    def test_decision_block_hidden(self) -> None:
        assert _run(["你好，这是回答。", DECISION]) == "你好，这是回答。"

    def test_block_split_across_deltas(self) -> None:
        text = "回答正文" + DECISION
        pieces = [text[i : i + 7] for i in range(0, len(text), 7)]
        assert _run(pieces) == "回答正文"

    def test_ordinary_code_fence_passes(self) -> None:
        text = "看代码：```python\nprint(1)\n``` 完。"
        assert _run([text]) == text

    def test_no_fence_passes_through(self) -> None:
        text = "普通回答，没有任何代码块"
        assert _run([text[:5], text[5:]]) == text

    def test_block_at_start(self) -> None:
        assert _run([DECISION, ""]) == ""

    def test_prose_after_block_never_arrives_by_protocol(self) -> None:
        # The decision block is terminal by protocol; suppression of the
        # remainder is correct. But if no marker follows, prose survives.
        text = '前文```json\n{"other": 1}\n```后文'
        assert _run([text]) == text
