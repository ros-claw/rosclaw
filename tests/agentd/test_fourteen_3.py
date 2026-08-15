"""十四审 PR-14.3 红测试：WorkerTranscriptStore——完整公开 transcript
分页读取（总纲 §4.4）。

红测试先行——修复前必须红：
1. 结构化记录（tseq/channel）双向分页（after_seq/before_seq/limit）；
2. channel 过滤（conversation/tools/files/artifacts/usage/control）；
3. 十二审 legacy 格式（无 tseq/channel）兼容映射为 conversation；
4. 完整 assistant 文本不再只有 4000 字节尾部切片。
"""

from __future__ import annotations

import json
from pathlib import Path

from rosclaw.agentd.workers.transcript_store import TranscriptStore


def _write(home: Path, wo: str, records: list[dict]) -> None:
    d = home / "work" / wo
    d.mkdir(parents=True, exist_ok=True)
    (d / "transcript.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


class TestStructuredPagination:
    def test_after_seq_forward_page(self, tmp_path: Path) -> None:
        records = [
            {"tseq": i, "channel": "conversation", "role": "assistant",
             "text": f"msg-{i}", "ts": "t"}
            for i in range(1, 21)
        ]
        _write(tmp_path, "wo_a", records)
        store = TranscriptStore(tmp_path)
        page = store.read_page("wo_a", after_seq=0, limit=5)
        assert [r["tseq"] for r in page["records"]] == [1, 2, 3, 4, 5]
        assert page["has_more"] is True
        assert page["next_cursor"] == 5
        page2 = store.read_page("wo_a", after_seq=page["next_cursor"], limit=5)
        assert [r["tseq"] for r in page2["records"]] == [6, 7, 8, 9, 10]

    def test_before_seq_backward_page(self, tmp_path: Path) -> None:
        records = [
            {"tseq": i, "channel": "conversation", "role": "assistant",
             "text": f"msg-{i}", "ts": "t"}
            for i in range(1, 21)
        ]
        _write(tmp_path, "wo_b", records)
        store = TranscriptStore(tmp_path)
        page = store.read_page("wo_b", before_seq=18, limit=5)
        assert [r["tseq"] for r in page["records"]] == [13, 14, 15, 16, 17]
        assert page["has_more"] is True

    def test_channel_filter(self, tmp_path: Path) -> None:
        _write(tmp_path, "wo_c", [
            {"tseq": 1, "channel": "conversation", "role": "assistant",
             "text": "你好", "ts": "t"},
            {"tseq": 2, "channel": "tools", "phase": "start", "tool": "bash",
             "args": "python3 sim.py", "ts": "t"},
            {"tseq": 3, "channel": "tools", "phase": "end", "tool": "bash",
             "is_error": False, "output": "ok 2379 points", "ts": "t"},
            {"tseq": 4, "channel": "files", "op": "write", "path": "sim.py",
             "bytes": 420, "ts": "t"},
            {"tseq": 5, "channel": "artifacts",
             "files": [{"name": "star5.gif", "bytes": 8192, "sha256": "ab"}], "ts": "t"},
            {"tseq": 6, "channel": "usage", "input": 100, "output": 50, "ts": "t"},
        ])
        store = TranscriptStore(tmp_path)
        tools = store.read_page("wo_c", channel="tools", limit=50)
        assert [r["tseq"] for r in tools["records"]] == [2, 3]
        conv = store.read_page("wo_c", channel="conversation", limit=50)
        assert [r["tseq"] for r in conv["records"]] == [1]
        artifacts = store.read_page("wo_c", channel="artifacts", limit=50)
        assert artifacts["records"][0]["files"][0]["name"] == "star5.gif"

    def test_total_and_empty(self, tmp_path: Path) -> None:
        _write(tmp_path, "wo_d", [
            {"tseq": 1, "channel": "conversation", "role": "assistant",
             "text": "x", "ts": "t"},
        ])
        store = TranscriptStore(tmp_path)
        page = store.read_page("wo_d", limit=50)
        assert page["total"] == 1
        assert page["has_more"] is False
        missing = store.read_page("wo_missing", limit=50)
        assert missing["records"] == [] and missing["total"] == 0


class TestLegacyCompat:
    def test_legacy_records_mapped(self, tmp_path: Path) -> None:
        """十二审格式（{ts, role, text} 无 tseq/channel）→ conversation
        channel，行号为合成 tseq。"""
        _write(tmp_path, "wo_legacy", [
            {"ts": "t1", "role": "assistant", "text": "第一段"},
            {"ts": "t2", "role": "tool", "tool": "bash", "is_error": False},
            {"ts": "t3", "role": "assistant", "text": "第二段"},
        ])
        store = TranscriptStore(tmp_path)
        page = store.read_page("wo_legacy", limit=50)
        assert page["total"] == 3
        assert [r["tseq"] for r in page["records"]] == [1, 2, 3]
        assert all(r["channel"] in ("conversation", "tools")
                   for r in page["records"])
        page2 = store.read_page("wo_legacy", after_seq=2, limit=50)
        assert [r["tseq"] for r in page2["records"]] == [3]

    def test_full_text_not_tail_slice(self, tmp_path: Path) -> None:
        """超过 4000 字节的 transcript 必须可完整分页读取（旧实现只给
        尾部 4000 字节切片）。"""
        big = "长文本" * 4000  # 12000 字符
        _write(tmp_path, "wo_big", [
            {"tseq": 1, "channel": "conversation", "role": "assistant",
             "text": big, "ts": "t"},
            {"tseq": 2, "channel": "conversation", "role": "assistant",
             "text": "结尾", "ts": "t"},
        ])
        store = TranscriptStore(tmp_path)
        page = store.read_page("wo_big", limit=50)
        assert page["records"][0]["text"] == big
        assert page["records"][1]["text"] == "结尾"
