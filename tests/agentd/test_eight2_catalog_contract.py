"""PR-EIGHT-2 红测试（八审 §4 P0-2）：Catalog 前后端 contract。

红测试先行——后端 pi.capabilities 返回的每个能力桶，TS
capabilities 工具必须全部透出；后端新增桶而前端静默丢弃时
contract 测试必须红（七审实测：compute 桶被丢了一个审查周期）。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERVER = REPO / "src" / "rosclaw" / "agentd" / "pi_bridge" / "server.py"
CAPS_TS = REPO / "packages" / "rosclaw-agent" / "src" / "tools" / "capabilities.ts"
PROMPT = (
    REPO / "src" / "rosclaw" / "agentd" / "context" / "prompts" / "native_agent_v2.md"
)


def _backend_buckets() -> list[str]:
    """pi.capabilities 响应里实际返回的能力桶键。"""
    source = SERVER.read_text(encoding="utf-8")
    match = re.search(
        r'if method == "pi\.capabilities":(.*?)\n        if method ==', source, re.DOTALL
    )
    assert match, "找不到 pi.capabilities handler"
    return sorted(set(re.findall(r'"(\w+_capabilities)"', match.group(1))))


class TestCatalogContract:
    def test_every_backend_bucket_surfaced_in_ts_tool(self) -> None:
        buckets = _backend_buckets()
        assert buckets, "后端无能力桶（解析失败）"
        ts_source = CAPS_TS.read_text(encoding="utf-8")
        missing = [b for b in buckets if b not in ts_source]
        assert not missing, (
            f"后端能力桶未在 TS 工具透出: {missing}——模型看不到这些能力"
        )

    def test_prompt_mentions_four_execution_classes(self) -> None:
        """系统提示词必须同步 observe/compute/action/task 四类语义。"""
        text = PROMPT.read_text(encoding="utf-8")
        for word in ("COMPUTE", "OBSERVE", "PHYSICAL_ACTION"):
            assert word in text, f"提示词缺 {word} 语义"
        assert "rosclaw_compute" in text, "提示词未引 rosclaw_compute"

    def test_prompt_has_no_stale_human_decides_all(self) -> None:
        """删除"所有动作都由人工 operator 决定"的过时描述——默认 SIM
        走 POLICY_AUTO，REAL 始终 rosclawd 门禁。"""
        text = PROMPT.read_text(encoding="utf-8")
        stale_patterns = [
            r"every action.{0,40}human operator",
            r"all actions.{0,40}human",
            r"每个动作.{0,20}人工",
            r"所有动作.{0,20}(人工|operator)",
        ]
        for pattern in stale_patterns:
            assert not re.search(pattern, text, re.IGNORECASE), (
                f"提示词仍含过时人工审批描述: {pattern}"
            )
