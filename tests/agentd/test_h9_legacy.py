"""PR-H9 结构防回归（总纲 v2 §18.1 第 5 条：CI 禁止重新引入扫描）。

红测试先行——本测试在删除前为红（文件存在/引用存在），删除后转绿，
此后任何重新引入旧内核文件或生产引用的 PR 都会变红。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "rosclaw"

#: §18.1 删除清单（Python 侧）。
DELETED_PATHS = [
    "agentd/pi_task_runner.py",
    "agentd/workers",
    "agentd/control_plane.py",
    "agentd/task_runner.py",
    "agentd/task_compiler.py",
    "agentd/loop.py",
    "agentd/handlers.py",
    "agentd/runner.py",
    "agentd/decisions",
    "contracts/worker",
]

#: 不得再出现在生产代码的引用形态。
BANNED_PATTERNS = [
    r"agentd\.workers",
    r"agentd\.pi_task_runner",
    r"agentd\.control_plane",
    r"agentd\.task_runner",
    r"agentd\.task_compiler",
    r"from rosclaw\.agentd\.loop import",
    r"from rosclaw\.agentd\.handlers import",
    r"from rosclaw\.agentd\.runner import",
    r"agentd\.decisions",
    r"contracts\.worker",
]


def test_deleted_files_stay_deleted() -> None:
    resurrected = [p for p in DELETED_PATHS if (SRC / p).exists()]
    assert not resurrected, f"旧内核文件被重新引入: {resurrected}"


def test_no_production_references() -> None:
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in BANNED_PATTERNS:
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(SRC)}: {pattern}")
    assert not offenders, "生产代码仍引用旧内核:\n" + "\n".join(offenders)


def test_workorder_tool_surface_gone() -> None:
    """WorkOrder/Worker 治理工具不再出现在 dispatcher 工具表。"""
    dispatch = (SRC / "agentd" / "pi_bridge" / "tool_dispatch.py").read_text(
        encoding="utf-8"
    )
    table = re.search(r"_TOOL_TABLE[^=]*= \{(.*?)\n\}", dispatch, re.S)
    assert table is not None
    for banned in (
        "rosclaw_delegate", "rosclaw_check_work", "rosclaw_cancel_work",
        "rosclaw_list_work", "rosclaw_update_work", "rosclaw_retry_work",
        "rosclaw_answer_work", "rosclaw_resume_work", "rosclaw_extend_work",
        "rosclaw_task_submit", "rosclaw_task_pause", "rosclaw_task_resume",
        "rosclaw_task_cancel", "rosclaw_task_steer", "rosclaw_task_answer",
        "rosclaw_task_observe",
    ):
        assert f'"{banned}"' not in table.group(1), f"{banned} 仍在工具表"


def test_legacy_engine_removed_from_cli() -> None:
    """rosclaw chat 不再有 --legacy/非 pi 引擎路径（唯一引擎）。"""
    cli = (SRC / "agentd" / "cli.py").read_text(encoding="utf-8")
    assert "_chat_repl" not in cli
    assert "_chat_tui" not in cli
