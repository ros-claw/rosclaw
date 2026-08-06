"""UI contracts package (批次 B)：命令、快照、交互的语义源。

TUI/ACP/Web 客户端共享这些契约；TypeScript 类型由 JSON Schema 导出生成。
"""

from rosclaw.contracts.ui.commands import (
    CommandCategory,
    CommandOwner,
    CommandRequestV1,
    CommandResultV1,
    CommandSpecV1,
)
from rosclaw.contracts.ui.interactions import InteractionRequestV1
from rosclaw.contracts.ui.snapshots import MissionSnapshotV1

__all__ = [
    "CommandCategory",
    "CommandOwner",
    "CommandRequestV1",
    "CommandResultV1",
    "CommandSpecV1",
    "InteractionRequestV1",
    "MissionSnapshotV1",
]
