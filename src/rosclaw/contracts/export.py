"""JSON Schema export for all v1 contracts (golden-schema testing + tooling).

Usage:
    python -m rosclaw.contracts.export [out_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rosclaw.contracts.agent.context import EmbodiedContextBundleV1
from rosclaw.contracts.agent.decision import DecisionV1
from rosclaw.contracts.agent.mission import MissionSessionV1
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1
from rosclaw.contracts.agent.task_graph import TaskGraphPatchV1, TaskGraphV1, TaskNodeV1
from rosclaw.contracts.agent.tool import ToolDescriptorV2
from rosclaw.contracts.common import ContractModel
from rosclaw.contracts.operator.approval import ApprovalRequestV2
from rosclaw.contracts.operator.grant import MissionGrantV1
from rosclaw.contracts.team.member import TeamMemberCardV1
from rosclaw.contracts.team.role import RoleLeaseV1
from rosclaw.contracts.team.world import SharedWorldDeltaV1, SharedWorldSnapshotV1
from rosclaw.contracts.worker.card import WorkerCardV1
from rosclaw.contracts.worker.order import WorkOrderV1, WorkResultV1

#: All top-level v1 contracts, keyed by schema stem.
ALL_CONTRACTS: dict[str, type[ContractModel]] = {
    cls.SCHEMA: cls
    for cls in (
        MissionSessionV1,
        TaskNodeV1,
        TaskGraphV1,
        TaskGraphPatchV1,
        EmbodiedContextBundleV1,
        DecisionV1,
        ModelTurnResultV1,
        WorkerCardV1,
        WorkOrderV1,
        WorkResultV1,
        TeamMemberCardV1,
        RoleLeaseV1,
        SharedWorldSnapshotV1,
        SharedWorldDeltaV1,
        MissionGrantV1,
        ApprovalRequestV2,
        ToolDescriptorV2,
    )
}


def export_json_schemas(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for schema, cls in sorted(ALL_CONTRACTS.items()):
        doc = cls.model_json_schema()
        doc["$id"] = f"rosclaw://schemas/{schema}"
        doc["title"] = schema
        path = out_dir / f"{schema}.json"
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("schemas_out")
    for path in export_json_schemas(target):
        print(path)
