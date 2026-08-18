"""DashboardExporter — export Auto evolution data for rosclaw-dashboard."""

import json
from datetime import UTC, datetime
from typing import Any


class DashboardExporter:
    """Export Auto evolution data as dashboard-consumable JSON.

    Output schema matches rosclaw-dashboard expectations:
    - summary: high-level KPIs
    - tasks: list of active tasks
    - champions: champion leaderboard
    - deadends: rejected directions
    - lineage: skill evolution graph
    - timeline: experiment events over time
    """

    def __init__(self, engine: Any):
        self._engine = engine

    def export(self, task_id: str | None = None) -> dict:
        """Export full dashboard dataset."""
        data = {
            "generated_at": datetime.now(UTC).isoformat(),
            "summary": self._build_summary(task_id),
            "tasks": self._build_tasks(task_id),
            "champions": self._build_champions(task_id),
            "deadends": self._build_deadends(task_id),
            "lineage": self._build_lineage(task_id),
            "timeline": self._build_timeline(task_id),
        }
        return data

    def export_json(self, task_id: str | None = None) -> str:
        return json.dumps(self.export(task_id), indent=2, default=str)

    def _build_summary(self, task_id: str | None = None) -> dict:
        tasks = self._engine.list_tasks()
        if task_id:
            tasks = [t for t in tasks if t.id == task_id]
        total_proposals = len(self._engine.list_proposals())
        total_champions = len(self._engine.list_champions())
        total_deadends = len(self._engine.list_deadends())
        return {
            "total_tasks": len(tasks),
            "total_proposals": total_proposals,
            "total_champions": total_champions,
            "total_deadends": total_deadends,
            "evolution_success_rate": self._compute_success_rate(),
        }

    def _compute_success_rate(self) -> float:
        champs = len(self._engine.list_champions())
        proposals = max(1, len(self._engine.list_proposals()))
        return round(champs / proposals, 3)

    def _build_tasks(self, task_id: str | None = None) -> list[dict]:
        tasks = self._engine.list_tasks()
        if task_id:
            tasks = [t for t in tasks if t.id == task_id]
        result = []
        for t in tasks:
            result.append(
                {
                    "id": t.id,
                    "name": t.name,
                    "status": t.status,
                    "robot_id": t.robot_id,
                    "target_skill_id": t.target_skill_id,
                    "proposals": len(self._engine.list_proposals(t.name)),
                    "champions": len(self._engine.list_champions(t.id)),
                    "deadends": len(self._engine.list_deadends(t.id)),
                }
            )
        return result

    def _build_champions(self, task_id: str | None = None) -> list[dict]:
        champs = self._engine.list_champions(task_id)
        return [
            {
                "id": c.id,
                "skill_id": c.skill_id,
                "level": c.level,
                "task_id": c.task_id,
                "metrics": c.metrics,
                "parent_skill_id": c.parent_skill_id,
                "promotion_date": c.created_at,
            }
            for c in champs
        ]

    def _build_deadends(self, task_id: str | None = None) -> list[dict]:
        des = self._engine.list_deadends(task_id)
        return [
            {
                "id": d.id,
                "task_id": d.task_id,
                "direction": d.direction,
                "rejection_reason": d.rejection_reason,
                "evidence": d.evidence,
            }
            for d in des
        ]

    def _build_lineage(self, task_id: str | None = None) -> list[dict]:
        """Export lineage as nodes + edges graph format."""
        nodes = []
        edges = []
        seen_skills = set()

        # Gather all champion skills
        champs = self._engine.list_champions(task_id)
        for champ in champs:
            if champ.skill_id not in seen_skills:
                seen_skills.add(champ.skill_id)
                nodes.append(
                    {
                        "id": champ.skill_id,
                        "level": champ.level,
                        "metrics": champ.metrics,
                    }
                )
            if champ.parent_skill_id and champ.parent_skill_id not in seen_skills:
                seen_skills.add(champ.parent_skill_id)
                nodes.append(
                    {
                        "id": champ.parent_skill_id,
                        "level": "baseline",
                        "metrics": {},
                    }
                )
            if champ.parent_skill_id:
                edges.append(
                    {
                        "from": champ.parent_skill_id,
                        "to": champ.skill_id,
                        "patch_id": champ.patch_id,
                        "experiment_id": champ.experiment_id,
                    }
                )
        return {"nodes": nodes, "edges": edges}

    def _build_timeline(self, task_id: str | None = None) -> list[dict]:
        """Build experiment timeline for dashboard charts."""
        from ..core.experiment import ExperimentSpec

        timeline = []
        # Build set of matching task identifiers (id or name)
        matching_tasks = set()
        if task_id:
            matching_tasks.add(task_id)
            for t in self._engine.list_tasks():
                if t.id == task_id:
                    matching_tasks.add(t.name)
        for raw in self._engine.store.iterate("experiments"):
            exp = ExperimentSpec.from_dict(raw)
            if task_id and exp.task not in matching_tasks:
                continue
            timeline.append(
                {
                    "experiment_id": exp.id,
                    "task": exp.task,
                    "status": exp.status,
                    "baseline_skill": exp.baseline_skill_id,
                    "candidate_skill": exp.candidate_skill_id,
                    "created_at": exp.created_at,
                }
            )
        return sorted(timeline, key=lambda x: x["created_at"])
