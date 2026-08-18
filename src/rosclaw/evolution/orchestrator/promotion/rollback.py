"""RollbackManager — skill rollback and restore."""

import logging
from typing import Any

from ..core.champion import Champion
from ..core.patch import Patch

logger = logging.getLogger("rosclaw.auto.promotion.rollback")


class RollbackManager:
    """Manage skill rollback to previous champions."""

    def __init__(self, store_backend: Any):
        self._store = store_backend

    def rollback(self, task_id: str, target_level: str | None = None) -> Champion | None:
        """Rollback to the previous champion for a task.

        If target_level is None, roll back one level from current best.
        """
        champs = [Champion.from_dict(d) for d in self._store.iterate("champions")]
        champs = [c for c in champs if c.task_id == task_id]
        if not champs:
            logger.warning("RollbackManager: no champions found for %s", task_id)
            return None

        level_rank = {"baseline": 0, "sim": 1, "sandbox": 2, "real_candidate": 3, "real": 4}
        champs.sort(key=lambda c: level_rank.get(c.level, 0), reverse=True)

        current = champs[0]
        if target_level is None:
            current_rank = level_rank.get(current.level, 0)
            target_rank = max(0, current_rank - 1)
            target_level = [k for k, v in level_rank.items() if v == target_rank][0]

        # Find champion at target level
        target = next((c for c in champs if c.level == target_level), None)
        if target is None:
            logger.warning("RollbackManager: no champion at level %s for %s", target_level, task_id)
            return None

        logger.info(
            "RollbackManager: rolled back %s from %s to %s", task_id, current.level, target.level
        )
        return target

    def create_rollback_patch(self, from_skill: str, to_skill: str) -> Patch:
        """Generate a rollback patch to restore a previous skill version."""
        patch = Patch(
            id=f"rollback_{from_skill}_to_{to_skill}",
            proposal_id="rollback",
            patch_type="config_patch",
            target_skill=from_skill,
            changes=[
                {
                    "action": "rollback",
                    "from": from_skill,
                    "to": to_skill,
                    "reason": "Manual or automatic rollback",
                }
            ],
            rollback_plan={"restore_from": to_skill},
            human_approval_required=False,
        )
        self._store.save("patches", patch.id, patch.to_dict())
        return patch
