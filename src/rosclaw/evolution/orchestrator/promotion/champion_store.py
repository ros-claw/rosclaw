"""ChampionStore — persistent champion skill registry."""

import logging
from typing import Any

from ..core.champion import Champion

logger = logging.getLogger("rosclaw.auto.promotion.champion_store")


class ChampionStore:
    """Manage champion skills with lineage and rollback support."""

    def __init__(self, store_backend: Any):
        self._store = store_backend

    def save_champion(self, champion: Champion) -> None:
        self._store.save("champions", champion.id, champion.to_dict())
        logger.info("ChampionStore: saved %s (level=%s)", champion.id, champion.level)

    def get_champion(self, task_id: str, level: str | None = None) -> Champion | None:
        champs = [Champion.from_dict(d) for d in self._store.iterate("champions")]
        champs = [c for c in champs if c.task_id == task_id]
        if level:
            champs = [c for c in champs if c.level == level]
        return champs[-1] if champs else None

    def list_champions(self, task_id: str | None = None) -> list[Champion]:
        champs = [Champion.from_dict(d) for d in self._store.iterate("champions")]
        if task_id:
            champs = [c for c in champs if c.task_id == task_id]
        return champs

    def get_best_champion(self, task_id: str) -> Champion | None:
        """Return the highest-level champion for a task."""
        champs = self.list_champions(task_id)
        if not champs:
            return None
        level_rank = {"baseline": 0, "sim": 1, "sandbox": 2, "real_candidate": 3, "real": 4}
        return max(champs, key=lambda c: level_rank.get(c.level, 0))

    def deprecate(self, champion_id: str, reason: str = "") -> bool:
        data = self._store.load("champions", champion_id)
        if not data:
            return False
        data["status"] = "deprecated"
        data["deprecation_reason"] = reason
        self._store.save("champions", champion_id, data)
        logger.info("ChampionStore: deprecated %s — %s", champion_id, reason)
        return True
