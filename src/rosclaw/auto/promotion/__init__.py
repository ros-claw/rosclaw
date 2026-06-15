"""rosclaw.auto.promotion — Champion promotion, dead-end registry, rollback."""
from .champion_store import ChampionStore
from .gate import PromotionGate
from .lineage import LineageTracker
from .rollback import RollbackManager

__all__ = ["PromotionGate", "ChampionStore", "RollbackManager", "LineageTracker"]
