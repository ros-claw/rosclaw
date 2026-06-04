"""rosclaw.auto.promotion — Champion promotion, dead-end registry, rollback."""
from .gate import PromotionGate
from .champion_store import ChampionStore
from .rollback import RollbackManager
from .lineage import LineageTracker

__all__ = ["PromotionGate", "ChampionStore", "RollbackManager", "LineageTracker"]
