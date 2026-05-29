"""rosclaw.how — Heuristic Rules & Recovery Strategies for v1.0.

Provides HeuristicEngine: fast rule-based recovery suggestions that
augment the LLM-based failure analysis with deterministic, cached,
measurable recovery strategies.

Integration:
  Runtime    -> _how = HeuristicEngine(seekdb_client)
  Firewall   -> on block: query _how.suggest_recovery() -> EventBus
  Agent      -> analyze_failure(): try _how first, fall back to LLM
  Practice   -> on praxis.failed: record outcome -> update rule stats
"""
from .engine import HeuristicEngine
from .recovery import RecoveryEngine, RecoveryFormatter, format_recovery_suggestion

__all__ = ["HeuristicEngine", "RecoveryEngine", "RecoveryFormatter", "format_recovery_suggestion"]
