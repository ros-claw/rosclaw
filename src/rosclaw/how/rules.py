"""rosclaw_how.rules — Rule CRUD and management utilities.

Thin wrapper around SeekDB heuristic_rules table for administrative
operations (add, update, delete, list). The HeuristicEngine hot path
avoids this module for latency; it is used by admin tools and tests.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("rosclaw.how.rules")


class RuleManager:
    """Administrative CRUD for heuristic rules."""

    def __init__(self, seekdb_client: Any) -> None:
        self._seekdb = seekdb_client
        self._table = "heuristic_rules"

    def add_rule(
        self,
        rule_id: str,
        condition: str,
        action: str,
        priority: int = 0,
    ) -> str:
        """Insert a new rule. Returns rule_id."""
        self._seekdb.insert(self._table, {
            "id": rule_id,
            "condition": condition,
            "action": action,
            "priority": priority,
            "success_count": 0,
            "failure_count": 0,
            "last_triggered": None,
        })
        logger.info("Rule added: %s (condition=%s)", rule_id, condition)
        return rule_id

    def update_rule(self, rule_id: str, **fields: Any) -> bool:
        """Update rule fields. Allowed: condition, action, priority."""
        allowed = {"condition", "action", "priority"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        try:
            return self._seekdb.update(self._table, rule_id, updates)
        except Exception as exc:  # noqa: BLE001
            logger.warning("update_rule %s failed: %s", rule_id, exc)
            return False

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by id."""
        try:
            # SeekDBClient interface doesn't expose delete;
            # use update with a tombstone flag (v1.0 limitation)
            return self._seekdb.update(self._table, rule_id, {"priority": -999})
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_rule %s failed: %s", rule_id, exc)
            return False

    def list_rules(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return all active rules (priority >= 0)."""
        rows = self._seekdb.query(self._table, limit=limit)
        return [r for r in rows if int(r.get("priority", 0)) >= 0]

    def get_rule(self, rule_id: str) -> dict[str, Any] | None:
        """Get a single rule by id."""
        rows = self._seekdb.query(self._table, filters={"id": rule_id}, limit=1)
        return dict(rows[0]) if rows else None
