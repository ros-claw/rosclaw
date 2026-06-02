"""Additional coverage tests for how/rules.py."""

from unittest.mock import MagicMock

import pytest

from rosclaw.how.rules import RuleManager


class TestRuleManagerUpdate:
    def test_update_rule_empty_fields_returns_false(self):
        seekdb = MagicMock()
        mgr = RuleManager(seekdb)
        result = mgr.update_rule("rule_1")
        assert result is False
        seekdb.update.assert_not_called()

    def test_update_rule_allowed_fields(self):
        seekdb = MagicMock()
        seekdb.update.return_value = True
        mgr = RuleManager(seekdb)
        result = mgr.update_rule("rule_1", condition="new_cond", priority=5)
        assert result is True
        seekdb.update.assert_called_once_with(
            "heuristic_rules", "rule_1", {"condition": "new_cond", "priority": 5}
        )

    def test_update_rule_ignores_disallowed_fields(self):
        seekdb = MagicMock()
        seekdb.update.return_value = True
        mgr = RuleManager(seekdb)
        result = mgr.update_rule("rule_1", condition="c", forbidden="x")
        assert result is True
        seekdb.update.assert_called_once_with(
            "heuristic_rules", "rule_1", {"condition": "c"}
        )

    def test_update_rule_exception_returns_false(self):
        seekdb = MagicMock()
        seekdb.update.side_effect = RuntimeError("db down")
        mgr = RuleManager(seekdb)
        result = mgr.update_rule("rule_1", condition="c")
        assert result is False


class TestRuleManagerDelete:
    def test_delete_rule_tombstone(self):
        seekdb = MagicMock()
        seekdb.update.return_value = True
        mgr = RuleManager(seekdb)
        result = mgr.delete_rule("rule_1")
        assert result is True
        seekdb.update.assert_called_once_with(
            "heuristic_rules", "rule_1", {"priority": -999}
        )

    def test_delete_rule_exception_returns_false(self):
        seekdb = MagicMock()
        seekdb.update.side_effect = RuntimeError("db down")
        mgr = RuleManager(seekdb)
        result = mgr.delete_rule("rule_1")
        assert result is False


class TestRuleManagerList:
    def test_list_rules_filters_negative_priority(self):
        seekdb = MagicMock()
        seekdb.query.return_value = [
            {"id": "r1", "priority": "5"},
            {"id": "r2", "priority": "-1"},
            {"id": "r3", "priority": "0"},
        ]
        mgr = RuleManager(seekdb)
        rules = mgr.list_rules()
        assert len(rules) == 2
        assert rules[0]["id"] == "r1"
        assert rules[1]["id"] == "r3"

    def test_list_rules_empty(self):
        seekdb = MagicMock()
        seekdb.query.return_value = []
        mgr = RuleManager(seekdb)
        rules = mgr.list_rules()
        assert rules == []

    def test_list_rules_all_negative(self):
        seekdb = MagicMock()
        seekdb.query.return_value = [
            {"id": "r1", "priority": "-5"},
            {"id": "r2", "priority": "-1"},
        ]
        mgr = RuleManager(seekdb)
        rules = mgr.list_rules()
        assert rules == []

    def test_list_rules_respects_limit(self):
        seekdb = MagicMock()
        seekdb.query.return_value = []
        mgr = RuleManager(seekdb)
        mgr.list_rules(limit=10)
        seekdb.query.assert_called_once_with("heuristic_rules", limit=10)


class TestRuleManagerGet:
    def test_get_rule_not_found(self):
        seekdb = MagicMock()
        seekdb.query.return_value = []
        mgr = RuleManager(seekdb)
        result = mgr.get_rule("missing")
        assert result is None

    def test_get_rule_found(self):
        seekdb = MagicMock()
        seekdb.query.return_value = [{"id": "rule_1", "condition": "test"}]
        mgr = RuleManager(seekdb)
        result = mgr.get_rule("rule_1")
        assert result == {"id": "rule_1", "condition": "test"}
