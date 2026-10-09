"""Read-only action-state mismatch warnings; no parent/child ownership inference."""

from __future__ import annotations

from datetime import datetime

from rosclaw.connectors.ros.intelligence import RosSystemModel


def action_lifecycle_issues(model: RosSystemModel, *, now: datetime) -> list[dict]:
    observed = model.observations.get("action_statuses", {})
    issues = []
    for navigator, state in observed.items():
        if not navigator.endswith("/navigate_to_pose") or not isinstance(state, dict):
            continue
        controller = navigator.rsplit("/", 1)[0] + "/follow_path"
        control_state = observed.get(controller)
        if not isinstance(control_state, dict):
            continue
        try:
            times = [datetime.fromisoformat(s["captured_at"]) for s in (state, control_state)]
            if not all(-100 <= (now - t).total_seconds() * 1000 <= 5000 for t in times):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        # Status arrays are latched transition observations. Do not correlate
        # goals by topic names or call an unrelated active goal an orphan.
        failed = [g for g in state.get("goals", []) if g.get("status") in (5, 6)]
        active = [g for g in control_state.get("goals", []) if g.get("status") in (1, 2, 3)]
        navigating = any(g.get("status") in (1, 2, 3) for g in state.get("goals", []))
        if not failed or not active or navigating:
            continue
        issues.append(
            {
                "issue_code": "NAV2_ACTION_001",
                "severity": "warning",
                "confidence": 1.0,
                "evidence": [
                    {
                        "source": model.snapshot_id,
                        "timestamp": s["captured_at"],
                        "observation": {"server": name, "state": s},
                    }
                    for name, s in ((navigator, state), (controller, control_state))
                ],
                "hypotheses": [
                    "A failed/canceled navigator and an active controller coexist. This may be an orphaned FollowPath goal; parent-child goal ownership is not established by status topics."
                ],
                "next_checks": [
                    "Correlate exact goal UUIDs and ownership using the daemon action ledger; refresh action state and independently measure physical velocity."
                ],
                "recommended_repairs": [
                    "If owned motion may remain active, use the existing guarded emergency-stop path and verify fresh independent stopped-state evidence. Review action acknowledgement/cancel budgets in isolated simulation."
                ],
                "official_sources": [
                    "https://design.ros2.org/articles/actions.html",
                    "https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/core_servers/configuring_bt_navigator/",
                ],
                "runtime_mutation_required": False,
                "physical_stop_verified": False,
                "ownership_verified": False,
            }
        )
    return issues
