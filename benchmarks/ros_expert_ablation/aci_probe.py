"""Named read-only probes in one additional disposable, never-admitted SDK World.

This adapter is host-created. It cannot read arbitrary model paths or operate a
live stack. Original and refreshed captures are distinct sealed observations;
neither grants current Body/L1 or authority. Full geometry/source admission is
explicitly not a read-only probe and still needs the existing SDK validation.
"""

import hashlib
import json
from pathlib import Path

from benchmarks.ros_expert_ablation.aci_operator_context import aci_from_capture


class SdkReadOnlyProbe:
    def __init__(self, operator):
        self.operator = operator
        self.calls = 0

    def __call__(self, issue_id, check_id, *, remaining_sec):
        op = self.operator
        base = {
            "check_id": check_id,
            "issue_id": issue_id,
            "fresh_probe_performed": False,
            "source": op.source,
            "current_authority": "NONE",
            "current_case_L1": "NOT_EVALUATED",
        }
        if issue_id in {"F-GEOM-001", "F-SOURCE-001"}:
            return {
                **base,
                "status": "NEEDS_OPERATOR_FULL_L1_SOURCE_REVIEW",
                "reason": "full effective-footprint/source admission is not performed by a read-only replica capture",
            }
        if self.calls >= 1 or sum(w["case"] != "positive" for w in op.worlds) >= 2:
            return {**base, "status": "READ_ONLY_PREPARATION_WORLD_BUDGET_EXHAUSTED"}
        if remaining_sec < 40:
            return {**base, "status": "INSUFFICIENT_WALL_FOR_READ_ONLY_SDK_CAPTURE"}
        allowed = {
            "F-TF-001": {"inspect_transform_authorities", "inspect_localization_lifecycle"},
            "F-SCAN-001": {"inspect_sensor_bridge_and_publishers"},
            "F-LIFE-001": {"inspect_managed_nodes_and_transition_failures"},
            "F-ROUTE-001": {
                "identify_ownership_of_competing_endpoints",
                "review_authorized_control_topology",
            },
        }
        if check_id not in allowed.get(issue_id, set()):
            raise ValueError("declared named read-only check required")
        self.calls += 1
        stage = (
            op.healthy_stage
            if op.case == "j100_navigation_onboarding"
            else op.fault_staging_root / op.case / "before_repair"
        )
        path, rc = op.world(op.case, stage, "readonly-probe00", remaining_sec)
        if rc:
            return {
                **base,
                "status": "READ_ONLY_SDK_CAPTURE_FAILED",
                "returncode": rc,
                "original_failure_locator": str(path.relative_to(op.evidence_root)),
                "no_admission_proof": "UNKNOWN; no positive readiness claim",
            }
        path = Path(path)
        marker = json.loads((path / "execution/engineering-capture-ready.json").read_text())
        if (
            marker["navigation_goals_sent"] != 0
            or marker["navigation_workspace_compiled"] is not False
            or any(
                marker[sink][key] != 0
                for sink in ("sink_before", "sink_after")
                for key in ("lease_updates", "received_commands", "published_velocity_messages")
            )
        ):
            raise ValueError("read-only replica must prove no admission, goals or commands")
        fresh = aci_from_capture(path, worktree=op.worktree)
        model = fresh.model
        if (
            fresh.case.source_urdf_sha256
            != hashlib.sha256(op.files["robot.urdf"].encode()).hexdigest()
        ):
            raise ValueError("read-only replica changed original URDF")
        selected = next(c for c in fresh.case.constraints if c.id == issue_id)
        observations = {"constraint_delta_at_capture": selected.model_dump(mode="json")}
        completion = "FRESH_ORIGINAL_OBSERVATIONS_CAPTURED_NOT_ADMISSION"
        if issue_id == "F-TF-001":
            observations["transforms"] = [row.model_dump(mode="json") for row in model.transforms]
            observations["localization_lifecycle"] = [
                row.model_dump(mode="json")
                for row in model.lifecycle
                if row.name == fresh.requirements["lifecycle_nodes"].get("localization")
            ]
            if check_id == "inspect_transform_authorities":
                completion = "PARTIAL_TF_SNAPSHOT_AUTHORITY_REVIEW_STILL_UNKNOWN"
        elif issue_id == "F-SCAN-001":
            topic = fresh.requirements["interface_names"]["sensing.lidar"]
            observations["signals"] = [
                row.model_dump(mode="json") for row in model.signals if row.topic == topic
            ]
            observations["publishers"] = [
                row for row in model.graph.get("topics", []) if row["name"] == topic
            ]
            observations["declared_staging_bridge_state"] = fresh.tools.staging["bringup.json"].get(
                "scan_bridge_state"
            )
            completion = "PARTIAL_SCAN_AND_PUBLISHER_SNAPSHOT_BRIDGE_RUNTIME_CAUSE_NOT_PROVEN"
        elif issue_id == "F-LIFE-001":
            observations["lifecycle"] = [row.model_dump(mode="json") for row in model.lifecycle]
            observations["managed_lists"] = {
                node: params["node_names"]
                for node, params in model.observations.get("node_parameters", {}).items()
                if "node_names" in params
            }
        else:
            observations["competing_routes"] = selected.observed
            observations["operator_owned_fixture_observed"] = bool(fresh.ownership_reviews)
            completion = "PARTIAL_OWNERSHIP_AND_ROUTE_SNAPSHOT_NO_LIVE_CONTROL_AUTHORITY_REVIEW"
        return {
            **base,
            "status": "FRESH_READ_ONLY_REPLICA_OBSERVATION_RECORDED",
            "fresh_probe_performed": True,
            "new_world_run": json.loads((path / "settings.json").read_text())["run_id"],
            "original_capture_ref": op.reference(
                path / "execution/engineering-original-system.json"
            ),
            "captured_at": model.captured_at.isoformat(),
            "snapshot_hash": model.snapshot_hash,
            "check_completion": completion,
            "observations": observations,
            "scope": "DIFFERENT_SEALED_REPLICA_WORLD_FRESH_AT_CAPTURE_NOT_LIVE_SELF_OR_CURRENT_L1",
            "original_case_rebound": False,
            "navigation_goals_sent": 0,
        }
