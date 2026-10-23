"""SDK-only, parked operator TF input preparation after canonical healthy goals.

No actions, lifecycle transitions, publishers, threshold changes or model calls.
The external operator independently binds and reviews each original capture.
"""

import hashlib
import json
import os
from collections import deque
from datetime import UTC, datetime
from pathlib import Path


def original_ecm_tail(root):
    """Copy unmodified recent records; external review must verify the full chain."""
    paths = [root / "ecm-original.jsonl"] + sorted(root.glob("ecm-original.jsonl.segment-*.jsonl"))
    if len(paths) > 8:
        raise ValueError("registered source stream segment budget exceeded")
    records = deque(maxlen=64)
    references = []
    for index, path in enumerate(paths):
        expected = (
            "ecm-original.jsonl" if index == 0 else f"ecm-original.jsonl.segment-{index:06}.jsonl"
        )
        if path.name != expected or path.is_symlink() or path.stat().st_size > 15_000_000:
            raise ValueError("original bounded contiguous source segments required")
        raw = path.read_bytes()
        complete = raw[: raw.rfind(b"\n") + 1]
        references.append(
            {
                "file": path.name,
                "prefix_bytes": len(complete),
                "prefix_sha256": hashlib.sha256(complete).hexdigest(),
            }
        )
        records.extend(complete.splitlines(keepends=True))
    if len(records) < 2:
        raise ValueError("advancing original source capture required")
    return b"".join(records), references


def validate_policy(settings, *, isolated):
    policy = settings.get("tf_input_preparation")
    if (
        not isolated
        or settings.get("generic_source_observer") is not True
        or settings.get("fault") != "none"
        or "engineering_capture" in settings
        or "governed_sink" not in settings
        or type(policy) is not dict
        or set(policy) != {"schema_version", "run_id", "seed"}
        or policy["schema_version"] != "rosclaw.tf_input_preparation.v1"
        or policy["run_id"] != settings.get("run_id")
        or type(policy["seed"]) is not int
        or policy["seed"] not in {101, 202}
        or policy["seed"] != settings.get("seed")
    ):
        raise ValueError("closed isolated operator TF input preparation required")
    return policy


def capture_tf_inputs(root, settings, state, node, spin_until, save):
    policy = validate_policy(
        settings,
        isolated=Path("/.dockerenv").is_file() and os.getenv("ROS_LOCALHOST_ONLY") == "1",
    )
    ack = json.loads((root / "host_capture_ack.json").read_text())
    if ack.get("journey_returncode") != 0 or ack.get("initial_L1") != "READY_FOR_SIM_TEST":
        raise ValueError("independently accepted healthy canonical journey required")
    receipts = [
        json.loads(p.read_text())["receipt"]
        for p in root.glob("governed-action-*-receipt-original.json")
    ]
    if len(receipts) != 2 or any(
        r["final_state"] != "COMPLETED"
        or r["verified"] is not True
        or r["evidence_domain"] != "SIMULATION"
        or r["verification_result"].get("physical_stop_verified") is not True
        for r in receipts
    ):
        raise ValueError("two original stopped canonical SIM receipts required")
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters

    from rosclaw.connectors.ros.context.collision_assets import bind_prepared_collision_originals
    from rosclaw.connectors.ros.engineering import compile_engineering_case

    requirements = json.loads(Path("/assets/requirements.json").read_text())
    urdf = Path("/assets/robot.urdf").read_bytes()
    loc = requirements["lifecycle_nodes"]["localization"]
    # Both immutable configurations are registered before the World starts.
    # The fault file drives a runtime override, not a claim of cold-start loading.
    from copy import deepcopy

    import yaml

    startup_path = Path("/assets/nav2.yaml")
    fault_path = Path("/assets/tf-input-fault-nav2.yaml")
    startup = yaml.safe_load(startup_path.read_text())
    fault_config = yaml.safe_load(fault_path.read_text())
    namespace, _, local_name = loc.rpartition("/")
    expected = deepcopy(startup)
    if expected[namespace][local_name]["ros__parameters"]["tf_broadcast"] is not True:
        raise ValueError("healthy startup must actually enable localization TF")
    expected[namespace][local_name]["ros__parameters"]["tf_broadcast"] = False
    if fault_config != expected:
        raise ValueError("fault source must differ only in localization tf_broadcast")
    selected = set(requirements["lifecycle_nodes"].values())
    limits = settings["navigation_fixture_policy"]["navigation_limits"]
    client = node.create_client(SetParameters, loc + "/set_parameters")
    events = []

    def parked():
        p = state.get("physics") or {}
        return (
            p.get("run_id") == policy["run_id"]
            and p.get("lease_enabled") is False
            and type(p.get("linear_speed_mps")) in (int, float)
            and abs(p["linear_speed_mps"]) <= limits["stop_speed_mps"]
            and type(p.get("angular_speed_rps")) in (int, float)
            and abs(p["angular_speed_rps"]) <= limits["stop_yaw_rate_rps"]
        )

    def set_tf(value):
        if not parked():
            raise ValueError("current independent parked, lease-closed observation required")
        if not client.wait_for_service(timeout_sec=3):
            raise ValueError("owned localization parameter service unavailable")
        request = SetParameters.Request(
            parameters=[
                Parameter(
                    name="tf_broadcast",
                    value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=value),
                )
            ]
        )
        start = datetime.now(UTC).isoformat()
        future = client.call_async(request)
        spin_until(future.done, 5)
        response = future.result()
        row = {
            "requested_at": start,
            "completed_at": datetime.now(UTC).isoformat(),
            "service": loc + "/set_parameters",
            "only_parameter": "tf_broadcast",
            "value": value,
            "configuration_source_sha256": hashlib.sha256(
                (startup_path if value else fault_path).read_bytes()
            ).hexdigest(),
            "configuration_application": "RUNTIME_OVERRIDE_AFTER_HEALTHY_CANONICAL_JOURNEY",
            "results": [{"successful": r.successful, "reason": r.reason} for r in response.results],
        }
        events.append(row)
        save("tf-input-parameter-events.json", events)
        if len(response.results) != 1 or not response.results[0].successful:
            raise ValueError("owned TF parameter update rejected")

    def capture(phase, expected_tf, broadcast):
        observations = []
        accepted = None

        def ready():
            nonlocal accepted
            m = state.get("model")
            if m is None or not parked():
                return False
            clock = datetime.now(UTC)
            reference = m.observations.get("prepared_collision_assets", {})
            if reference.get("schema_version") == "rosclaw.prepared_collision_asset_reference.v2":
                bind_prepared_collision_originals(
                    m.observations, Path("/assets/collision-asset-originals.json").read_bytes()
                )
            case = compile_engineering_case(
                m,
                urdf,
                requirements,
                source_ref="original-runtime-capture",
                now=clock,
                evidence_domain="OBSERVATION",
            )
            statuses = {c.id: c.status for c in case.constraints}
            params = m.observations.get("node_parameters", {}).get(loc, {})
            observed_states = {r.name: r.state for r in m.lifecycle if r.name in selected}
            observations.append(
                {
                    "at": clock.isoformat(),
                    "snapshot_hash": m.snapshot_hash,
                    "states": statuses,
                    "lifecycle": observed_states,
                }
            )
            if phase == "fault" and any(s != "ACTIVE" for s in observed_states.values()):
                raise ValueError("additional lifecycle fault observed; not a single-TF original")
            if (
                statuses["F-TF-001"] == expected_tf
                and params.get("tf_broadcast") is broadcast
                and all(
                    v == "SATISFIED"
                    for k, v in statuses.items()
                    if k not in {"F-TF-001", "F-SOURCE-001"}
                )
            ):
                accepted = (m.to_dict(), state["native"], dict(state["physics"]), clock.isoformat())
                return True
            return False

        try:
            spin_until(ready, 20)
        finally:
            save("tf-input-" + phase + "-acquisition-timeline.json", observations)
        model, native, physics, clock = accepted
        prefix = "tf-input-" + phase
        save(prefix + "-system.json", model)
        save(prefix + "-native.json", native)
        save(prefix + "-physics.json", physics)
        raw, references = original_ecm_tail(root)
        save(prefix + "-ecm-original-prefix-refs.json", references)
        with (root / (prefix + "-ecm.jsonl")).open("xb") as f:
            f.write(raw)
        save(
            prefix + "-ready.json",
            {
                "phase": phase,
                "clock": clock,
                "run_id": policy["run_id"],
                "snapshot_hash": model["snapshot_hash"],
                "ecm_sha256": hashlib.sha256(raw).hexdigest(),
                "model_calls": 0,
                "new_goals_in_capture_phase": 0,
                "scope": "OPERATOR_INPUT_PREPARATION_NOT_MODEL_REPAIR",
            },
        )
        spin_until(lambda: (root / (prefix + "-operator-review.json")).exists(), 20)
        review = json.loads((root / (prefix + "-operator-review.json")).read_text())
        if review.get("status") != "PASS_INDEPENDENT_CURRENT_SOURCE_AND_CASE":
            raise ValueError("external original admission review failed")

    changed = False
    try:
        capture("healthy", "SATISFIED", True)
        changed = True
        set_tf(False)
        capture("fault", "VIOLATED", False)
        set_tf(True)
        changed = False
        capture("restored", "SATISFIED", True)
        save(
            "tf-input-preparation-outcome.json",
            {
                "status": "PASS_OPERATOR_HEALTHY_SINGLE_TF_RESTORED_ORIGINALS",
                "run_id": policy["run_id"],
                "seed": policy["seed"],
                "model_calls": 0,
                "new_goals_in_capture_phase": 0,
                "parameter_changes": events,
            },
        )
    except Exception as error:
        save(
            "tf-input-preparation-outcome.json",
            {
                "status": "FAILED_OPERATOR_INPUT_PREPARATION_PRESERVE_ORIGINALS",
                "run_id": policy["run_id"],
                "seed": policy["seed"],
                "error": str(error),
                "model_calls": 0,
                "new_goals_in_capture_phase": 0,
            },
        )
        raise
    finally:
        if changed:
            try:
                set_tf(True)
            except Exception as error:
                save(
                    "tf-input-restore-failure.json",
                    {"error": str(error), "requires_owned_world_shutdown": True},
                )
        node.destroy_client(client)
