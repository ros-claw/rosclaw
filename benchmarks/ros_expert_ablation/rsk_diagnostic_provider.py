"""Owned sensor-only SDK probe contract and original-file observational review.

There is no Runtime, robot transport or motion path here. The operator supplies
an exclusive original manifest from its fixed isolated SDK. Booleans never stand
in for original Native captures, container/source pins or an independent review.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from benchmarks.ros_expert_ablation.rsk_episode_runner import Closed, OriginalRef, _read
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.engineering.probe_planner import (
    assess_diagnostic_followup,
    plan_diagnostic_probes,
)
from rosclaw.connectors.ros.intelligence import build_system_model
from rosclaw.connectors.ros.intelligence.operational_graph import compile_operational_graph
from rosclaw.connectors.ros.intelligence.system_model import RosSystemModel
from rosclaw.contracts.common import content_hash

SDK_IMAGE = "sha256:51ffb0e0aeb391f0efbef3013c7fc83586794c222b233e3dea688f3015ef1c6b"
RECONSTRUCTED_SDK_IMAGE = "sha256:8cc3fbf1eaa57058f0c1386c061253a55651106bd46b98fb046439f966c9019b"


class DiagnosticProbePolicyV1(Closed):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["rosclaw.rsk_diagnostic_probe_policy.v1"] = (
        "rosclaw.rsk_diagnostic_probe_policy.v1"
    )
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    original_run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    original_snapshot_hash: str = Field(pattern=r"^rossnap_[0-9a-f]{64}$")
    source_urdf_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    requirements_hash: str | None = Field(min_length=1, max_length=128)
    case_staging_hash: str | None = Field(min_length=1, max_length=128)
    execution_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    operations: tuple[Literal["ENDPOINT_QOS", "CLOCK_TF_WINDOW", "NATIVE_DIAGNOSTICS"], ...] = (
        Field(min_length=1, max_length=3)
    )
    minimum_wall_sec: int = Field(ge=8, le=40)
    observation_window_sec: Literal[5] = 5
    scope: Literal["SENSOR_ONLY_DDS_NO_PHYSICS_OR_BODY"] = "SENSOR_ONLY_DDS_NO_PHYSICS_OR_BODY"
    sdk_image: Literal[
        "sha256:51ffb0e0aeb391f0efbef3013c7fc83586794c222b233e3dea688f3015ef1c6b"
    ] = "sha256:51ffb0e0aeb391f0efbef3013c7fc83586794c222b233e3dea688f3015ef1c6b"

    @model_validator(mode="after")
    def complete_optional_navigation_context(self):
        values = [self.source_urdf_sha256, self.requirements_hash, self.case_staging_hash]
        if any(value is None for value in values) and not all(value is None for value in values):
            raise ValueError(
                "native-only context cannot contain partial invented navigation bindings"
            )
        return self


class DiagnosticProbePolicyV2(DiagnosticProbePolicyV1):
    """Explicit reconstructed SDK contract; V1 originals keep their original pin."""

    schema_version: Literal["rosclaw.rsk_diagnostic_probe_policy.v2"] = (
        "rosclaw.rsk_diagnostic_probe_policy.v2"
    )
    sdk_image: Literal[
        "sha256:8cc3fbf1eaa57058f0c1386c061253a55651106bd46b98fb046439f966c9019b"
    ] = RECONSTRUCTED_SDK_IMAGE


def diagnostic_probe_policy_type(schema_version):
    if schema_version == "rosclaw.rsk_diagnostic_probe_policy.v1":
        return DiagnosticProbePolicyV1
    if schema_version == "rosclaw.rsk_diagnostic_probe_policy.v2":
        return DiagnosticProbePolicyV2
    raise ValueError("unsupported diagnostic SDK policy version")


def parse_diagnostic_probe_policy(value):
    if isinstance(value, DiagnosticProbePolicyV1):
        value = value.model_dump()
    if type(value) is not dict:
        raise ValueError("closed diagnostic SDK policy required")
    return diagnostic_probe_policy_type(
        value.get("schema_version", "rosclaw.rsk_diagnostic_probe_policy.v1")
    ).model_validate(value)


@dataclass(frozen=True)
class NativeDiagnosticContext:
    """Source-only observation context, deliberately no Body or navigation case."""

    model: RosSystemModel
    operator_source: str
    original_run_id: str

    def checked_model(self):
        return RosSystemModel.from_dict(self.model.to_dict())


class DiagnosticProbeOriginalsV1(Closed):
    schema_version: Literal["rosclaw.rsk_diagnostic_probe_originals.v1"] = (
        "rosclaw.rsk_diagnostic_probe_originals.v1"
    )
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    original_run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    probe_run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    requested_operation: Literal["ENDPOINT_QOS", "CLOCK_TF_WINDOW", "NATIVE_DIAGNOSTICS"]
    started_at: str = Field(min_length=1, max_length=64)
    finished_at: str = Field(min_length=1, max_length=64)
    execution_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    native_before: OriginalRef
    native_after: OriginalRef
    owner_observations: OriginalRef
    container_before: OriginalRef
    container_after: OriginalRef


class DiagnosticProbeProvider:
    """Frozen host operator; model requests can select only registered probe IDs."""

    def __init__(self, policy, *, operator, worktree, evidence_root):
        from benchmarks.ros_expert_ablation.rsk_native_diagnostic_operator import (
            SensorOnlyDiagnosticOperator,
        )

        self.policy = parse_diagnostic_probe_policy(policy)
        if not isinstance(operator, SensorOnlyDiagnosticOperator) or operator.policy != self.policy:
            raise ValueError("closed sensor-only SDK operator with the same frozen policy required")
        self._policy_hash = content_hash("rskdiagnosticpolicy", self.policy.model_dump(mode="json"))
        self.operator = operator
        self.worktree = Path(worktree).resolve(strict=True)
        self.root = Path(evidence_root).resolve(strict=True)
        self.trace = []

    def check_context(self, base):
        if (
            Path(__file__).resolve().parents[2] != self.worktree
            or self.operator.worktree != self.worktree
            or self.operator.root != self.root
        ):
            raise ValueError(
                "diagnostic SDK implementation and operator must use the same pinned worktree/root"
            )
        if (
            content_hash("rskdiagnosticpolicy", self.policy.model_dump(mode="json"))
            != self._policy_hash
        ):
            raise ValueError("frozen diagnostic provider policy changed")
        if isinstance(base, NativeDiagnosticContext):
            snapshot = base.checked_model().snapshot_hash
            urdf_hash = requirements_hash = staging_hash = None
        else:
            base._bound(base.case.case_hash, read_only=True)
            snapshot = base.case.snapshot_hash
            urdf_hash = base.case.source_urdf_sha256
            requirements_hash = content_hash("navrequirements", base.requirements)
            staging_hash = content_hash("acistaging", base.tools.initial_staging)
        if (
            self.policy.source_revision != base.operator_source
            or self.policy.original_run_id != base.original_run_id
            or self.policy.original_snapshot_hash != snapshot
            or self.policy.source_urdf_sha256 != urdf_hash
            or self.policy.requirements_hash != requirements_hash
            or self.policy.case_staging_hash != staging_hash
        ):
            raise ValueError(
                "diagnostic provider differs from original source/run/snapshot/navigation context"
            )

        def git(*args):
            return subprocess.check_output(["git", *args], cwd=self.worktree, text=True).strip()

        if git("rev-parse", "HEAD") != self.policy.source_revision or git("status", "--porcelain"):
            raise ValueError("diagnostic SDK requires the exact clean frozen source")

    def run_and_review(self, plan, selected, base, *, original_graph, remaining_sec):
        if (
            type(remaining_sec) not in (int, float)
            or not math.isfinite(remaining_sec)
            or not self.policy.minimum_wall_sec <= remaining_sec <= 300
        ):
            raise ValueError(
                "sufficient real host wall budget required before diagnostic SDK dispatch"
            )
        self.check_context(base)
        started = time.monotonic()
        reference = None
        try:
            reference = self.operator.collect(plan, selected, remaining_sec=remaining_sec)
            reference = OriginalRef.model_validate(reference)
            result = self.review_originals(
                reference, plan, selected, base, original_graph=original_graph
            )
            self.check_context(base)
            if time.monotonic() - started > remaining_sec:
                raise ValueError("diagnostic provider completed outside the host wall budget")
        except Exception as error:
            result = {
                "status": "DIAGNOSTIC_SDK_FAILED_OR_INCOMPLETE_NO_RETRY",
                "probe_performed": "UNVERIFIED_OR_INCOMPLETE",
                "original_manifest_ref": None
                if reference is None
                else reference.model_dump(mode="json")
                if isinstance(reference, OriginalRef)
                else reference,
                "actual_new_Worlds": 0,
                "error_type": type(error).__name__,
                "error": str(error),
                "current_authority": "NONE",
                "current_case_L1": "NOT_EVALUATED",
                "causal_verification": "NOT_ESTABLISHED",
            }
        self.trace.append(
            {
                "experiment_hash": plan.experiment_hash,
                "probe_id": selected.request.id,
                "result": result,
            }
        )
        return result

    def review_originals(self, reference, plan, selected, base, *, original_graph):
        raw = _read(self.root, reference, limit=1_000_000)
        manifest = DiagnosticProbeOriginalsV1.model_validate_json(raw)
        if (
            manifest.source_revision != self.policy.source_revision
            or manifest.original_run_id != self.policy.original_run_id
            or manifest.probe_run_id == self.policy.original_run_id
            or manifest.requested_operation != selected.request.operation
            or manifest.execution_config_sha256 != self.policy.execution_config_sha256
        ):
            raise ValueError("probe originals differ from frozen source/run/config/request")
        begun, ended = (
            datetime.fromisoformat(v) for v in [manifest.started_at, manifest.finished_at]
        )
        if (
            begun.tzinfo is None
            or ended.tzinfo is None
            or not begun <= ended <= datetime.now(UTC)
            or (ended - begun).total_seconds() > 300
        ):
            raise ValueError("aware bounded original SDK observation window required")
        refs = [
            manifest.native_before,
            manifest.native_after,
            manifest.owner_observations,
            manifest.container_before,
            manifest.container_after,
        ]
        if len({ref.path for ref in refs}) != len(refs) or len({ref.sha256 for ref in refs}) != len(
            refs
        ):
            raise ValueError(
                "distinct source originals required for both captures and control roles"
            )
        docs = [json.loads(_read(self.root, ref, limit=8_000_000)) for ref in refs]
        before, after, owner, created, stopped = docs
        actual = json.loads(
            subprocess.check_output(["docker", "inspect", stopped["Id"]], text=True, timeout=15)
        )[0]
        if any(
            actual[key] != stopped[key] for key in ["Id", "Image", "State", "Config", "HostConfig"]
        ) or sorted(json.dumps(mount, sort_keys=True) for mount in actual["Mounts"]) != sorted(
            json.dumps(mount, sort_keys=True) for mount in stopped["Mounts"]
        ):
            raise ValueError("current stopped SDK container differs from the original inspection")
        if (
            created["Id"] != stopped["Id"]
            or stopped["State"]["Running"]
            or stopped["State"]["Pid"] != 0
            or stopped["State"]["ExitCode"] != 0
        ):
            raise ValueError("same owned SDK container must be stopped successfully")
        for container in [created, stopped]:
            host = container["HostConfig"]
            env = container["Config"]["Env"]
            if (
                container["Image"] != self.policy.sdk_image
                or host["NetworkMode"] != "none"
                or host["ReadonlyRootfs"] is not True
                or host["Privileged"]
                or host.get("Devices")
                or not any(
                    opt.startswith("no-new-privileges") for opt in host.get("SecurityOpt", [])
                )
                or container["Config"]["User"] != f"{os.getuid()}:{os.getgid()}"
                or "ALL" not in (host.get("CapDrop") or [])
                or "RSK_SOURCE_COMMIT=" + self.policy.source_revision not in env
                or "RSK_RUN_ID=" + manifest.probe_run_id not in env
                or not any(
                    m["Destination"] == "/repo"
                    and m["Source"] == str(self.worktree)
                    and not m["RW"]
                    for m in container["Mounts"]
                )
            ):
                raise ValueError("pinned isolated read-only no-device SDK context required")
            evidence = (self.root / manifest.native_before.path).parent.resolve(strict=True)
            if (self.root / manifest.native_after.path).parent.resolve(
                strict=True
            ) != evidence or not any(
                m["Destination"] == "/evidence" and m["Source"] == str(evidence) and m["RW"]
                for m in container["Mounts"]
            ):
                raise ValueError("actual Native originals must come from the owned evidence mount")
        if (
            owner.get("schema_version") != "rosclaw.rsk_sensor_observation_owner.v1"
            or owner.get("run_id") != manifest.probe_run_id
            or owner.get("source") != self.policy.source_revision
            or owner.get("execution_config_sha256") != self.policy.execution_config_sha256
            or owner.get("native_before_sha256") != manifest.native_before.sha256
            or owner.get("native_after_sha256") != manifest.native_after.sha256
            or owner.get("publisher_policy_changed_during_window") is not False
            or owner.get("robot_driver_processes") != []
            or owner.get("navigation_goals") != 0
            or owner.get("control_commands") != 0
            or owner.get("body_admission") != "NOT_RUN"
            or owner.get("physics") != "NOT_RUN"
        ):
            raise ValueError(
                "original sensor-only owner and unchanged policy/zero-command evidence required"
            )
        required_sources = {
            "integrations/ros_probe/ros2/probe.py",
            "benchmarks/ros_expert_ablation/rsk_native_diagnostic_fixture.py",
        }
        if set(owner.get("source_file_sha256", {})) != required_sources:
            raise ValueError("exact original observer/fixture source pins required")
        for name, h in owner["source_file_sha256"].items():
            path = self.worktree / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != h:
                raise ValueError("original SDK source file differs from current frozen code")
        steady = owner.get("observation_steady_sec")
        if type(steady) not in (int, float) or not math.isfinite(steady) or not 5 <= steady <= 10:
            raise ValueError("independent full steady observation window required")
        for key in ["consumer_before", "consumer_after"]:
            if type(owner.get(key)) is not dict or any(
                type(owner[key].get(field)) is not int or owner[key][field] < 0
                for field in ["count", "sent"]
            ):
                raise ValueError("original independent consumer counts required")
        if any(
            owner["consumer_after"][key] < owner["consumer_before"][key]
            for key in ["count", "sent"]
        ):
            raise ValueError("original sensor counters cannot move backwards")
        config = json.loads(_read(self.root, self.operator.execution_config, limit=64_000))
        policies = []
        for native in [before, after]:
            chosen = []
            for kind, node, expected_reliability in [
                ("publisher", config["namespace"] + "/emitter", config["offered_reliability"]),
                ("subscriber", config["namespace"] + "/receiver", config["requested_reliability"]),
            ]:
                matches = [
                    row
                    for row in native.get("qos", {}).get("endpoints", [])
                    if row.get("topic") == config["topic"]
                    and row.get("node") == node
                    and row.get("kind") == kind
                ]
                if len(matches) != 1 or matches[0].get("reliability") != expected_reliability:
                    raise ValueError(
                        "actual sensor endpoint policies differ from the frozen fixture source"
                    )
                chosen.append(
                    {
                        key: matches[0].get(key)
                        for key in [
                            "topic",
                            "node",
                            "kind",
                            "gid",
                            "reliability",
                            "durability",
                            "liveliness",
                            "deadline_ns",
                        ]
                    }
                )
            policies.append(chosen)
        if policies[0] != policies[1]:
            raise ValueError(
                "actual endpoint identity/policies changed during the read-only window"
            )
        capture_times = [datetime.fromisoformat(n["captured_at"]) for n in [before, after]]
        if not begun <= capture_times[0] < capture_times[1] <= ended:
            raise ValueError(
                "both actual Native clocks must be within the same original SDK window"
            )
        if not 5 <= (capture_times[1] - capture_times[0]).total_seconds() <= 10:
            raise ValueError(
                "full five-second measurement cannot be replaced with a shortened capture"
            )
        calls = owner.get("native_snapshot_call_durations_sec")
        if (
            type(calls) is not list
            or len(calls) != 2
            or any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= selected.request.max_duration_sec
                for value in calls
            )
        ):
            raise ValueError("actual Native reads must fit the original request duration")
        # The full ambient observation interval is distinct from each bounded
        # Native read. It stays inside the unmodified Core experiment budget.
        if (capture_times[1] - capture_times[0]).total_seconds() > plan.budget_sec:
            raise ValueError("ambient follow-up exceeds the original diagnostic experiment budget")
        for native in [before, after]:
            if (
                native["environment"].get("rmw") != "rmw_cyclonedds_cpp"
                or native.get("completeness", {}).get("graph") is not True
            ):
                raise ValueError("complete original fixed-RMW Native graph required")
            if native["graph"].get("actions"):
                raise ValueError("sensor-only source cannot contain action servers")
        graphs = []
        for native in [before, after]:
            model = build_system_model(
                RosGraphSnapshot.from_dict(
                    {
                        "ros_version": "ros2",
                        "distro": "jazzy",
                        "endpoint": "fixture://owned-read-only-diagnostic",
                        "captured_at": native["captured_at"],
                        **native["graph"],
                    }
                ),
                native=native,
            )
            context = (
                {}
                if isinstance(base, NativeDiagnosticContext)
                else {
                    "urdf_bytes": base.tools.files["robot.urdf"].encode(),
                    "requirements": base.requirements,
                }
            )
            graphs.append(
                compile_operational_graph(
                    model,
                    source_ref="owned-sensor-probe:" + manifest.probe_run_id,
                    now=datetime.fromisoformat(native["captured_at"]),
                    evidence_domain="SEALED_REPLAY",
                    **context,
                )
            )
        initial_graph = graphs[0]
        wanted = {
            (h.issue_id)
            for h in plan.hypotheses
            if h.id in {p.hypothesis_id for p in selected.predictions}
        }
        # Map only an invariant plus exact observed targets into the replica;
        # endpoint GIDs and graph hashes legitimately differ between containers.
        if (
            original_graph.snapshot_hash
            != (
                base.checked_model().snapshot_hash
                if isinstance(base, NativeDiagnosticContext)
                else base.case.snapshot_hash
            )
            or original_graph.graph_hash != plan.graph_hash
        ):
            raise ValueError("probe intent differs from the original case operational graph")
        scopes = {
            (f.invariant, tuple(f.targets)) for f in original_graph.findings if f.id in wanted
        }
        ids = [f.id for f in initial_graph.findings if (f.invariant, tuple(f.targets)) in scopes]
        if not ids:
            raise ValueError("original diagnostic target scope is absent from the owned probe")
        rebound = plan_diagnostic_probes(
            initial_graph, issue_ids=ids, budget_sec=plan.budget_sec, max_probes=plan.max_probes
        )
        assessment = assess_diagnostic_followup(
            rebound, graphs[0], graphs[1], now=graphs[1].reviewed_at
        )
        return {
            "status": "INDEPENDENT_ORIGINAL_PROBE_OBSERVATIONAL_REVIEW_NOT_ADMISSION",
            "probe_performed": True,
            "fresh_probe_performed": True,
            "source": self.policy.source_revision,
            "new_world_run": manifest.probe_run_id,
            "probe_scope": self.policy.scope,
            "actual_new_Worlds": 0,
            "original_capture_ref": manifest.native_after.model_dump(mode="json"),
            "original_manifest_ref": reference.model_dump(mode="json"),
            "before_graph_hash": graphs[0].graph_hash,
            "after_graph_hash": graphs[1].graph_hash,
            "original_intent_experiment_hash": plan.experiment_hash,
            "replica_rebound_experiment": rebound.model_dump(mode="json"),
            "assessment": assessment.model_dump(mode="json"),
            "original_case_rebound": False,
            "current_case_L1": "NOT_EVALUATED",
            "canonical_navigation_success": False,
            "causal_verification": "NOT_ESTABLISHED",
            "Memory_promotion": "NOT_GRANTED",
            "navigation_goals_sent": 0,
            "current_authority": "NONE",
        }
