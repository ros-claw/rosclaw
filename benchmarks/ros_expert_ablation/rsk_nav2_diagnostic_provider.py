"""Closed Nav2 read-only replica provider; observation grants no motion authority."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from benchmarks.ros_expert_ablation.rsk_diagnostic_provider import (
    DiagnosticProbePolicyV1,
    DiagnosticProbeProvider,
)
from benchmarks.ros_expert_ablation.rsk_nav2_diagnostic_operator import (
    Nav2SourceOnlyDiagnosticOperator,
)
from benchmarks.ros_expert_ablation.rsk_nav2_original_review import (
    _check_sink,
    _original,
    review_stopped_owned_nav2_window,
)
from integrations.ros_probe.acceptance.simulation_atomic_read import read_bounded_regular
from rosclaw.connectors.ros.engineering.probe_planner import plan_diagnostic_probes
from rosclaw.contracts.common import content_hash


class Nav2DiagnosticProbePolicyV1(DiagnosticProbePolicyV1):
    schema_version: Literal["rosclaw.rsk_nav2_diagnostic_probe_policy.v1"] = (
        "rosclaw.rsk_nav2_diagnostic_probe_policy.v1"
    )
    scope: Literal["OWNED_SOURCE_ONLY_NAV2_PHYSICS_NO_BODY"] = (
        "OWNED_SOURCE_ONLY_NAV2_PHYSICS_NO_BODY"
    )
    minimum_wall_sec: int = Field(ge=60, le=180)
    observation_role: Literal[
        "readonly_preinitialization_observation", "readonly_localized_observation"
    ]

    @model_validator(mode="after")
    def original_navigation_context_required(self):
        if any(
            value is None
            for value in [self.source_urdf_sha256, self.requirements_hash, self.case_staging_hash]
        ):
            raise ValueError("complete original Nav2 context required; no invented Body admission")
        return self


class Nav2DiagnosticProbeProvider(DiagnosticProbeProvider):
    """C/E-only subtype with a closed Nav2 owner and explicit one-World cost."""

    def __init__(self, policy, *, operator, worktree, evidence_root):
        self.policy = Nav2DiagnosticProbePolicyV1.model_validate(policy)
        if type(operator) is not Nav2SourceOnlyDiagnosticOperator:
            raise ValueError("closed Nav2 source-only owner required; no callbacks")
        self.operator = operator
        self.worktree = Path(worktree).resolve(strict=True)
        self.root = Path(evidence_root).resolve(strict=True)
        if (
            self.operator.worktree != self.worktree
            or self.operator.root != self.root
            or self.operator.recipe_reference.sha256 != self.policy.execution_config_sha256
            or self.operator.recipe.source_revision != self.policy.source_revision
            or self.operator.recipe.source_urdf_sha256 != self.policy.source_urdf_sha256
            or self.operator.recipe.role != self.policy.observation_role
        ):
            raise ValueError("Nav2 operator/recipe differs from the registered policy")
        self._policy_hash = content_hash(
            "rsknav2diagnosticpolicy", self.policy.model_dump(mode="json")
        )
        self.trace = []

    def check_context(self, base):
        if Path(__file__).resolve().parents[2] != self.worktree:
            raise ValueError("Nav2 provider implementation differs from frozen worktree")
        self.operator.check_ready()
        base._bound(base.case.case_hash, read_only=True)
        if (
            content_hash("rsknav2diagnosticpolicy", self.policy.model_dump(mode="json"))
            != self._policy_hash
            or self.policy.source_revision != base.operator_source
            or self.policy.original_run_id != base.original_run_id
            or self.policy.original_snapshot_hash != base.case.snapshot_hash
            or self.policy.source_urdf_sha256 != base.case.source_urdf_sha256
            or self.policy.requirements_hash != content_hash("navrequirements", base.requirements)
            or self.policy.case_staging_hash
            != content_hash("acistaging", base.tools.initial_staging)
        ):
            raise ValueError("Nav2 diagnostic policy differs from original source/run/context")
        # Validate semantic configuration before the shared ACI probe slot is
        # consumed, including a missing or modified operator recipe/inventory.
        import yaml

        staging = {
            name: yaml.safe_load((self.operator.assets / name).read_bytes())
            for name in ["nav2.yaml", "bringup.json"]
        }
        if content_hash("acistaging", staging) != self.policy.case_staging_hash:
            raise ValueError("owned Nav2 replica would change the original diagnostic staging")

    def run_and_review(self, plan, selected, base, *, original_graph, remaining_sec):
        from benchmarks.ros_expert_ablation.rsk_operational_aci import original_operational_graph

        self.check_context(base)
        if (
            type(remaining_sec) not in (int, float)
            or not math.isfinite(remaining_sec)
            or not self.policy.minimum_wall_sec <= remaining_sec <= 300
            or selected not in plan.probes
            or selected.request.operation not in self.policy.operations
            or selected.request.provider != "EXISTING_NATIVE_READ_ONLY_PROBE"
            or original_graph != original_operational_graph(base)
            or plan
            != plan_diagnostic_probes(
                original_graph,
                issue_ids=sorted({h.issue_id for h in plan.hypotheses}),
                budget_sec=plan.budget_sec,
                max_probes=plan.max_probes,
            )
        ):
            raise ValueError("original registered diagnostic intent and real host budget required")
        begun = time.monotonic()
        world = None
        try:
            world = self.operator.collect(plan, selected, remaining_sec=remaining_sec)
            result = self.review_originals(
                world, plan, selected, base, original_graph=original_graph
            )
            self.check_context(base)
            if time.monotonic() - begun > remaining_sec:
                raise ValueError("Nav2 diagnostic exceeded the original host deadline")
        except Exception as error:
            result = {
                "status": "NAV2_DIAGNOSTIC_FAILED_OR_INCOMPLETE_NO_RETRY",
                "probe_performed": "UNVERIFIED_OR_INCOMPLETE",
                "error_type": type(error).__name__,
                "error": str(error),
                "original_World": None
                if self.operator.last_world is None
                else str(self.operator.last_world),
                "reserved_probe_world_slots": self.operator.calls,
                "created_owned_containers": self.operator.started_containers,
                "actual_new_physics_Worlds": "UNVERIFIED_PRESERVE_ORIGINALS",
                "actual_new_physics_Worlds_upper_bound": 1,
                "navigation_goals": 0,
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

    def review_originals(self, world, plan, selected, base, *, original_graph):
        self.check_context(base)
        world = Path(world)
        if world.parent != self.operator.output_parent or not world.name.startswith(
            "RSK-nav2-observation-"
        ):
            raise ValueError("original Nav2 World must belong to this closed owner")
        intent, intent_ref = _original(self.root, world / "original-dispatch-intent.json")
        if (
            intent["source"] != self.policy.source_revision
            or intent["recipe"] != self.operator.recipe_reference.model_dump(mode="json")
            or intent["plan"] != plan.model_dump(mode="json")
            or intent["selected_probe_id"] != selected.request.id
        ):
            raise ValueError(
                "Nav2 original pre-dispatch intent differs from registered plan/recipe"
            )
        window, window_ref = _original(
            self.root, world / "read-only-window/window-original-manifest.json"
        )
        durations = window.get("native_original_file_read_durations_sec")
        if (
            type(durations) is not list
            or len(durations) != 2
            or any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= selected.request.max_duration_sec
                for value in durations
            )
            or window["native_capture_span_sec"] > plan.budget_sec
            or window["host_window_elapsed_sec"] > plan.budget_sec
        ):
            raise ValueError(
                "bounded individual Native reads and ambient experiment budget required"
            )
        result = review_stopped_owned_nav2_window(
            world,
            root=self.root,
            worktree=self.worktree,
            capture_source=self.policy.source_revision,
            original_run_id=self.policy.original_run_id,
            original_graph=original_graph,
            plan=plan,
            selected=selected,
            source_urdf_sha256=self.policy.source_urdf_sha256,
            requirements_hash=self.policy.requirements_hash,
            staging_hash=self.policy.case_staging_hash,
        )
        if intent["run_id"] != result["probe_run_id"]:
            raise ValueError("original Nav2 dispatch identity differs from captured replica")
        physics, physics_ref = _original(
            self.root, world / "execution/engineering-original-physics.json"
        )
        if (
            physics.get("evidence_domain") != "GAZEBO_PHYSICS"
            or physics.get("source_only_observer") is not True
            or physics.get("observation_complete") is not True
            or any(
                type(physics.get(key)) is not int or physics[key] != 0
                for key in [
                    "source_received_control_messages",
                    "source_published_velocity_messages",
                    "lease_updates",
                ]
            )
        ):
            raise ValueError("original source-only physics and zero-command witness required")
        _check_sink(
            physics["independent_control_sink_observation"]["status"], result["probe_run_id"]
        )
        ecm = read_bounded_regular(world / "execution/ecm-original.jsonl", 64_000_000)
        frames = [json.loads(line) for line in ecm.splitlines() if line.strip()]
        if not frames or any(
            len(frame.get("worlds", [])) != 1
            or frame.get("observer_run_label") != result["probe_run_id"]
            for frame in frames
        ):
            raise ValueError("original ECM must contain exactly one same-run physics World")
        if (
            len({(frame["worlds"][0]["entity"], frame["worlds"][0]["name"]) for frame in frames})
            != 1
        ):
            raise ValueError("original ECM physics World identity changed")
        result.update(
            status="REVIEWED_OWNED_NAV2_DIAGNOSTIC_OBSERVATIONS_NOT_CAUSAL_OR_ADMISSION",
            probe_performed=True,
            original_intent_ref=intent_ref,
            original_window_ref=window_ref,
            actual_new_physics_Worlds=1,
            navigation_goals=0,
            current_case_L1="NOT_EVALUATED",
            provider_scope=self.policy.scope,
            original_physics_ref=physics_ref,
            ecm_sha256=hashlib.sha256(ecm).hexdigest(),
            measurement_backend="OWNED_NATIVE_CAPTURE_ORIGINAL_FILE_READS",
            read_duration_scope="NATIVE_ORIGINAL_FILE_READ_NOT_ROS_SNAPSHOT_CALL",
        )
        return result
