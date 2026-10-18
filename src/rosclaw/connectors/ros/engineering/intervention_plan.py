"""Pure source-bound, single-parameter SIM staging proposals, never execution."""

import copy
import json
import math
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from rosclaw.contracts.common import content_hash

from ..intelligence.graph_contracts import GraphRecord, RosOperationalGraphV1
from .contracts import EvidenceRefV1


class ParameterInterventionV1(GraphRecord):
    node: str = Field(min_length=1, max_length=512)
    parameter: str = Field(min_length=1, max_length=256)
    operation: Literal["SET", "RESET_COMPONENT_DEFAULT"] = "SET"
    proposed_value: Any = None
    expected_after_value: Any
    hypothesis: str = Field(min_length=1, max_length=2048)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    max_wall_sec: int = Field(default=900, ge=30, le=3600, strict=True)
    max_worlds: Literal[3] = 3
    max_after_goals: Literal[2] = 2
    max_total_goals: Literal[2] = 2

    @model_validator(mode="after")
    def bounded_semantics(self):
        if not re.fullmatch(r"/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*", self.node):
            raise ValueError("explicit fully-qualified ROS node required")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", self.parameter):
            raise ValueError("concrete ROS parameter identity required")
        value = self.expected_after_value
        if self.operation == "SET":
            if _encoded(value) != _encoded(self.proposed_value):
                raise ValueError("SET prediction must match the proposed value")
        elif (
            not self.parameter.endswith(".sensor_frame")
            or self.proposed_value is not None
            or value != ""
        ):
            raise ValueError("only an explicit empty-frame default hypothesis can be reset")
        if self.parameter == "transform_tolerance":
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 5:
                raise ValueError("bounded transform window in seconds required")
        elif self.parameter.endswith((".sensor_frame", ".plugin")):
            if type(value) is not str or len(value) > 512 or any(ord(c) < 32 for c in value):
                raise ValueError("bounded literal frame or registered plugin candidate required")
            if self.parameter.endswith(".plugin") and (not value or self.operation != "SET"):
                raise ValueError("explicit nonempty plugin class candidate required")
        else:
            raise ValueError(
                "unsupported staging semantics; Body/control/limits require their existing compilers"
            )
        return self


PROOFS = (
    "independent_review_of_the_single_staging_difference",
    "new_owned_isolated_SIM_World_source_and_Body_binding",
    "fresh_full_L1_READY_before_any_After_action",
    "two_canonical_MCP_rosclawd_COMPLETED_receipts",
    "independent_stop_zero_collision_and_source_continuity",
    "unchanged_500ms_source_250ms_proof_and_Body_limits",
    "independent_rollback_World_reobserves_original_condition",
    "original_consumer_effect_and_explicit_negative_hypotheses",
    "installed_component_default_or_class_and_current_frame_checks",
)


class RosInterventionPlanV1(GraphRecord):
    schema_version: Literal["rosclaw.ros_intervention_plan.v1"] = "rosclaw.ros_intervention_plan.v1"
    plan_hash: str
    graph_hash: str
    snapshot_hash: str
    source_ref: str
    proposed_at: datetime
    evidence_domain: Literal["SEALED_REPLAY", "OBSERVATION"]
    max_age_ms: float = Field(gt=0, le=500)
    request: ParameterInterventionV1
    status: Literal[
        "CONDITIONAL_SIM_STAGING_PROPOSAL", "NEEDS_READ_ONLY_PROBE", "ABSTAIN", "NO_CHANGE"
    ]
    reasons: list[str] = Field(max_length=16)
    before_staging_hash: str
    after_staging_hash: str | None = None
    rollback_staging_hash: str | None = None
    changed_path: list[str] = Field(default_factory=list, max_length=32)
    original_readback: EvidenceRefV1 | None = None
    staged_documents: dict[str, dict] | None = None
    rollback_documents: dict[str, dict] | None = None
    intervention_class: Literal["SIM_STAGING_ONLY"] = "SIM_STAGING_ONLY"
    classification: Literal["INFERRED_CANDIDATE"] = "INFERRED_CANDIDATE"
    execution_status: Literal["PROPOSED_NOT_EXECUTED"] = "PROPOSED_NOT_EXECUTED"
    execution_authority: Literal["NONE"] = "NONE"
    current_l1: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"
    source_authentication: Literal["NOT_ESTABLISHED_BY_HASH"] = "NOT_ESTABLISHED_BY_HASH"
    operator_review: Literal["REQUIRED_BEFORE_NEW_WORLD"] = "REQUIRED_BEFORE_NEW_WORLD"
    proof_obligations: tuple[str, ...] = PROOFS
    on_failure: Literal["STOP_AND_PRESERVE_ORIGINALS"] = "STOP_AND_PRESERVE_ORIGINALS"

    @model_validator(mode="after")
    def linked(self):
        if self.proposed_at.tzinfo is None or self.proof_obligations != PROOFS:
            raise ValueError("aware clock and unchanged physical proof obligations required")
        if self.status == "CONDITIONAL_SIM_STAGING_PROPOSAL":
            if not self.changed_path or self.original_readback is None:
                raise ValueError(
                    "original readback and concrete single staging difference required"
                )
            if self.staged_documents is None or self.rollback_documents is None:
                raise ValueError("concrete After and rollback documents required")
            _staging(self.staged_documents)
            _staging(self.rollback_documents)
            differences = _differences(self.rollback_documents, self.staged_documents)
            if differences != [self.changed_path] or self.changed_path[0] != "nav2.yaml":
                raise ValueError("exactly the declared single parameter difference required")
            tokens = self.changed_path[1:]
            if tokens.count("ros__parameters") != 1:
                raise ValueError("concrete ROS parameters node path required")
            boundary = tokens.index("ros__parameters")
            node = "/" + "/".join(p.strip("/") for p in tokens[:boundary])
            if (
                node != self.request.node
                or ".".join(tokens[boundary + 1 :]) != self.request.parameter
            ):
                raise ValueError("changed parameter path differs from the requested target")
            original = self.rollback_documents
            for key in self.changed_path:
                original = original[key]
            expected_pointer = (
                "/observations/node_parameters/"
                + self.request.node.replace("~", "~0").replace("/", "~1")
                + "/"
                + self.request.parameter
            )
            if (
                self.original_readback.snapshot_hash != self.snapshot_hash
                or self.original_readback.source_ref != self.source_ref
                or self.original_readback.pointer != expected_pointer
                or self.original_readback.classification != "MEASURED"
                or self.original_readback.value_hash != content_hash("opvalue", original)
            ):
                raise ValueError("original parameter provenance differs from the plan baseline")
            stamp = datetime.fromisoformat(self.original_readback.observed_at)
            if (
                stamp.tzinfo is None
                or not 0 <= (self.proposed_at - stamp).total_seconds() <= self.max_age_ms / 1000
            ):
                raise ValueError("original parameter readback cannot be renewed by a new plan")
            parent = self.staged_documents
            for key in self.changed_path[:-1]:
                parent = parent[key]
            if self.request.operation == "SET":
                if _encoded(parent[self.changed_path[-1]]) != _encoded(self.request.proposed_value):
                    raise ValueError("staged parameter differs from candidate")
            elif self.changed_path[-1] in parent:
                raise ValueError("component-default reset must actually remove the override")
            if self.after_staging_hash != content_hash("acistaging", self.staged_documents):
                raise ValueError("After staging digest mismatch")
            if (
                self.rollback_staging_hash != self.before_staging_hash
                or self.rollback_staging_hash != content_hash("acistaging", self.rollback_documents)
            ):
                raise ValueError("rollback must reproduce original staging")
        elif (
            any(
                x is not None
                for x in (
                    self.staged_documents,
                    self.rollback_documents,
                    self.after_staging_hash,
                    self.rollback_staging_hash,
                )
            )
            or self.changed_path
        ):
            raise ValueError("abstention cannot carry a partial staging change")
        return self

    @classmethod
    def from_dict(cls, data):
        obj = cls.model_validate(data)
        document = obj.model_dump(mode="json")
        digest = document.pop("plan_hash")
        if not digest or digest != content_hash("rosintervention", document):
            raise ValueError("intervention plan hash mismatch")
        return obj


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _staging(value):
    if (
        type(value) is not dict
        or set(value) != {"nav2.yaml", "bringup.json"}
        or any(type(v) is not dict for v in value.values())
    ):
        raise ValueError("two concrete original staging objects required")
    if len(_encoded(value).encode()) > 5_000_000:
        raise ValueError("bounded original staging objects required")


def _differences(before, after, path=()):
    if type(before) is dict and type(after) is dict:
        changes = []
        for key in sorted(set(before) | set(after)):
            if key not in before or key not in after:
                changes.append(list(path + (key,)))
            else:
                changes.extend(_differences(before[key], after[key], path + (key,)))
        return changes
    return [] if _encoded(before) == _encoded(after) else [list(path)]


def _node_parameters(mapping, path=()):
    for key, value in mapping.items():
        if type(key) is not str or "*" in key:
            raise ValueError("concrete unambiguous YAML node selectors required")
        if key == "ros__parameters":
            if type(value) is not dict:
                raise ValueError("ROS parameters object required")
            yield "/" + "/".join(p.strip("/") for p in path), value, path + (key,)
        elif type(value) is dict:
            yield from _node_parameters(value, path + (key,))


def _parameter_paths(mapping, name, path=(), prefix=""):
    for key, value in mapping.items():
        if type(key) is not str:
            raise ValueError("literal parameter keys required")
        full = prefix + key
        if full == name:
            yield path + (key,), value
        if type(value) is dict:
            yield from _parameter_paths(value, name, path + (key,), full + ".")


def compile_intervention_plan(graph, model, staging, request):
    """Compile an untrusted candidate; native readback is integrity, not authority."""
    graph = RosOperationalGraphV1.from_dict(graph.model_dump(mode="json"))
    request = ParameterInterventionV1.model_validate(request)
    _staging(staging)
    if (
        model.snapshot_hash != graph.snapshot_hash
        or model.compute_snapshot_hash() != graph.snapshot_hash
    ):
        raise ValueError("unchanged original model and graph required")
    base = copy.deepcopy(staging)
    values = {
        "plan_hash": "",
        "graph_hash": graph.graph_hash,
        "snapshot_hash": graph.snapshot_hash,
        "source_ref": graph.source_ref,
        "proposed_at": graph.reviewed_at,
        "evidence_domain": graph.evidence_domain,
        "max_age_ms": graph.max_age_ms,
        "request": request,
        "status": "ABSTAIN",
        "reasons": [],
        "before_staging_hash": content_hash("acistaging", base),
    }

    def finish(status, reason):
        values.update(status=status, reasons=[reason])
        obj = RosInterventionPlanV1(**values)
        payload = obj.model_dump(mode="json")
        payload.pop("plan_hash")
        return RosInterventionPlanV1.from_dict(
            {"plan_hash": content_hash("rosintervention", payload), **payload}
        )

    if graph.source_status == "MISMATCH" or graph.snapshot_freshness != "FRESH":
        return finish("ABSTAIN", "original source mismatch or expired capture; no partial patch")
    observed = model.observations.get("node_parameters", {}).get(request.node, {})
    stamp = model.observations.get("parameter_captured_at", {}).get(request.node)
    try:
        at = datetime.fromisoformat(stamp)
        current = (
            at.tzinfo is not None
            and 0 <= (graph.reviewed_at - at).total_seconds() * 1000 <= graph.max_age_ms
        )
    except (TypeError, ValueError):
        current = False
    if not current or type(observed) is not dict or request.parameter not in observed:
        return finish(
            "NEEDS_READ_ONLY_PROBE", "actual current per-node original parameter readback required"
        )
    nodes = [row for row in _node_parameters(base["nav2.yaml"]) if row[0] == request.node]
    if len(nodes) != 1:
        return finish("ABSTAIN", "one concrete YAML node binding required")
    _, params, node_path = nodes[0]
    matches = list(_parameter_paths(params, request.parameter))
    if len(matches) != 1:
        return finish(
            "ABSTAIN", "one explicit original parameter path required; defaults are not invented"
        )
    path, before_value = matches[0]
    if _encoded(before_value) != _encoded(observed[request.parameter]):
        return finish("ABSTAIN", "original staging value differs from original native readback")
    if _encoded(before_value) == _encoded(request.expected_after_value):
        return finish("NO_CHANGE", "candidate has no semantic parameter change")
    after = copy.deepcopy(base)
    changed = ("nav2.yaml",) + node_path + path
    parent = after
    for key in changed[:-1]:
        parent = parent[key]
    if request.operation == "SET":
        parent[changed[-1]] = copy.deepcopy(request.proposed_value)
    else:
        del parent[changed[-1]]
    _staging(after)
    escaped_node = request.node.replace("~", "~0").replace("/", "~1")
    values.update(
        after_staging_hash=content_hash("acistaging", after),
        rollback_staging_hash=content_hash("acistaging", base),
        changed_path=list(changed),
        staged_documents=after,
        rollback_documents=base,
        original_readback=EvidenceRefV1(
            source_ref=graph.source_ref,
            snapshot_hash=graph.snapshot_hash,
            pointer="/observations/node_parameters/" + escaped_node + "/" + request.parameter,
            observed_at=at.isoformat(),
            value_hash=content_hash("opvalue", observed[request.parameter]),
            classification="MEASURED",
        ),
    )
    return finish(
        "CONDITIONAL_SIM_STAGING_PROPOSAL",
        "single reviewed parameter hypothesis only; component/default/frame facts and all new World proofs remain pending",
    )
