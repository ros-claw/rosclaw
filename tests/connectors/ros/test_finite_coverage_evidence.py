"""Software evidence/serialization fixtures only; no physics or Native qualification."""

import gzip
import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from rosclaw.connectors.ros.mission.finite_coverage_evidence import validate_archive
from rosclaw.connectors.ros.mission.public_coverage import PublicCoverageExecutor
from rosclaw.kernel import (
    ActionState,
    EvidenceDomain,
    EvidenceLevel,
    ExecutionMode,
    ExecutionReceipt,
)
from tests.connectors.ros.test_public_coverage_execution import SoftwarePort, setup_executor


class ProofPort(SoftwarePort):
    """Explicit synthetic native terminals for contract testing, not actual ROS."""

    def execute(self, capability, arguments, **kwargs):
        result = super().execute(capability, arguments, **kwargs)
        dispatch = dict(result.dispatch_result)
        if capability == "navigation.follow_path":
            dispatch.update(
                status=4, action="/follow_path", action_type="nav2_msgs/action/FollowPath"
            )
        return replace(
            result,
            dispatch_result=dispatch,
            evidence_domain=EvidenceDomain.FIXTURE
            if self.binding.execution_mode is ExecutionMode.FIXTURE
            else EvidenceDomain.SIMULATION,
        )


def fixture_run(directory, *, simulation_contract=False):
    _, old, port, action = setup_executor()
    binding = replace(
        old.binding,
        durable_task_evidence=True,
        execution_mode=ExecutionMode.SIMULATION if simulation_contract else ExecutionMode.FIXTURE,
    )
    port = ProofPort(binding)
    executor = PublicCoverageExecutor(
        binding=binding, port=port, ledger=old.ledger, evidence_store=directory
    )
    action = replace(action, execution_mode=binding.execution_mode)
    return executor, port, action


def receipt(action, result):
    return ExecutionReceipt(
        action_id=action.action_id,
        trace_id="fixture-proof-only",
        mode=action.execution_mode,
        body_id=action.body_id,
        body_snapshot_hash=action.body_snapshot_hash,
        capability_id=action.capability_id,
        final_state=result.final_state,
        evidence_level=result.evidence_level,
        evidence_domain=result.evidence_domain,
        verification_result=result.verification_result,
    )


@pytest.mark.parametrize("simulation_contract", [False, True])
def test_durable_archive_binds_finite_goal_original_sources_and_scoped_grade(
    tmp_path, simulation_contract
):
    executor, port, action = fixture_run(tmp_path, simulation_contract=simulation_contract)
    result = executor(action)
    assert result.final_state is ActionState.COMPLETED, result.errors
    assert result.evidence_level is EvidenceLevel.TASK_VERIFIED
    assert receipt(action, result).verified is simulation_contract
    assert receipt(action, result).valid_for_promotion is False
    assert result.verification_result["whole_house_success"] is None
    evidence = result.verification_result["task_evidence"]
    path = tmp_path / evidence["filename"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == evidence["sha256"]
    archive = json.loads(gzip.decompress(raw))
    summary = validate_archive(archive)
    assert summary["actual_brush_motion_m"] > 0.05
    assert archive["action"]["action_id"] == action.action_id
    assert len(archive["primitives"]) == len(port.calls)
    assert result.artifacts == [evidence["uri"]]
    assert path.stat().st_mode & 0o777 == 0o600


def test_no_store_retains_existing_driver_confirmed_classification():
    _, executor, _, action = setup_executor()
    result = executor(action)
    assert result.final_state is ActionState.COMPLETED
    assert result.evidence_level is EvidenceLevel.DRIVER_CONFIRMED
    assert result.artifacts == []
    assert result.verification_result["task_evidence"] is None


def test_missing_original_native_terminal_cannot_upgrade_generic_driver_ack(tmp_path):
    _, old, port, action = setup_executor()
    binding = replace(old.binding, durable_task_evidence=True)
    port.binding = binding
    executor = PublicCoverageExecutor(
        binding=binding, port=port, ledger=old.ledger, evidence_store=tmp_path
    )
    result = executor(action)
    assert result.final_state is ActionState.FAILED
    assert result.evidence_level is EvidenceLevel.REQUESTED
    assert executor.stopping.is_set() and port.stop_calls > 0
    assert result.verification_result["task_evidence"] is None


@pytest.mark.parametrize("failure", ["write", "readback"])
def test_archive_failure_is_a_task_failure_not_an_evidence_upgrade(tmp_path, monkeypatch, failure):
    executor, port, action = fixture_run(tmp_path)
    if failure == "write":

        def unavailable(*_args, **_kwargs):
            raise OSError("fixture persistence unavailable")

        monkeypatch.setattr(
            "rosclaw.connectors.ros.mission.public_coverage.persist_archive", unavailable
        )
    else:
        from pathlib import Path

        original = Path.read_bytes
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda path: (
                original(path) + b"corrupt" if str(path).endswith(".json.gz") else original(path)
            ),
        )
    result = executor(action)
    assert result.final_state is ActionState.FAILED
    assert result.evidence_level is EvidenceLevel.REQUESTED
    assert result.verification_result["task_evidence"] is None
    assert executor.stopping.is_set() and port.stop_calls > 0


def test_duplicate_action_archive_is_never_overwritten(tmp_path):
    executor, _, action = fixture_run(tmp_path)
    first = executor(action)
    assert first.final_state is ActionState.COMPLETED
    archive = tmp_path / first.verification_result["task_evidence"]["filename"]
    original = archive.read_bytes()
    second = executor(action)
    assert second.final_state is ActionState.FAILED
    assert archive.read_bytes() == original


@pytest.fixture
def completed_archive(tmp_path):
    executor, _, action = fixture_run(tmp_path)
    result = executor(action)
    assert result.final_state is ActionState.COMPLETED, result.errors
    return json.loads(
        gzip.decompress(
            (tmp_path / result.verification_result["task_evidence"]["filename"]).read_bytes()
        )
    )


@pytest.mark.parametrize(
    "fault",
    [
        "whole_house",
        "real",
        "body",
        "publisher",
        "receive_age",
        "header_age",
        "other_parent",
        "duplicate_primitive",
        "ack_only",
        "moving",
        "brush_on",
        "unknown_goal",
        "no_work",
        "no_trace",
        "replayed_sample",
        "outside_action",
    ],
)
def test_modified_or_incomplete_archive_is_rejected(completed_archive, fault):
    p = deepcopy(completed_archive)
    if fault == "whole_house":
        p["whole_house_success"] = True
    elif fault == "real":
        p["execution_mode"] = "REAL"
    elif fault == "body":
        p["action"]["body_snapshot_hash"] = "other"
    elif fault == "publisher":
        p["final"]["sources"][0]["publisher_gid"] = "2" * 48
    elif fault == "receive_age":
        p["final"]["sources"][0]["received_monotonic_s"] -= 10
    elif fault == "header_age":
        fact = next(f for f in p["final"]["sources"] if f["role"] == "odom")
        fact["header_native_s"] -= 10
    elif fault == "other_parent":
        p["primitives"][0]["parent_action_id"] = "other"
    elif fault == "duplicate_primitive":
        p["primitives"].append(p["primitives"][0])
    elif fault == "ack_only":
        step = next(s for s in p["primitives"] if s["capability_id"] == "navigation.follow_path")
        step["dispatch"]["status"] = None
    elif fault == "moving":
        p["final"]["linear_speed"] = 0.1
    elif fault == "brush_on":
        p["final"]["cleaner"] = True
    elif fault == "unknown_goal":
        p["final"]["native_motion_settled"] = False
    elif fault == "no_work":
        p["completed_work"] = 0
    elif fault == "no_trace":
        p["samples"] = []
    elif fault == "replayed_sample":
        p["samples"].insert(1, p["samples"][0])
    elif fault == "outside_action":
        p["samples"][0]["captured_monotonic_s"] = p["initial"]["captured_monotonic_s"] - 1
    with pytest.raises(ValueError):
        validate_archive(p)


@pytest.mark.parametrize("store", [None, "fixture-store"])
def test_durable_contract_requires_both_compiled_body_and_trusted_store(tmp_path, store):
    _, old, port, _ = setup_executor()
    binding = replace(old.binding, durable_task_evidence=store is None)
    port.binding = binding
    with pytest.raises(ValueError, match="must agree"):
        PublicCoverageExecutor(
            binding=binding,
            port=port,
            ledger=old.ledger,
            evidence_store=None if store is None else tmp_path / store,
        )


def test_native_channel_accepts_scoped_task_evidence_and_still_rejects_driver_confirmation(
    tmp_path,
):
    from rosclaw.agentd.action_channel import DaemonActionChannel

    executor, _, action = fixture_run(tmp_path, simulation_contract=True)
    result = executor(action)
    assert result.final_state is ActionState.COMPLETED, result.errors
    canonical = receipt(action, result).to_dict()
    channel = DaemonActionChannel(
        None, actor_id="fixture", body_id=action.body_id, body_hash=action.body_snapshot_hash
    )
    outcome = channel._verify_outcome(
        action.action_id, {"state": "FINISHED"}, {"receipt": canonical}, action
    )
    assert outcome.verified is True
    assert canonical["verification_result"]["scope"] == "finite_observed_coverage_slice_only"
    assert (
        canonical["valid_for_promotion"] is False
        and canonical["usable_for_real_execution"] is False
    )
    legacy = {**canonical, "evidence_level": "DRIVER_CONFIRMED", "verified": False}
    assert (
        channel._verify_outcome(
            action.action_id, {"state": "FINISHED"}, {"receipt": legacy}, action
        ).verified
        is False
    )


@pytest.mark.parametrize(
    "fault",
    [
        "energy",
        "docked",
        "undeclared",
        "binding_mode",
        "source_set",
        "bool_goal",
        "status_contradiction",
        "endpoint",
        "start_missing_source",
    ],
)
def test_archive_cannot_weaken_original_work_guards(completed_archive, fault):
    p = deepcopy(completed_archive)
    if fault == "energy":
        p["initial"]["soc"] = p["binding"]["minimum_work_soc"] - 0.001
    elif fault == "docked":
        p["initial"]["docked"] = True
    elif fault == "undeclared":
        p["binding"]["durable_task_evidence"] = False
    elif fault == "binding_mode":
        p["binding"]["execution_mode"] = "SIMULATION"
    elif fault == "source_set":
        p["binding"]["sources"] = [s for s in p["binding"]["sources"] if s["role"] != "depth"]
    elif fault == "bool_goal":
        p["completed_work"] = True
    elif fault == "status_contradiction":
        next(s for s in p["primitives"] if s["capability_id"] == "navigation.follow_path")[
            "dispatch"
        ]["status"] = 5
    elif fault == "endpoint":
        next(s for s in p["primitives"] if s["capability_id"] == "navigation.follow_path")["after"][
            "map_xy"
        ] = [100.0, 100.0]
    elif fault == "start_missing_source":
        p["initial"]["required_roles"].remove("depth")
        p["initial"]["sources"] = [s for s in p["initial"]["sources"] if s["role"] != "depth"]
    with pytest.raises(ValueError):
        validate_archive(p)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_durable_binding_flag_is_a_boolean_contract(value):
    _, executor, _, _ = setup_executor()
    with pytest.raises(ValueError, match="boolean durable"):
        replace(executor.binding, durable_task_evidence=value)


class StreamingProofPort(ProofPort):
    def bind_observation_service(self, service):
        self.service = service

    def _sample(self, *, work=True):
        super()._sample(work=work)
        self.maximum_buffer = max(getattr(self, "maximum_buffer", 0), len(self.buffer))
        service = getattr(self, "service", None)
        if service is not None:
            service()


def test_blocking_primitive_services_original_evidence_only_while_parent_owned(tmp_path):
    _, old, _, action = setup_executor()
    binding = replace(old.binding, durable_task_evidence=True)
    port = StreamingProofPort(binding)
    executor = PublicCoverageExecutor(
        binding=binding, port=port, ledger=old.ledger, evidence_store=tmp_path
    )
    initial_count = len(port.buffer)
    port.service()
    assert len(port.buffer) == initial_count  # No owner, no unowned credit drain.
    result = executor(action)
    assert result.final_state is ActionState.COMPLETED, result.errors
    assert port.maximum_buffer <= 2
    payload = json.loads(
        gzip.decompress(
            (tmp_path / result.verification_result["task_evidence"]["filename"]).read_bytes()
        )
    )
    assert len(payload["samples"]) > 20 and payload["completed_strokes"] >= 1
    port._sample(work=False)
    assert len(port.buffer) == 1  # Closed parent cannot consume a new sample.
