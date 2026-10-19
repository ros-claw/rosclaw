"""Synthetic policy/entry/budget contracts; no SDK, model or physics execution."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from benchmarks.ros_expert_ablation import rsk_nav2_diagnostic_operator as owner
from benchmarks.ros_expert_ablation.rsk_episode_runner import OriginalRef
from benchmarks.ros_expert_ablation.rsk_nav2_diagnostic_provider import (
    Nav2DiagnosticProbePolicyV1,
    Nav2DiagnosticProbeProvider,
)
from benchmarks.ros_expert_ablation.rsk_operational_aci import OperationalRskACI
from benchmarks.ros_expert_ablation.rsk_registered_inputs import _registered_diagnostic_context
from rosclaw.contracts.common import content_hash
from tests.benchmarks.test_rsk_nav2_diagnostic_operator import recipe
from tests.benchmarks.test_rsk_operational_aci import facade, plan, req
from tests.benchmarks.test_rsk_pilot_protocol import evidence, protocol


def provider_fixture(tmp_path, monkeypatch, aci):
    data = recipe()
    data["source_revision"] = "b" * 40
    data["source_urdf_sha256"] = aci.base.case.source_urdf_sha256
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(data))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(owner.Nav2SourceOnlyDiagnosticOperator, "check_ready", lambda self: None)
    worktree = Path(owner.__file__).resolve().parents[2]
    op = owner.Nav2SourceOnlyDiagnosticOperator(
        recipe=OriginalRef(path=path.name, sha256=digest),
        worktree=worktree,
        evidence_root=tmp_path,
        output_parent=tmp_path,
    )
    op.assets = tmp_path / "assets"
    op.assets.mkdir()
    for name, value in aci.base.tools.initial_staging.items():
        (op.assets / name).write_text(yaml.safe_dump(value))
    aci.base.operator_source, aci.base.original_run_id = "b" * 40, "a" * 32
    policy = Nav2DiagnosticProbePolicyV1(
        source_revision="b" * 40,
        original_run_id="a" * 32,
        original_snapshot_hash=aci.base.case.snapshot_hash,
        source_urdf_sha256=aci.base.case.source_urdf_sha256,
        requirements_hash=content_hash("navrequirements", aci.base.requirements),
        case_staging_hash=content_hash("acistaging", aci.base.tools.initial_staging),
        execution_config_sha256=digest,
        operations=("ENDPOINT_QOS", "CLOCK_TF_WINDOW"),
        minimum_wall_sec=60,
        observation_role=data["role"],
    )
    provider = Nav2DiagnosticProbeProvider(
        policy, operator=op, worktree=worktree, evidence_root=tmp_path
    )
    p = protocol().model_copy(update={"harness_source": "b" * 40})
    return provider, {
        "pilot_protocol": p,
        "pilot_evidence": evidence(tmp_path, p),
        "evidence_root": tmp_path,
    }


def test_original_entry_is_required_and_b_cannot_bind_nav2_owner(tmp_path, monkeypatch):
    original = facade()
    provider, options = provider_fixture(tmp_path, monkeypatch, original)
    with pytest.raises(ValueError, match="pilot entry"):
        OperationalRskACI(original.delegate, diagnostic_probe_provider=provider)
    original.delegate.arm = "B"
    with pytest.raises(ValueError, match="C/E"):
        OperationalRskACI(original.delegate, diagnostic_probe_provider=provider, **options)
    assert provider.operator.calls == original.base.probe_count == 0


def test_nav2_policy_requires_actual_original_navigation_bindings(tmp_path, monkeypatch):
    provider, _ = provider_fixture(tmp_path, monkeypatch, facade())
    values = provider.policy.model_dump()
    values.update(source_urdf_sha256=None, requirements_hash=None, case_staging_hash=None)
    with pytest.raises(ValueError, match="complete original Nav2 context"):
        Nav2DiagnosticProbePolicyV1.model_validate(values)
    with pytest.raises(ValueError, match="closed Nav2"):
        Nav2DiagnosticProbeProvider(
            provider.policy,
            operator=lambda: None,
            worktree=provider.worktree,
            evidence_root=tmp_path,
        )


@pytest.mark.parametrize("arm", ["A", "B", "C", "D", "E"])
def test_common_nav2_policy_does_not_grant_baseline_expert_operations(tmp_path, monkeypatch, arm):
    from tests.benchmarks.test_rsk_baselines import generic

    original = facade()
    provider, _ = provider_fixture(tmp_path, monkeypatch, original)
    if arm == "A":
        aci = generic(original.base)
        aci.base.operator_source, aci.base.original_run_id = "b" * 40, "a" * 32
    elif arm == "D":
        from benchmarks.ros_expert_ablation.rsk_physical_aci import RskPhysicalACI

        aci = RskPhysicalACI(original.base, arm="B")
    elif arm == "E":
        from benchmarks.ros_expert_ablation.rsk_physical_aci import RskPhysicalACI
        from tests.benchmarks.test_cross_body_memory_trace import query_fixture

        query, *_ = query_fixture(
            tmp_path, target_urdf_sha256=original.base.case.source_urdf_sha256
        )
        aci = OperationalRskACI(RskPhysicalACI(original.base, arm="E", memory_query=query))
    else:
        aci = facade("B" if arm == "B" else "C")
        aci.base = aci.delegate.base = original.base
        aci._graph_json, aci._graph_hash = original._graph_json, original._graph_hash
    registered = original.registered_diagnostic_inputs()
    registered["provider_policy"] = provider.policy.model_dump(mode="json")
    _registered_diagnostic_context(aci, registered)
    mapping = aci.request_schema()["discriminator"]["mapping"]
    assert ("ros_engineer.plan_diagnostic_experiment" in mapping) is (arm in {"C", "E"})
    assert original.base.probe_count == provider.operator.calls == 0


@pytest.mark.parametrize(
    "field",
    [
        "source_revision",
        "original_run_id",
        "original_snapshot_hash",
        "source_urdf_sha256",
        "requirements_hash",
        "case_staging_hash",
    ],
)
def test_rehashed_wrong_registered_nav2_context_is_rejected(tmp_path, monkeypatch, field):
    aci = facade()
    provider, _ = provider_fixture(tmp_path, monkeypatch, aci)
    registered = aci.registered_diagnostic_inputs()
    registered["provider_policy"] = provider.policy.model_dump(mode="json")
    value = registered["provider_policy"][field]
    registered["provider_policy"][field] = ("c" if value[0] != "c" else "d") + value[1:]
    with pytest.raises(ValueError):
        _registered_diagnostic_context(aci, registered)


def test_60sec_host_budget_and_single_shared_slot_are_explicit(tmp_path, monkeypatch):
    original = facade()
    provider, options = provider_fixture(tmp_path, monkeypatch, original)
    aci = OperationalRskACI(original.delegate, diagnostic_probe_provider=provider, **options)
    provider.check_context(aci.base)
    assert (
        aci.context_projection()["operational_diagnostics"]["owned_probe_physics_World_cost"] == 1
    )
    calls = []
    monkeypatch.setattr(
        provider,
        "run_and_review",
        lambda *args, **kwargs: calls.append(True) or {"status": "SYNTHETIC_NO_SDK"},
    )
    experiment = plan(aci)["experiment"]
    selected = next(p for p in experiment["probes"] if p["request"]["operation"] == "ENDPOINT_QOS")
    request = req(
        aci,
        "run_diagnostic_probe",
        experiment_hash=experiment["experiment_hash"],
        probe_id=selected["request"]["id"],
    )
    assert aci.call(request, remaining_sec=59)["status"].startswith("REJECTED")
    assert not calls and aci.base.probe_count == 0
    assert aci.call(request, remaining_sec=60)["status"] == "SYNTHETIC_NO_SDK"
    assert len(calls) == aci.base.probe_count == 1
    assert aci.call(request, remaining_sec=60)["status"].startswith("REJECTED") and len(calls) == 1


def test_changed_staging_is_rejected_before_slot_or_owner_dispatch(tmp_path, monkeypatch):
    original = facade()
    provider, options = provider_fixture(tmp_path, monkeypatch, original)
    aci = OperationalRskACI(original.delegate, diagnostic_probe_provider=provider, **options)
    experiment = plan(aci)["experiment"]
    selected = next(p for p in experiment["probes"] if p["request"]["operation"] == "ENDPOINT_QOS")
    (provider.operator.assets / "bringup.json").write_text('{"namespace":"/changed"}')
    result = aci.call(
        req(
            aci,
            "run_diagnostic_probe",
            experiment_hash=experiment["experiment_hash"],
            probe_id=selected["request"]["id"],
        ),
        remaining_sec=90,
    )
    assert result["status"].startswith("REJECTED")
    assert aci.base.probe_count == provider.operator.calls == 0


def test_corrupt_policy_shape_is_rejected_without_attribute_error(tmp_path, monkeypatch):
    original = facade()
    registered = copy.deepcopy(original.registered_diagnostic_inputs())
    registered["provider_policy"] = "not-a-policy"
    with pytest.raises(ValueError, match="closed registered"):
        _registered_diagnostic_context(original, registered)


@pytest.mark.parametrize(
    "mutation", ["root", "worktree", "recipe-sha", "source", "urdf", "role", "callback"]
)
def test_swapped_operator_binding_is_rejected_before_shared_slot(tmp_path, monkeypatch, mutation):
    original = facade()
    provider, options = provider_fixture(tmp_path, monkeypatch, original)
    aci = OperationalRskACI(original.delegate, diagnostic_probe_provider=provider, **options)
    experiment = plan(aci)["experiment"]
    selected = next(p for p in experiment["probes"] if p["request"]["operation"] == "ENDPOINT_QOS")
    op = provider.operator
    if mutation in {"root", "worktree"}:
        setattr(op, mutation, tmp_path / "changed")
    elif mutation == "recipe-sha":
        op.recipe_reference = op.recipe_reference.model_copy(update={"sha256": "f" * 64})
    elif mutation == "callback":
        provider.operator = lambda: None
    else:
        key, value = {
            "source": ("source_revision", "c" * 40),
            "urdf": ("source_urdf_sha256", "c" * 64),
            "role": ("role", "readonly_preinitialization_observation"),
        }[mutation]
        op.recipe = op.recipe.model_copy(update={key: value})
    result = aci.call(
        req(
            aci,
            "run_diagnostic_probe",
            experiment_hash=experiment["experiment_hash"],
            probe_id=selected["request"]["id"],
        ),
        remaining_sec=90,
    )
    assert result["status"].startswith("REJECTED")
    assert aci.base.probe_count == op.calls == 0
