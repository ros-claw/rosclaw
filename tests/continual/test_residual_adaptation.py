from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw.continual.residual_adaptation import (
    ParameterIsolationEvidence,
    ResidualAdaptationContract,
    load_residual_adaptation_contract,
    write_residual_adaptation_contract,
)
from tests.continual.helpers import digest


def _contract() -> ResidualAdaptationContract:
    return ResidualAdaptationContract(
        run_id="adapter.run-1",
        backend_contract_hash=digest("backend"),
        parent_artifact_hash=digest("parent"),
        body_hash=digest("body"),
        rehearsal_dataset_hash=digest("rehearsal"),
        acquisition_dataset_hash=digest("acquisition"),
        frozen_parameter_selectors=("policy.hidden_*",),
        trainable_parameter_selectors=("policy.adapter_*",),
        device_ids=(0, 1, 2, 3),
        maximum_world_steps=5_000_000,
        policy_learning_rate=3e-5,
        rehearsal_fraction=0.6,
        acquisition_fraction=0.4,
        maximum_residual_output_rms=0.05,
    )


def _evidence(contract: ResidualAdaptationContract) -> ParameterIsolationEvidence:
    return ParameterIsolationEvidence(
        adaptation_contract_hash=contract.contract_hash,
        parent_artifact_hash=contract.parent_artifact_hash,
        candidate_artifact_hash=digest("candidate"),
        frozen_base_hash_before=digest("frozen"),
        frozen_base_hash_after=digest("frozen"),
        matched_exam_hash=digest("exam"),
        examined_frozen_parameter_count=10,
        examined_trainable_parameter_count=10,
        candidate_world_steps=4_000_000,
        maximum_frozen_parameter_drift=0.0,
        residual_output_rms=0.02,
        retention_passed=True,
        acquisition_passed=True,
        critical_safety_regressions=0,
    )


def test_residual_contract_binds_rehearsal_and_trainable_scope() -> None:
    contract = _contract()

    assert contract.contract_hash.startswith("sha256:")
    assert _evidence(contract).passes(contract)
    with pytest.raises(ValueError, match="disjoint"):
        replace(contract, trainable_parameter_selectors=("policy.hidden_*",))


def test_parameter_isolation_rejects_forgetting_or_base_mutation() -> None:
    contract = _contract()
    evidence = _evidence(contract)

    assert not replace(evidence, retention_passed=False).passes(contract)
    assert not replace(evidence, frozen_base_hash_after=digest("mutated")).passes(contract)
    assert not replace(evidence, critical_safety_regressions=1).passes(contract)


def test_parameter_isolation_rejects_excessive_residual_churn() -> None:
    contract = _contract()
    evidence = replace(_evidence(contract), residual_output_rms=0.051)

    assert not evidence.passes(contract)


def test_parameter_isolation_rejects_training_beyond_sealed_budget() -> None:
    contract = _contract()
    evidence = replace(_evidence(contract), candidate_world_steps=5_000_001)

    assert not evidence.passes(contract)


def test_contract_writer_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    contract = _contract()
    output = tmp_path / "contract.json"

    payload = write_residual_adaptation_contract(contract, output)

    assert payload["contract_hash"] == contract.contract_hash
    assert load_residual_adaptation_contract(output) == contract
    assert output.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(ValueError, match="new JSON"):
        write_residual_adaptation_contract(contract, output)
