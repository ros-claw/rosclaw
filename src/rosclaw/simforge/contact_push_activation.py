"""Champion activation, ordinary use, Canary regression, and automatic rollback."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.simforge.champion_registry import (
    ActiveChampionEpisodeReceipt,
    CanaryReceipt,
    ChampionActivationReceipt,
    SimulationChampionRegistry,
)
from rosclaw.simforge.contact_push_learning import ContactPushCandidate
from rosclaw.simforge.models import Partition
from rosclaw.simforge.phase3_gate import ContactPushPromotionRecord
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.contact_push_v3 import (
    CONTACT_PUSH_BODY_HASH,
    CONTACT_PUSH_TASK_ID,
    ContactPushPhysics,
    ContactPushScenario,
    generate_contact_push_scenarios,
)


@dataclass(frozen=True)
class ContactPushActivationResult:
    champion_activation: ChampionActivationReceipt
    ordinary_episode: ActiveChampionEpisodeReceipt
    regression_activation: ChampionActivationReceipt
    canary: CanaryReceipt
    rollback_receipt: dict[str, Any]
    rollback_retry: ActiveChampionEpisodeReceipt
    canary_scenario_commitments: tuple[str, ...]
    final_active_candidate_hash: str
    ledger_verified: bool
    wrong_body_slot_empty: bool
    schema_version: str = "rosclaw.contact_push_activation_result.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "champion_activation": self.champion_activation.to_dict(),
            "ordinary_episode": self.ordinary_episode.to_dict(),
            "regression_activation": self.regression_activation.to_dict(),
            "canary": self.canary.to_dict(),
            "rollback_receipt": self.rollback_receipt,
            "rollback_retry": self.rollback_retry.to_dict(),
            "canary_scenario_commitments": list(self.canary_scenario_commitments),
            "final_active_candidate_hash": self.final_active_candidate_hash,
            "ledger_verified": self.ledger_verified,
            "wrong_body_slot_empty": self.wrong_body_slot_empty,
        }


def activate_canary_and_rollback(
    *,
    champion: ContactPushCandidate,
    champion_promotion: ContactPushPromotionRecord,
    regression_candidate: ContactPushCandidate,
    regression_promotion: ContactPushPromotionRecord,
    registry_root: Path,
    output_root: Path,
    source_checkout: Path,
    root_seed: int,
) -> ContactPushActivationResult:
    """Exercise D8/D9 and roll an independently promoted regression back."""

    output = _external_root(output_root, source_checkout)
    output.mkdir(parents=True, exist_ok=False)
    registry = SimulationChampionRegistry(
        root=registry_root,
        source_checkout=source_checkout,
    )
    physics = ContactPushPhysics(trace_stride=50)
    champion_activation = registry.activate(
        candidate=champion,
        promotion=champion_promotion,
        body_snapshot_hash=CONTACT_PUSH_BODY_HASH,
    )
    ordinary_scenario = _find_successful_scenario(
        candidate=champion,
        partition=Partition.STRESS,
        root_seed=root_seed + 301,
        purpose="ordinary-active-episode",
    )
    ordinary = registry.run_active_episode(
        task_id=CONTACT_PUSH_TASK_ID,
        body_snapshot_hash=CONTACT_PUSH_BODY_HASH,
        scenario=ordinary_scenario,
        physics=physics,
    )
    if ordinary.candidate_hash != champion.candidate_hash or not ordinary.success:
        raise RuntimeError("ordinary task did not use the activated Champion successfully")

    regression_activation = registry.activate(
        candidate=regression_candidate,
        promotion=regression_promotion,
        body_snapshot_hash=CONTACT_PUSH_BODY_HASH,
    )
    canary_scenarios = _find_regression_scenarios(
        champion=champion,
        regression=regression_candidate,
        root_seed=root_seed + 302,
    )
    canary = registry.run_canary(
        task_id=CONTACT_PUSH_TASK_ID,
        body_snapshot_hash=CONTACT_PUSH_BODY_HASH,
        scenarios=canary_scenarios,
        physics=physics,
        minimum_success_rate=0.95,
    )
    if canary.passed or not canary.frozen or canary.rollback_receipt_hash is None:
        raise RuntimeError("Canary failed to freeze and roll back the regressed Champion")
    rollback_receipt = registry.receipt(canary.rollback_receipt_hash)
    final_status = registry.status(
        task_id=CONTACT_PUSH_TASK_ID,
        body_snapshot_hash=CONTACT_PUSH_BODY_HASH,
    )
    final_hash = str(final_status["active_candidate_hash"])
    if final_hash != champion.candidate_hash:
        raise RuntimeError("automatic Canary rollback did not restore the prior Champion")
    rollback_retry = registry.run_active_episode(
        task_id=CONTACT_PUSH_TASK_ID,
        body_snapshot_hash=CONTACT_PUSH_BODY_HASH,
        scenario=canary_scenarios[0],
        physics=physics,
    )
    if rollback_retry.candidate_hash != champion.candidate_hash or not rollback_retry.success:
        raise RuntimeError("the restored Champion failed the same Canary scenario")
    wrong_body_hash = "sha256:" + "0" * 64
    result = ContactPushActivationResult(
        champion_activation=champion_activation,
        ordinary_episode=ordinary,
        regression_activation=regression_activation,
        canary=canary,
        rollback_receipt=rollback_receipt,
        rollback_retry=rollback_retry,
        canary_scenario_commitments=tuple(
            scenario.scenario_commitment for scenario in canary_scenarios
        ),
        final_active_candidate_hash=final_hash,
        ledger_verified=registry.verify_ledger(),
        wrong_body_slot_empty=(
            registry.resolve(
                task_id=CONTACT_PUSH_TASK_ID,
                body_snapshot_hash=wrong_body_hash,
            )
            is None
        ),
    )
    if not result.ledger_verified or not result.wrong_body_slot_empty:
        raise RuntimeError("Registry ledger or body scope verification failed")
    _atomic_json(output / "activation-canary-rollback.json", result.to_dict())
    return result


def _find_successful_scenario(
    *,
    candidate: ContactPushCandidate,
    partition: Partition,
    root_seed: int,
    purpose: str,
) -> ContactPushScenario:
    scenarios = generate_contact_push_scenarios(
        ledger=SeedLedger(
            task_id=CONTACT_PUSH_TASK_ID,
            secret=_secret(root_seed, purpose),
        ),
        partition=partition,
        count=128,
        root_seed=root_seed,
    )
    physics = ContactPushPhysics(trace_stride=100)
    for scenario in scenarios:
        if physics.run(scenario, candidate.policy_for(scenario)).success:
            return scenario
    raise RuntimeError("no successful ordinary Champion scenario was found")


def _find_regression_scenarios(
    *,
    champion: ContactPushCandidate,
    regression: ContactPushCandidate,
    root_seed: int,
) -> tuple[ContactPushScenario, ...]:
    physics = ContactPushPhysics(trace_stride=100)
    selected: list[ContactPushScenario] = []
    for batch in range(4):
        scenarios = generate_contact_push_scenarios(
            ledger=SeedLedger(
                task_id=CONTACT_PUSH_TASK_ID,
                secret=_secret(root_seed, f"canary-counterexample-{batch}"),
            ),
            partition=Partition.COUNTEREXAMPLE_REGRESSION,
            count=256,
            root_seed=root_seed + batch,
        )
        for scenario in scenarios:
            prior = physics.run(scenario, champion.policy_for(scenario))
            current = physics.run(scenario, regression.policy_for(scenario))
            if prior.success and not current.success:
                selected.append(scenario)
                if len(selected) == 10:
                    return tuple(selected)
    if not selected:
        raise RuntimeError("no real Canary regression separated the two Champions")
    return tuple(selected)


def _external_root(output_root: Path, source_checkout: Path) -> Path:
    root = output_root.expanduser().resolve()
    checkout = source_checkout.resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("ContactPush activation artifacts must stay outside the checkout")
    return root


def _secret(root_seed: int, purpose: str) -> bytes:
    return hashlib.sha256(f"rosclaw.contact_push.phase3\0{root_seed}\0{purpose}".encode()).digest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "ContactPushActivationResult",
    "activate_canary_and_rollback",
]
