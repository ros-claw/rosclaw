"""Body-scoped simulation Champion activation, Canary, and rollback."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from filelock import FileLock

from rosclaw.simforge.contact_push_learning import ContactPushCandidate
from rosclaw.simforge.phase3_gate import ContactPushPromotionRecord, Phase3Decision
from rosclaw.simforge.tasks.contact_push_v3 import (
    CONTACT_PUSH_TASK_ID,
    ContactPushPhysics,
    ContactPushScenario,
    ContactPushStatus,
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class ChampionActivationReceipt:
    skill_id: str
    old_candidate_hash: str | None
    new_candidate_hash: str
    version: int
    body_snapshot_hash: str
    evaluation_hash: str
    dataset_snapshot_hash: str
    runtime_generation: int
    scope: str
    activated_at_unix: float
    previous_receipt_hash: str | None
    schema_version: str = "rosclaw.champion_activation_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["receipt_hash"] = _hash_json(value)
        return value

    @property
    def receipt_hash(self) -> str:
        return str(self.to_dict()["receipt_hash"])


@dataclass(frozen=True)
class CanaryReceipt:
    skill_id: str
    candidate_hash: str
    body_snapshot_hash: str
    evaluated_episodes: int
    success_rate: float
    critical_regressions: int
    passed: bool
    frozen: bool
    rollback_receipt_hash: str | None
    evidence_refs: tuple[str, ...]
    schema_version: str = "rosclaw.champion_canary_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_refs"] = list(self.evidence_refs)
        value["receipt_hash"] = _hash_json(value)
        return value

    @property
    def receipt_hash(self) -> str:
        return str(self.to_dict()["receipt_hash"])


@dataclass(frozen=True)
class RollbackReceipt:
    skill_id: str
    from_candidate_hash: str
    to_candidate_hash: str
    body_snapshot_hash: str
    reason: str
    runtime_generation: int
    rolled_back_at_unix: float
    previous_receipt_hash: str | None
    schema_version: str = "rosclaw.champion_rollback_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["receipt_hash"] = _hash_json(value)
        return value

    @property
    def receipt_hash(self) -> str:
        return str(self.to_dict()["receipt_hash"])


@dataclass(frozen=True)
class ActiveChampionEpisodeReceipt:
    skill_id: str
    candidate_hash: str
    body_snapshot_hash: str
    runtime_generation: int
    scenario_commitment: str
    policy_hash: str
    status: str
    success: bool
    physics_executed: bool
    strict_replay: bool
    active_slot_receipt_hash: str | None
    schema_version: str = "rosclaw.active_champion_episode_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["receipt_hash"] = _hash_json(value)
        return value

    @property
    def receipt_hash(self) -> str:
        return str(self.to_dict()["receipt_hash"])


class SimulationChampionRegistry:
    """Persistent active slot; simulation artifacts can never authorize REAL."""

    def __init__(self, *, root: Path, source_checkout: Path) -> None:
        self.root = root.expanduser().resolve()
        self.source_checkout = source_checkout.resolve()
        if self.root == self.source_checkout or self.source_checkout in self.root.parents:
            raise ValueError("Champion Registry state must be outside the source checkout")
        self.root.mkdir(parents=True, exist_ok=True)
        self._state_path = self.root / "registry.json"
        self._lock = FileLock(str(self.root / "registry.lock"))
        if not self._state_path.exists():
            self._write_state(
                {
                    "schema_version": "rosclaw.simulation_champion_registry.v1",
                    "runtime_generation": 0,
                    "active": {},
                    "history": {},
                    "champions": {},
                    "last_receipt_hash": None,
                }
            )

    def activate(
        self,
        *,
        candidate: ContactPushCandidate,
        promotion: ContactPushPromotionRecord,
        body_snapshot_hash: str,
        scope: str = "simulation",
    ) -> ChampionActivationReceipt:
        if scope != "simulation":
            raise PermissionError("SimForge Champion activation is simulation-only")
        if promotion.decision is not Phase3Decision.SIM_CHAMPION or not promotion.passed:
            raise PermissionError("Champion activation requires a passing Phase 3 promotion")
        if promotion.candidate_hash != candidate.candidate_hash:
            raise ValueError("promotion/candidate hash mismatch")
        if promotion.body_snapshot_hash != body_snapshot_hash:
            raise ValueError("promotion/body snapshot mismatch")
        if not _SHA256_RE.fullmatch(body_snapshot_hash):
            raise ValueError("body snapshot hash must be a sha256 identifier")
        if candidate.dataset_snapshot_hash != promotion.dataset_snapshot_hash:
            raise ValueError("candidate/promotion dataset snapshot mismatch")
        with self._lock:
            state = self._read_state()
            slot = _slot(CONTACT_PUSH_TASK_ID, body_snapshot_hash)
            old_hash = state["active"].get(slot)
            history = list(state["history"].get(slot, []))
            if old_hash == candidate.candidate_hash:
                raise ValueError("candidate is already active in this body slot")
            if old_hash is not None:
                history.append(old_hash)
            version = len(history) + 1
            generation = int(state["runtime_generation"]) + 1
            receipt = ChampionActivationReceipt(
                skill_id=CONTACT_PUSH_TASK_ID,
                old_candidate_hash=old_hash,
                new_candidate_hash=candidate.candidate_hash,
                version=version,
                body_snapshot_hash=body_snapshot_hash,
                evaluation_hash=promotion.promotion_hash,
                dataset_snapshot_hash=promotion.dataset_snapshot_hash,
                runtime_generation=generation,
                scope=scope,
                activated_at_unix=time.time(),
                previous_receipt_hash=state.get("last_receipt_hash"),
            )
            state["runtime_generation"] = generation
            state["active"][slot] = candidate.candidate_hash
            state["history"][slot] = history
            state["champions"][candidate.candidate_hash] = {
                "candidate": candidate.to_dict(),
                "promotion": promotion.to_dict(),
                "status": "active",
                "version": version,
                "body_snapshot_hash": body_snapshot_hash,
            }
            if old_hash in state["champions"]:
                state["champions"][old_hash]["status"] = "superseded"
            state["last_receipt_hash"] = receipt.receipt_hash
            self._write_receipt("activation", receipt.receipt_hash, receipt.to_dict())
            self._write_state(state)
            return receipt

    def status(self, *, task_id: str, body_snapshot_hash: str) -> dict[str, Any]:
        state = self._read_state()
        candidate_hash = state["active"].get(_slot(task_id, body_snapshot_hash))
        champion = state["champions"].get(candidate_hash) if candidate_hash else None
        return {
            "task_id": task_id,
            "body_snapshot_hash": body_snapshot_hash,
            "active_candidate_hash": candidate_hash,
            "runtime_generation": state["runtime_generation"],
            "champion": champion,
        }

    def resolve(
        self,
        *,
        task_id: str,
        body_snapshot_hash: str,
    ) -> ContactPushCandidate | None:
        state = self._read_state()
        candidate_hash = state["active"].get(_slot(task_id, body_snapshot_hash))
        if candidate_hash is None:
            return None
        record = state["champions"].get(candidate_hash)
        if not isinstance(record, dict) or record.get("status") != "active":
            raise RuntimeError("active Champion slot is corrupt or frozen")
        if record.get("body_snapshot_hash") != body_snapshot_hash:
            raise RuntimeError("active Champion body scope mismatch")
        candidate = ContactPushCandidate.from_dict(record["candidate"])
        if candidate.candidate_hash != candidate_hash:
            raise RuntimeError("active Champion artifact hash mismatch")
        return candidate

    def rollback(
        self,
        *,
        task_id: str,
        body_snapshot_hash: str,
        reason: str,
    ) -> RollbackReceipt:
        if not reason:
            raise ValueError("rollback reason is required")
        with self._lock:
            state = self._read_state()
            slot = _slot(task_id, body_snapshot_hash)
            current = state["active"].get(slot)
            history = list(state["history"].get(slot, []))
            if current is None or not history:
                raise ValueError("no previous Champion is available for rollback")
            target = history.pop()
            if target not in state["champions"]:
                raise RuntimeError("rollback target artifact is missing")
            generation = int(state["runtime_generation"]) + 1
            receipt = RollbackReceipt(
                skill_id=task_id,
                from_candidate_hash=current,
                to_candidate_hash=target,
                body_snapshot_hash=body_snapshot_hash,
                reason=reason,
                runtime_generation=generation,
                rolled_back_at_unix=time.time(),
                previous_receipt_hash=state.get("last_receipt_hash"),
            )
            state["runtime_generation"] = generation
            state["active"][slot] = target
            state["history"][slot] = history
            state["champions"][current]["status"] = "frozen"
            state["champions"][target]["status"] = "active"
            state["last_receipt_hash"] = receipt.receipt_hash
            self._write_receipt("rollback", receipt.receipt_hash, receipt.to_dict())
            self._write_state(state)
            return receipt

    def run_active_episode(
        self,
        *,
        task_id: str,
        body_snapshot_hash: str,
        scenario: ContactPushScenario,
        physics: ContactPushPhysics,
    ) -> ActiveChampionEpisodeReceipt:
        """Resolve the active slot and execute one ordinary simulation task."""

        state = self._read_state()
        candidate = self.resolve(
            task_id=task_id,
            body_snapshot_hash=body_snapshot_hash,
        )
        if candidate is None:
            raise ValueError("ordinary episode requires an active Champion")
        policy = candidate.policy_for(scenario)
        result = physics.run(scenario, policy)
        replay = physics.run(scenario, policy)
        strict_replay = _hash_json(result.summary_dict()) == _hash_json(replay.summary_dict())
        receipt = ActiveChampionEpisodeReceipt(
            skill_id=task_id,
            candidate_hash=candidate.candidate_hash,
            body_snapshot_hash=body_snapshot_hash,
            runtime_generation=int(state["runtime_generation"]),
            scenario_commitment=scenario.scenario_commitment,
            policy_hash=policy.policy_hash,
            status=result.status.value,
            success=result.success,
            physics_executed=result.physics_executed,
            strict_replay=strict_replay,
            active_slot_receipt_hash=state.get("last_receipt_hash"),
        )
        self._write_receipt(
            "ordinary_episode",
            receipt.receipt_hash,
            receipt.to_dict(),
        )
        return receipt

    def run_canary(
        self,
        *,
        task_id: str,
        body_snapshot_hash: str,
        scenarios: tuple[ContactPushScenario, ...],
        physics: ContactPushPhysics,
        minimum_success_rate: float = 0.95,
    ) -> CanaryReceipt:
        if (
            not scenarios
            or not math.isfinite(minimum_success_rate)
            or not 0 <= minimum_success_rate <= 1
        ):
            raise ValueError("canary requires scenarios and a success threshold in [0, 1]")
        candidate = self.resolve(task_id=task_id, body_snapshot_hash=body_snapshot_hash)
        if candidate is None:
            raise ValueError("no active Champion exists for the Canary")
        results = [physics.run(scenario, candidate.policy_for(scenario)) for scenario in scenarios]
        critical = sum(
            result.status
            in {
                ContactPushStatus.FORCE_LIMIT,
                ContactPushStatus.NON_FINITE,
            }
            for result in results
        )
        success_rate = sum(result.success for result in results) / len(results)
        passed = critical == 0 and success_rate >= minimum_success_rate
        rollback_hash = None
        frozen = False
        if not passed:
            rollback = self.rollback(
                task_id=task_id,
                body_snapshot_hash=body_snapshot_hash,
                reason=(f"canary_regression:critical={critical},success_rate={success_rate:.6f}"),
            )
            rollback_hash = rollback.receipt_hash
            frozen = True
        evidence_refs = tuple(
            "scenario://" + scenario.scenario_commitment.removeprefix("sha256:")
            for scenario in scenarios
        )
        receipt = CanaryReceipt(
            skill_id=task_id,
            candidate_hash=candidate.candidate_hash,
            body_snapshot_hash=body_snapshot_hash,
            evaluated_episodes=len(results),
            success_rate=success_rate,
            critical_regressions=critical,
            passed=passed,
            frozen=frozen,
            rollback_receipt_hash=rollback_hash,
            evidence_refs=evidence_refs,
        )
        self._write_receipt("canary", receipt.receipt_hash, receipt.to_dict())
        return receipt

    def verify_ledger(self) -> bool:
        state = self._read_state()
        last_hash = state.get("last_receipt_hash")
        if last_hash is None:
            return True
        receipt_files = sorted((self.root / "receipts").glob("*.json"))
        by_hash = {}
        for path in receipt_files:
            value = json.loads(path.read_text(encoding="utf-8"))
            receipt_hash = value.pop("receipt_hash", None)
            if receipt_hash != _hash_json(value):
                return False
            by_hash[receipt_hash] = value
        current = last_hash
        visited = set()
        while current is not None:
            if current in visited or current not in by_hash:
                return False
            visited.add(current)
            current = by_hash[current].get("previous_receipt_hash")
        return True

    def receipt(self, receipt_hash: str) -> dict[str, Any]:
        if not _SHA256_RE.fullmatch(receipt_hash):
            raise ValueError("Registry receipt hash must be a sha256 identifier")
        for path in (self.root / "receipts").glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("receipt_hash") == receipt_hash:
                unsigned = dict(value)
                unsigned.pop("receipt_hash", None)
                if _hash_json(unsigned) != receipt_hash:
                    raise RuntimeError("Registry receipt hash verification failed")
                return value
        raise FileNotFoundError(f"Registry receipt not found: {receipt_hash}")

    def _read_state(self) -> dict[str, Any]:
        value = json.loads(self._state_path.read_text(encoding="utf-8"))
        if value.get("schema_version") != "rosclaw.simulation_champion_registry.v1":
            raise RuntimeError("invalid Simulation Champion Registry schema")
        return value

    def _write_state(self, value: dict[str, Any]) -> None:
        temporary = self._state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self._state_path)

    def _write_receipt(self, kind: str, receipt_hash: str, value: dict[str, Any]) -> None:
        root = self.root / "receipts"
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"{kind}_{receipt_hash.removeprefix('sha256:')[:24]}.json"
        if target.exists():
            raise RuntimeError("duplicate Registry receipt")
        target.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )


def _slot(task_id: str, body_snapshot_hash: str) -> str:
    return f"{task_id}|{body_snapshot_hash}"


def _hash_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "ActiveChampionEpisodeReceipt",
    "CanaryReceipt",
    "ChampionActivationReceipt",
    "RollbackReceipt",
    "SimulationChampionRegistry",
]
