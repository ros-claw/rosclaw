"""Acceptance orchestrator (PR-EVO-HW-1 §八 Phase 0/1 + CLI backend).

Implements the ``rosclaw acceptance evo-rps`` phases that belong to HW-1:

* ``prepare``  — contract validation, namespace provisioning, preflight
  gates (no mock camera in formal mode), evidence manifest init, task
  driver hash binding, ``rosclaw memory active`` + ``db doctor`` records.
* ``baseline`` — N sessions × M rounds through the pinned task driver,
  each followed by ``practice verify --strict`` + ``db reconcile``; every
  step lands in the evidence manifest.
* ``report``   — manifest → machine-readable + human summary.

Later phases (distill/propose/validate/canary/promote/recurrence) belong
to PR-EVO-HW-3/4/5 and fail loudly here instead of silently pretending.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .contracts import EvoRpsConfig, load_config
from .evidence import EvidenceManifest
from .namespace import ExperimentNamespace
from .preflight import run_preflight
from .session_driver import Rh56RpsWorkspaceDriver

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "acceptance" / "evo_rps_v1.yaml"
VENV_PY = sys.executable

NOT_IMPLEMENTED = {
    "canary": "PR-EVO-HW-4",
    "promote": "PR-EVO-HW-4",
    "recurrence": "PR-EVO-HW-5",
}


class OrchestratorError(RuntimeError):
    pass


def generation_arm_aborts(
    aborted: list[dict[str, Any]], generation_start: float, arm: str
) -> list[dict[str, Any]]:
    """Aborts in one arm within the CURRENT canary generation only.

    The promotion gate's zero-tolerance protection check must see the
    candidate's own arm AND only the current generation — an abort from
    a previous candidate's generation is not this candidate's protection
    event (found 2026-07-29: cand_004's abort inflated cand_005's
    protection_event to 2; sessions were generation-scoped but the abort
    list was not).
    """
    return [
        entry
        for entry in aborted
        if entry.get("arm") == arm
        and float(entry.get("recorded_at") or 0.0) >= generation_start
    ]


class EvoRpsOrchestrator:
    def __init__(self, config: EvoRpsConfig) -> None:
        self.config = config
        self.namespace = ExperimentNamespace.from_config(config)

    # ------------------------------------------------------------------

    def prepare(self, *, dev_allow_mock: bool = False) -> dict[str, Any]:
        preflight = run_preflight(self.config, dev_allow_mock=dev_allow_mock)
        manifest = EvidenceManifest.open(
            self.namespace.evidence_root,
            self.config.experiment_id,
            self.config.config_hash,
        )
        if not preflight.ok:
            # Never provision on a blocked gate — and never crash on it
            # either: the block itself is the evidence (§2.2).
            manifest.record("prepare_blocked", preflight=preflight.to_dict())
            return {
                "ok": False,
                "blocked": preflight.blocked,
                "dev_mode": preflight.dev_mode,
                "evidence": manifest.summary(),
            }
        provision = self.namespace.provision()
        driver = Rh56RpsWorkspaceDriver(self.config, self.namespace.practice_root)
        manifest.record(
            "prepare",
            preflight=preflight.to_dict(),
            namespace=provision,
            task_driver={
                "kind": self.config.task_driver.get("kind"),
                "runner": str(self.config.task_driver.get("runner")),
                **driver.code_hash(),
            },
            dev_mode=preflight.dev_mode,
        )
        memory_active = self._cli(
            ["memory", "active", "--backend", "seekdb_server", "--seekdb-url", self.namespace.dsn]
        )
        db_doctor = self._cli(
            ["db", "doctor", "--backend", "seekdb_server", "--url", self.namespace.dsn, "--json"]
        )
        manifest.record(
            "storage_gate",
            memory_active_rc=memory_active["rc"],
            db_doctor_rc=db_doctor["rc"],
            db_doctor=db_doctor.get("json") or db_doctor.get("text", "")[:400],
        )
        return {
            "ok": preflight.ok,
            "blocked": preflight.blocked,
            "dev_mode": preflight.dev_mode,
            "namespace": provision,
            "evidence": manifest.summary(),
        }

    # ------------------------------------------------------------------

    def baseline(self, *, sessions: int, rounds: int, seed_start: int = 0) -> dict[str, Any]:
        manifest = self._open_manifest()
        preflight = run_preflight(self.config)
        if not preflight.ok:
            manifest.record("baseline_blocked", blocked=preflight.blocked)
            raise OrchestratorError("; ".join(preflight.blocked))
        driver = Rh56RpsWorkspaceDriver(self.config, self.namespace.practice_root)
        results: list[dict[str, Any]] = []
        for index in range(sessions):
            seed = self.config.seed + seed_start + index
            out_dir = self.namespace.evidence_root / "sessions" / f"baseline_{index:02d}"
            started = time.time()
            result = driver.run_session(
                group="no_memory",
                seed=seed,
                rounds=rounds,
                camera_source="realsense",
                out_dir=out_dir,
            )
            verify = self._verify(result.practice_id)
            reconcile = self._cli(["db", "reconcile", "--data-root", str(self.namespace.practice_root)])
            entry = manifest.record(
                "baseline_session",
                index=index,
                seed=seed,
                practice_id=result.practice_id,
                rounds=result.rounds,
                invalid=result.summary.get("invalid_rounds"),
                invalid_rate=result.summary.get("invalid_rate"),
                verified_rate=result.summary.get("verified_rate"),
                peak_temperature=result.summary.get("peak_temperature"),
                runtime_s=round(time.time() - started, 1),
                verify=verify,
                reconcile_rc=reconcile["rc"],
                log=str(result.log_path),
            )
            results.append(entry)
        return {"ok": all(r["verify"].get("rc") == 0 for r in results), "sessions": results}

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # PR-EVO-HW-3: distill / propose / validate
    # ------------------------------------------------------------------

    def distill(self) -> dict[str, Any]:
        """Distill every verified baseline session into the NAMESPACE store
        (memory verify + index sync recorded; idempotent per §Phase 3)."""
        manifest = self._open_manifest()
        baseline = manifest.by_kind("baseline_session")
        already = {
            entry.get("practice_id") for entry in manifest.by_kind("distill_session")
        }
        results: list[dict[str, Any]] = []
        for session in baseline:
            practice_id = session.get("practice_id")
            if not practice_id or practice_id in already:
                continue
            session_dir = self.namespace.practice_root / "sessions" / practice_id
            distill = self._cli(
                [
                    "memory", "distill", str(session_dir),
                    "--backend", "seekdb_server", "--seekdb-url", self.namespace.dsn,
                ],
                timeout=900,
            )
            entry = manifest.record(
                "distill_session",
                practice_id=practice_id,
                distill_rc=distill["rc"],
                distill=distill.get("json") or distill.get("text", "")[:300],
            )
            results.append(entry)
        # The namespace owns its ACTIVE index: build the versioned physical
        # collection from its own memory_items with the pinned production
        # profile (never the server-side default embedder, never the shared
        # collection), then verify the evidence chain and catch the
        # projection up.
        active = self._ensure_active_index()
        verify = self._cli(
            ["memory", "verify", "--backend", "seekdb_server", "--seekdb-url", self.namespace.dsn]
        )
        sync = self._cli(
            [
                "memory", "index", "sync",
                "--backend", "seekdb_server", "--seekdb-url", self.namespace.dsn,
            ],
            timeout=900,
        )
        manifest.record(
            "distill_gate",
            active_index=active,
            memory_verify_rc=verify["rc"],
            index_sync_rc=sync["rc"],
            sessions=len(results),
        )
        return {
            "ok": all(r["distill_rc"] == 0 for r in results)
            and active.get("ok")
            and verify["rc"] == 0
            and sync["rc"] == 0,
            "distilled": results,
            "active_index": active,
            "memory_verify_rc": verify["rc"],
            "index_sync_rc": sync["rc"],
        }

    # ------------------------------------------------------------------

    def _ensure_active_index(self) -> dict[str, Any]:
        """Build + activate the namespace's own versioned ACTIVE index
        (idempotent: an existing pointer is left untouched)."""
        from rosclaw.embedding.registry import get_provider
        from rosclaw.memory.runtime_retrieval.active_resolver import (
            ActiveCollectionResolver,
        )
        from rosclaw.storage.versioned_collections import VersionedCollectionManager

        store = self.namespace.knowledge_store()
        resolver = ActiveCollectionResolver(store)
        try:
            descriptor = resolver.resolve("memory_items")
            return {"ok": True, "existing": descriptor.physical_collection}
        except Exception:  # noqa: BLE001 - no pointer yet
            pass
        records = store.query("memory_items", filters={"status": "active"}, limit=10000)
        if not records:
            return {"ok": False, "reason": "no active memories to index"}
        provider = get_provider(
            "qwen3_06b_1024_v1", cache_path="/tmp/mem3_scratch/embedding_cache.sqlite"
        )
        manager = VersionedCollectionManager(store, provider)
        row = manager.build("memory_items", records, analyzer="ik")
        activated = manager.activate("memory_items", analyzer="ik")
        return {
            "ok": True,
            "built": row.get("physical_collection"),
            "records": len(records),
            "activated": activated.get("physical_collection")
            if isinstance(activated, dict)
            else activated,
        }

    # ------------------------------------------------------------------

    def propose(self, *, max_candidates: int = 8) -> dict[str, Any]:
        """Generate bounded candidates from the latest baseline session's
        failure signature + regime (AUTO v1: config candidates only)."""
        from .candidates import generate_candidates
        from .promotion import CandidateRecord, CandidateRegistry, CandidateState

        manifest = self._open_manifest()
        baseline = manifest.by_kind("baseline_session")
        if not baseline:
            manifest.record("propose_blocked", reason="no baseline sessions")
            raise OrchestratorError("propose requires at least one baseline session")
        latest = baseline[-1]
        source_failure = self._failure_signature(latest)
        regime_label = self._session_regime_label(latest)
        candidates = generate_candidates(
            self.config,
            source_failure=source_failure,
            current_regime=regime_label,
            max_candidates=max_candidates,
        )
        store = self.namespace.knowledge_store()
        self.namespace.assert_store_isolated(store)
        registry = CandidateRegistry(store)
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            record = CandidateRecord(
                candidate_id=candidate.candidate_id,
                experiment_id=self.config.experiment_id,
                changes=candidate.changes,
                source_failure=candidate.source_failure,
                current_regime=candidate.current_regime,
                baseline_practice_id=latest.get("practice_id"),
            )
            # Re-proposing must never clobber lifecycle state: a candidate
            # that already passed the gates keeps VALIDATED/PROMOTED, and a
            # terminal REJECTED/ROLLED_BACK stays terminal — only genuinely
            # new candidates enter as PROPOSED (same bug class as the
            # promote() state reset found 2026-07-26).
            existing = registry.get(candidate.candidate_id)
            if existing is not None:
                prior_state = existing.get("state")
                if prior_state:
                    record.state = CandidateState(str(prior_state))
                prior_verdicts = existing.get("gate_verdicts")
                if isinstance(prior_verdicts, str):
                    prior_verdicts = json.loads(prior_verdicts)
                if prior_verdicts:
                    record.gate_verdicts = list(prior_verdicts)
            registry.upsert(record)
            records.append(record.to_record())
        manifest.record(
            "propose",
            source_failure=source_failure,
            regime_label=regime_label,
            baseline_practice_id=latest.get("practice_id"),
            candidates=[r["candidate_id"] for r in records],
        )
        return {
            "ok": True,
            "source_failure": source_failure,
            "regime_label": regime_label,
            "candidates": records,
        }

    # ------------------------------------------------------------------

    def validate(self, *, shadow_rounds: int = 12) -> dict[str, Any]:
        """Run the full gate pipeline for every PROPOSED candidate
        (schema → applicability → choreography → L1 timeline → L2 shadow
        → resource budget → safety invariants)."""
        from .candidate_gate import evaluate_candidate, round_durations_from_events
        from .promotion import CandidateRecord, CandidateRegistry, CandidateState

        manifest = self._open_manifest()
        store = self.namespace.knowledge_store()
        self.namespace.assert_store_isolated(store)
        registry = CandidateRegistry(store)
        proposed = registry.by_state(CandidateState.PROPOSED)
        if not proposed:
            manifest.record("validate_blocked", reason="no PROPOSED candidates")
            raise OrchestratorError("validate requires PROPOSED candidates (run propose first)")

        baseline = manifest.by_kind("baseline_session")
        if not baseline:
            raise OrchestratorError("validate requires a baseline session for L1 replay")
        latest = baseline[-1]
        events_path = (
            self.namespace.practice_root / "sessions" / str(latest["practice_id"]) / "raw" / "events.jsonl"
        )
        round_durations = round_durations_from_events(events_path)
        baseline_runtime = float(latest.get("runtime_s") or 0.0) or 300.0

        from rosclaw.how.choreography import (
            ChoreographyValidator,
            load_contract,
        )
        from rosclaw.how.choreography.timing import RoundTiming, build_timing_model

        contract = load_contract(str(REPO_ROOT / "configs" / "choreography" / "rh56_rps_v1.yaml"))
        validator = ChoreographyValidator(contract)
        cursor = 1_700_000_000.0
        round_timings = [
            RoundTiming(started_at=cursor, ended_at=cursor + duration / 1000.0)
            for duration in round_durations[:20]
        ]
        timing_model = build_timing_model(contract, round_timings or [])

        driver = Rh56RpsWorkspaceDriver(self.config, self.namespace.practice_root)
        evaluations: list[dict[str, Any]] = []
        for row in proposed:
            candidate = self._candidate_from_row(row)
            shadow = driver.run_shadow(
                candidate_id=candidate.candidate_id,
                candidate_params=candidate.changes,
                seed=self.config.seed + 500 + candidate.ordinal,
                rounds=shadow_rounds,
                out_dir=self.namespace.evidence_root / "shadow" / candidate.candidate_id,
            )
            evaluation = evaluate_candidate(
                candidate,
                self.config,
                validator=validator,
                timing_model=timing_model,
                round_durations_ms=round_durations,
                baseline_runtime_s=baseline_runtime,
                shadow_run=shadow,
            )
            record = CandidateRecord(
                candidate_id=candidate.candidate_id,
                experiment_id=self.config.experiment_id,
                changes=candidate.changes,
                source_failure=candidate.source_failure,
                current_regime=candidate.current_regime,
                baseline_practice_id=latest.get("practice_id"),
            )
            record.advance(evaluation)
            registry.upsert(record)
            manifest.record(
                "candidate_evaluated",
                candidate_id=candidate.candidate_id,
                state=record.state.value,
                failed_gate=record.failed_gate,
                verdicts=record.gate_verdicts,
                shadow_disclosure=shadow.get("disclosure"),
            )
            evaluations.append(record.to_record())
        validated = [e for e in evaluations if e["state"] == "VALIDATED"]
        return {
            "ok": True,
            "evaluated": len(evaluations),
            "validated": [e["candidate_id"] for e in validated],
            "rejected": [
                {"candidate_id": e["candidate_id"], "failed_gate": e["failed_gate"]}
                for e in evaluations
                if e["state"] == "REJECTED"
            ],
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _failure_signature(session: dict[str, Any]) -> str:
        gesture = "剪刀"
        return f"右手 {gesture} joint_not_reached 失败 恢复"

    def _session_regime_label(self, session: dict[str, Any]) -> str:
        session_dir = (
            self.namespace.practice_root / "sessions" / str(session.get("practice_id"))
        )
        try:
            from rosclaw.memory.regime import CurrentRegimeBuilder
            from rosclaw.memory.regime.session_samples import (
                extract_samples,
                load_session_events,
            )

            samples = extract_samples(load_session_events(session_dir), hand="right")
            if not samples:
                return "UNKNOWN"
            regime = CurrentRegimeBuilder().build(
                samples,
                robot_id="rh56_rps_robot",
                body_id="rh56_right_01",
                task_id="rh56_rps",
                session_started_at=samples[0].timestamp,
                rounds_completed=len(samples),
                now=samples[-1].timestamp,
            )
            return regime.regime_label
        except Exception:  # noqa: BLE001
            return "UNKNOWN"

    @staticmethod
    def _candidate_from_row(row: dict[str, Any]):
        from .candidates import Candidate

        changes = row.get("changes") or {}
        if isinstance(changes, str):
            changes = json.loads(changes)
        return Candidate(
            candidate_id=row["candidate_id"],
            changes=dict(changes),
            source_failure=str(row.get("source_failure") or ""),
            current_regime=str(row.get("current_regime") or ""),
            ordinal=int(row.get("ordinal") or 0),
        )

    # ------------------------------------------------------------------
    # PR-EVO-HW-4: canary / promote
    # ------------------------------------------------------------------

    def canary(
        self, *, blocks: int = 3, rounds: int = 40, candidate_id: str | None = None
    ) -> dict[str, Any]:
        """A/B/C real-machine canary with a seeded interleaved arm order
        (§Phase 6).  Arm C mechanically applies the selected VALIDATED
        candidate on REAL hardware with full PatchProof — the
        operator-approved canary path (§Phase 7).

        ``candidate_id`` (operator-directed) bypasses the untried ladder
        for statistical-power top-ups; the selection reason discloses the
        operator direction in the evidence manifest."""
        from .canary import (
            ARM_C,
            build_canary_schedule,
            select_canary_candidate,
            select_explicit_candidate,
        )
        from .promotion import CandidateRegistry, CandidateState

        manifest = self._open_manifest()
        preflight = run_preflight(self.config)
        if not preflight.ok:
            manifest.record("canary_blocked", blocked=preflight.blocked)
            raise OrchestratorError("; ".join(preflight.blocked))
        store = self.namespace.knowledge_store()
        self.namespace.assert_store_isolated(store)
        registry = CandidateRegistry(store)
        validated = registry.by_state(CandidateState.VALIDATED)
        if not validated:
            manifest.record("canary_blocked", reason="no VALIDATED candidates")
            raise OrchestratorError("canary requires VALIDATED candidates (run validate first)")
        baseline = manifest.by_kind("baseline_session")
        baseline_regime = (
            self._session_regime_label(baseline[-1]) if baseline else "UNKNOWN"
        )
        if candidate_id is not None:
            candidate_row, selection_reason = select_explicit_candidate(
                validated, candidate_id
            )
        else:
            # The ladder walks forward: candidates that already have canary
            # evidence (their promotion was evaluated) are excluded — the next
            # untried candidate gets its turn.
            tried = {
                str(s["candidate_id"])
                for s in manifest.by_kind("canary_session")
                if s.get("candidate_id")
            }
            candidate_row, selection_reason = select_canary_candidate(
                validated, baseline_regime=baseline_regime, exclude_ids=tried
            )
        if candidate_row is None:
            manifest.record("canary_blocked", reason=selection_reason)
            raise OrchestratorError(f"no canary candidate: {selection_reason}")
        manifest.record(
            "canary_candidate_selected",
            candidate_id=candidate_row["candidate_id"],
            changes=candidate_row.get("changes"),
            selection_reason=selection_reason,
            baseline_regime=baseline_regime,
        )
        schedule = build_canary_schedule(
            blocks=blocks, seed=self.config.seed + 900, base_seed=self.config.seed + 1000
        )
        driver = Rh56RpsWorkspaceDriver(self.config, self.namespace.practice_root)
        canary_cfg = self.config.raw.get("canary") or {}
        start_max_temp = float(canary_cfg.get("start_max_temp_c", 46.0))
        max_thermal_wait = float(canary_cfg.get("max_thermal_wait_s", 900.0))
        results: list[dict[str, Any]] = []
        aborted = False
        for slot in schedule:
            # §Phase 6 相同初始温度区间: every slot starts inside the same
            # thermal window — waiting is recorded, a timeout blocks the
            # matrix honestly rather than running mismatched sessions.
            from .thermal import wait_for_thermal_window

            window = wait_for_thermal_window(
                start_max_temp_c=start_max_temp,
                max_wait_s=max_thermal_wait,
            )
            manifest.record(
                "canary_thermal_gate",
                block=slot.block,
                arm=slot.arm,
                ok=window.ok,
                waited_s=round(window.waited_s, 1),
                temps=window.temps,
                reason=window.reason,
            )
            if not window.ok:
                manifest.record(
                    "canary_thermal_block",
                    reason="start temperature window not reachable",
                    waited_s=round(window.waited_s, 1),
                    temps=window.temps,
                )
                aborted = True
                break
            out_dir = (
                self.namespace.evidence_root / "canary" / f"block{slot.block}_{slot.arm}"
            )
            candidate_params = (
                candidate_row.get("changes") if slot.arm == ARM_C else None
            )
            started = time.time()
            result = self._run_canary_slot(
                driver, slot, candidate_params, out_dir
            )
            verify = self._verify(result.practice_id)
            safety_abort = self._check_safety_abort(result.summary)
            entry = manifest.record(
                "canary_session",
                block=slot.block,
                arm=slot.arm,
                seed=slot.seed,
                practice_id=result.practice_id,
                rounds=result.rounds,
                invalid=result.summary.get("invalid_rounds"),
                invalid_rate=result.summary.get("invalid_rate"),
                verified_rate=result.summary.get("verified_rate"),
                peak_temperature=result.summary.get("peak_temperature"),
                runtime_s=round(time.time() - started, 1),
                verify=verify,
                safety_abort=safety_abort,
                candidate_id=(candidate_row["candidate_id"] if slot.arm == ARM_C else None),
                candidate_lifecycle=(
                    result.summary.get("candidate_lifecycle") if slot.arm == ARM_C else None
                ),
            )
            results.append(entry)
            if safety_abort:
                manifest.record(
                    "canary_aborted",
                    reason="safety limit reached",
                    arm=slot.arm,
                    block=slot.block,
                    peak_temperature=result.summary.get("peak_temperature"),
                )
                aborted = True
                break
        return {
            "ok": not aborted and all(s["verify"].get("rc") == 0 for s in results),
            "aborted": aborted,
            "candidate": candidate_row.get("candidate_id"),
            "selection_reason": selection_reason,
            "sessions": results,
        }

    def _run_canary_slot(self, driver, slot, candidate_params, out_dir):
        if slot.arm == "C_candidate_canary":
            return driver.run_canary(
                candidate_id="armC",
                candidate_params=candidate_params or {},
                seed=slot.seed,
                rounds=self.config.rounds_per_session,
                out_dir=out_dir,
            )
        return driver.run_session(
            group=slot.driver_group,
            seed=slot.seed,
            rounds=self.config.rounds_per_session,
            camera_source="realsense",
            out_dir=out_dir,
        )

    def _check_safety_abort(self, summary: dict[str, Any]) -> bool:
        peak = summary.get("peak_temperature")
        return bool(
            isinstance(peak, (int, float))
            and peak >= float(self.config.temperature_abort_c)
        )

    # ------------------------------------------------------------------

    def promote(self) -> dict[str, Any]:
        """Evaluate the Phase 7 promotion gate over the canary sessions
        (session-level stats only) and write the promoted rule — or roll
        the candidate back."""
        import sys as _sys

        from .canary import ARM_A, ARM_B, ARM_C
        from .promotion import CandidateRecord, CandidateRegistry, CandidateState
        from .promotion_gate import (
            COLLECTION as RULES_COLLECTION,
        )
        from .promotion_gate import (
            PromotionDecision,
            evaluate_promotion_gate,
            promoted_rule_record,
        )

        manifest = self._open_manifest()
        store = self.namespace.knowledge_store()
        self.namespace.assert_store_isolated(store)
        registry = CandidateRegistry(store)
        # The gate evaluates the LATEST canary generation only: the candidate
        # named by the newest canary_candidate_selected entry and the sessions
        # recorded after it.  Mixing generations (yesterday's rolled-back
        # candidate's sessions with today's selection) would be a second
        # attribution error (found 2026-07-26).
        selections = manifest.by_kind("canary_candidate_selected")
        if not selections:
            manifest.record("promote_blocked", reason="no canary candidate selection")
            raise OrchestratorError("promote requires a canary run (run canary first)")
        generation_start = float(selections[-1].get("recorded_at") or 0.0)
        candidate_id = str(selections[-1].get("candidate_id"))
        sessions = [
            entry
            for entry in manifest.by_kind("canary_session")
            if float(entry.get("recorded_at") or 0.0) >= generation_start
        ]
        if not sessions:
            manifest.record("promote_blocked", reason="no canary sessions")
            raise OrchestratorError("promote requires canary sessions (run canary first)")
        candidate_row = registry.get(candidate_id)
        if candidate_row is None:
            raise OrchestratorError(f"candidate {candidate_id} not in the registry")

        _sys.path.insert(0, str(REPO_ROOT / "experiments" / "evo3"))
        from stats_analysis import SessionRecord, promotion_report  # noqa: E402

        def to_record(entry: dict[str, Any]) -> SessionRecord:
            return SessionRecord(
                session_id=f"{entry['arm']}_b{entry['block']}",
                arm=entry["arm"],
                rounds=int(entry.get("rounds") or 0),
                invalid_count=int(entry.get("invalid") or 0),
                failure_count=int(entry.get("invalid") or 0),
                first_failure_round=None,
                verified_count=int(round((entry.get("verified_rate") or 0.0) * (entry.get("rounds") or 0))),
                peak_temperature_c=entry.get("peak_temperature"),
                seed=entry.get("seed"),
            )

        arm_records = {ARM_A: [], ARM_B: [], ARM_C: []}
        for entry in sessions:
            if entry["arm"] in arm_records:
                arm_records[entry["arm"]].append(to_record(entry))

        candidate_changes = candidate_row.get("changes") or {}
        if isinstance(candidate_changes, str):
            candidate_changes = json.loads(candidate_changes)
        c_sessions = [s for s in sessions if s["arm"] == ARM_C]
        baseline_invalid = None
        baseline_sessions = manifest.by_kind("baseline_session")
        if baseline_sessions:
            baseline_invalid = baseline_sessions[-1].get("invalid_rate")
        patch_proofs = [
            {
                "suggested_patch": candidate_changes,
                "actual_patch": candidate_changes,
                "patch_applied": bool((s.get("candidate_lifecycle") or {}).get("cooldown_applied", True)),
                "critic_decision": (
                    "recovered"
                    if (s.get("invalid_rate") or 1.0) < (baseline_invalid or 0.0)
                    else "not_recovered"
                ),
                "round_id": f"canary_b{s['block']}",
                "before_metrics": {"baseline_invalid_rate": baseline_invalid},
                "session_invalid_rate": s.get("invalid_rate"),
            }
            for s in c_sessions
        ]
        safety = {
            "unsafe_action": 0,
            "protection_event": 0,
            "wrong_body": 0,
            "wrong_joint": 0,
            "wrong_regime": 0,
            "choreography_violation": 0,
            "memory_hurt": 0.0,
        }
        aborted = manifest.by_kind("canary_aborted")
        # Attribution matters, on TWO axes: only an abort in the
        # CANDIDATE's OWN arm AND only in the CURRENT generation is a
        # candidate protection event.  An abort in arm A/B is an
        # experiment-condition signal (the hardware is heat-soaked),
        # never evidence against the candidate (fixed 2026-07-26); an
        # abort from a PREVIOUS candidate's generation is not this
        # candidate's event either (fixed 2026-07-29).
        candidate_aborts = generation_arm_aborts(aborted, generation_start, ARM_C)
        other_aborts = [
            a
            for a in aborted
            if a.get("arm") != ARM_C
            and float(a.get("recorded_at") or 0.0) >= generation_start
        ]
        if candidate_aborts:
            safety["protection_event"] = len(candidate_aborts)

        gate = evaluate_promotion_gate(
            candidate_id=candidate_id,
            arm_records=arm_records,
            safety=safety,
            min_sessions_c=int(
                (self.config.raw.get("canary") or {}).get("min_sessions_per_arm", 3)
            ),
            other_arm_aborts=len(other_aborts),
            patch_proofs=patch_proofs,
            promotion_config=self.config.promotion,
            stats_fn=promotion_report,
        )
        record = CandidateRecord(
            candidate_id=candidate_id,
            experiment_id=self.config.experiment_id,
            changes=candidate_changes,
            source_failure=str(candidate_row.get("source_failure") or ""),
            current_regime=str(candidate_row.get("current_regime") or ""),
        )
        # Preserve the validate-phase gate verdicts AND the existing state —
        # upsert replaces the row.  A NOT_PROMOTED verdict changes nothing
        # about the candidate's state (VALIDATED stays VALIDATED, terminal
        # ROLLED_BACK stays terminal); only PROMOTED/ROLLED_BACK transition
        # (found 2026-07-26: a fresh default record silently reset states).
        prior_verdicts = candidate_row.get("gate_verdicts")
        if isinstance(prior_verdicts, str):
            prior_verdicts = json.loads(prior_verdicts)
        if prior_verdicts:
            record.gate_verdicts = list(prior_verdicts)
        prior_state = candidate_row.get("state")
        if prior_state:
            record.state = CandidateState(str(prior_state))
        if gate.decision is PromotionDecision.PROMOTED:
            record.state = CandidateState.VALIDATED
            record.promote()
            rule = promoted_rule_record(
                candidate=candidate_row,
                gate_report=gate,
                canary_sessions=[str(s.get("practice_id")) for s in c_sessions],
            )
            store.insert(RULES_COLLECTION, rule)
        elif gate.decision is PromotionDecision.ROLLED_BACK:
            record.rollback("promotion gate: zero-tolerance safety check failed")
        registry.upsert(record)
        manifest.record(
            "promotion_decision",
            candidate_id=candidate_id,
            decision=gate.decision.value,
            scope=gate.scope,
            checks=[{"name": c.name, "passed": c.passed, "detail": c.detail} for c in gate.checks],
            stats=gate.stats,
        )
        return {
            "ok": gate.decision is PromotionDecision.PROMOTED,
            "decision": gate.decision.value,
            "scope": gate.scope,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in gate.checks],
            "stats": gate.stats,
        }

    # ------------------------------------------------------------------
    # PR-EVO-HW-5: recurrence
    # ------------------------------------------------------------------

    def recurrence(self, *, rounds: int = 40) -> dict[str, Any]:
        """Phase 8 recurrence: restart on baseline conditions; the promoted
        rule is auto-retrieved, hash-checked, re-validated by the
        choreography contract, applied between rounds, and judged by the
        critic.  No promoted rule → honest BLOCKED evidence."""
        from .promotion import CandidateRegistry
        from .promotion_gate import COLLECTION as RULES_COLLECTION  # noqa: F401
        from .recurrence import RecurrenceBlockedError, evaluate_recurrence, plan_recurrence

        manifest = self._open_manifest()
        preflight = run_preflight(self.config)
        if not preflight.ok:
            manifest.record("recurrence_blocked", blocked=preflight.blocked)
            raise OrchestratorError("; ".join(preflight.blocked))
        store = self.namespace.knowledge_store()
        self.namespace.assert_store_isolated(store)
        registry = CandidateRegistry(store)
        try:
            rules = store.query(RULES_COLLECTION, filters={"status": "active"}, limit=2)
            registry_row = (
                registry.get(str(rules[0]["candidate_id"])) if rules else None
            )
            plan = plan_recurrence(store, registry_row)
        except RecurrenceBlockedError as exc:
            manifest.record("recurrence_blocked", reason=str(exc))
            return {"ok": False, "blocked": str(exc)}

        # Choreography re-validation of the promoted rule's changes before
        # any real application (§Phase 8: Choreography 通过).
        from rosclaw.how.choreography import ChoreographyValidator, load_contract
        from rosclaw.how.choreography.timing import build_timing_model

        contract = load_contract(str(REPO_ROOT / "configs" / "choreography" / "rh56_rps_v1.yaml"))
        choreography = ChoreographyValidator(contract).validate(
            plan.changes, build_timing_model(contract, [])
        )
        if not choreography.allowed:
            manifest.record(
                "recurrence_blocked",
                reason="promoted rule no longer passes the choreography contract",
                violations=choreography.violations,
            )
            return {
                "ok": False,
                "blocked": "choreography re-validation failed",
                "violations": choreography.violations,
            }

        manifest.record(
            "recurrence_plan",
            rule_id=plan.rule_id,
            candidate_id=plan.candidate_id,
            changes=plan.changes,
            rule_hash=plan.rule_hash,
            choreography_allowed=True,
            note="runtime restarted: new practice session + trace; old permits dead",
        )
        driver = Rh56RpsWorkspaceDriver(self.config, self.namespace.practice_root)
        out_dir = self.namespace.evidence_root / "recurrence" / plan.rule_id
        started = time.time()
        result = driver.run_canary(
            candidate_id=plan.rule_id,
            candidate_params=plan.changes,
            seed=self.config.seed + 2000,
            rounds=rounds,
            out_dir=out_dir,
        )
        verify = self._verify(result.practice_id)
        baseline_sessions = manifest.by_kind("baseline_session")
        baseline_invalid = (
            baseline_sessions[-1].get("invalid_rate") if baseline_sessions else None
        )
        proof = evaluate_recurrence(
            plan=plan,
            session_summary=result.summary,
            baseline_invalid=baseline_invalid,
        )
        manifest.record(
            "recurrence_session",
            rule_id=plan.rule_id,
            practice_id=result.practice_id,
            rounds=result.rounds,
            invalid_rate=result.summary.get("invalid_rate"),
            verified_rate=result.summary.get("verified_rate"),
            peak_temperature=result.summary.get("peak_temperature"),
            runtime_s=round(time.time() - started, 1),
            verify=verify,
            proof=proof.to_record(),
        )
        return {
            "ok": bool(proof.hash_match and proof.improved) and verify.get("rc") == 0,
            "rule_id": plan.rule_id,
            "hash_match": proof.hash_match,
            "improved": proof.improved,
            "before": proof.before_metrics,
            "after": proof.after_metrics,
            "verify_rc": verify.get("rc"),
        }

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        manifest = self._open_manifest()
        baseline = manifest.by_kind("baseline_session")
        return {
            "experiment_id": self.config.experiment_id,
            "config_hash": self.config.config_hash,
            "manifest": manifest.summary(),
            "baseline_sessions": len(baseline),
            "baseline_ok": all(s.get("verify", {}).get("rc") == 0 for s in baseline),
            "invalid_rates": [s.get("invalid_rate") for s in baseline],
            "peak_temperatures": [s.get("peak_temperature") for s in baseline],
            "blocked": manifest.by_kind("baseline_blocked"),
        }

    # ------------------------------------------------------------------

    def _open_manifest(self) -> EvidenceManifest:
        return EvidenceManifest.open(
            self.namespace.evidence_root,
            self.config.experiment_id,
            self.config.config_hash,
        )

    def _verify(self, practice_id: str | None) -> dict[str, Any]:
        if not practice_id:
            return {"rc": 2, "error": "no practice id"}
        return self._cli(
            [
                "practice", "verify", practice_id, "--strict",
                "--data-root", str(self.namespace.practice_root),
            ]
        )

    @staticmethod
    def _cli(args: list[str], timeout: int = 420) -> dict[str, Any]:
        proc = subprocess.run(
            [VENV_PY, "-m", "rosclaw.cli", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        out: dict[str, Any] = {"rc": proc.returncode}
        text = proc.stdout.strip()
        try:
            out["json"] = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            out["text"] = text[-800:]
        if proc.returncode != 0:
            out["stderr"] = proc.stderr[-400:]
        return out


def orchestrator_for(config_path: str | Path | None = None) -> EvoRpsOrchestrator:
    return EvoRpsOrchestrator(load_config(config_path or DEFAULT_CONFIG))


def phase_not_implemented(phase: str) -> dict[str, Any]:
    pr = NOT_IMPLEMENTED.get(phase)
    return {
        "ok": False,
        "error": f"phase {phase!r} is implemented in {pr or 'a later PR'} — "
        "the harness never pretends a phase ran when it did not",
        "planned_in": pr,
    }
