"""AutoEngine — ROSClaw Self-Evolution Control Plane."""
import contextlib
import uuid
from datetime import UTC, datetime
from typing import Any

from ..config import AutoConfig
from ..core import (
    AutoTask,
    Champion,
    DeadEnd,
    Diagnosis,
    EvaluationResult,
    EvolutionReport,
    ExperimentSpec,
    FailureCase,
    Patch,
    Proposal,
)
from ..events.publishers import AutoPublisher
from ..promotion import ChampionStore, LineageTracker, PromotionGate, RollbackManager
from ..runners import DarwinRunner, LocalRunner, SandboxRunner
from ..storage import LocalStore


class AutoEngine:
    """AutoEngine 主引擎.

    负责 Proposal -> Patch -> Run -> Evaluate -> Promote 的闭环调度.
    集成 rosclaw-runtime: event_bus, seekdb, skill_registry.
    """

    def __init__(self, config: AutoConfig | None = None,
                 event_bus: Any | None = None,
                 seekdb_client: Any | None = None,
                 skill_registry: Any | None = None,
                 sense_runtime: Any | None = None):
        self.config = config or AutoConfig()
        self._event_bus = event_bus
        self._seekdb = seekdb_client
        self._skill_registry = skill_registry
        self._sense_runtime = sense_runtime
        self._auto_context_adapter: Any | None = None
        if sense_runtime is not None:
            try:
                from rosclaw.sense.adapters.auto_context import AutoContextAdapter
                self._auto_context_adapter = AutoContextAdapter(sense_runtime)
            except Exception:
                pass
        self.store = LocalStore(self.config.local_store_path)
        # Sprint C: runners
        self.local_runner = LocalRunner({"simulate": True, "latency_sec": 0.01})
        self.sandbox_runner = SandboxRunner({"simulate": True})
        self.darwin_runner = DarwinRunner({"simulate": True})
        # Sprint D: promotion
        self.promotion_gate = PromotionGate({
            "min_success_improvement": self.config.promotion_min_success_improvement,
            "max_collision_increase": self.config.promotion_max_collision_increase,
        })
        self.champion_store = ChampionStore(self.store)
        self.rollback_manager = RollbackManager(self.store)
        self.lineage = LineageTracker(self.store)
        # Event publishing
        self.publisher = AutoPublisher(event_bus=event_bus)

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------
    def create_task(self, name: str, robot: str, skill: str,
                    task_type: str = "skill_tuning", env: str = "default",
                    objective: dict | None = None) -> AutoTask:
        task = AutoTask(
            id=f"task_{uuid.uuid4().hex[:8]}",
            name=name,
            task_type=task_type,
            robot_id=robot,
            environment_id=env,
            target_skill_id=skill,
            objective=objective or {},
        )
        self._save("tasks", task.id, task.to_dict())
        return task

    def get_task(self, task_id: str) -> AutoTask | None:
        data = self._load("tasks", task_id)
        return AutoTask.from_dict(data) if data else None

    def list_tasks(self) -> list[AutoTask]:
        return [AutoTask.from_dict(d) for d in self._iterate("tasks")]

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------
    def create_failure_case(self, praxis_event_id: str, task_id: str, skill_id: str,
                            phase: str = "", failure_mode: str = "",
                            severity: str = "medium", evidence: dict | None = None) -> FailureCase:
        evidence = evidence or {}
        if self._auto_context_adapter is not None:
            import contextlib
            with contextlib.suppress(Exception):
                evidence = self._auto_context_adapter.apply(evidence)
        fc = FailureCase(
            id=f"failure_{uuid.uuid4().hex[:8]}",
            praxis_event_id=praxis_event_id,
            task_id=task_id,
            skill_id=skill_id,
            phase=phase,
            failure_mode=failure_mode,
            severity=severity,
            evidence=evidence,
        )
        self._save("failures", fc.id, fc.to_dict())
        return fc

    def list_failures(self, task_id: str | None = None) -> list[FailureCase]:
        all_f = [FailureCase.from_dict(d) for d in self._iterate("failures")]
        if task_id:
            return [f for f in all_f if f.task_id == task_id]
        return all_f

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------
    def create_diagnosis(self, failure_id: str, task: str, skill: str,
                         root_causes: list[str] | None = None,
                         search_space: dict | None = None) -> Diagnosis:
        diag = Diagnosis(
            id=f"diag_{uuid.uuid4().hex[:8]}",
            failure_id=failure_id,
            task=task,
            skill=skill,
            root_cause_candidates=root_causes or [],
            recommended_search_space=search_space or {},
        )
        self._save("diagnoses", diag.id, diag.to_dict())
        return diag

    # ------------------------------------------------------------------
    # Hypothesis & Proposal
    # ------------------------------------------------------------------
    def create_proposal(self, failure_case_id: str, task: str, target_skill: str,
                        hypothesis_statement: str, search_space: dict,
                        patch_type: str = "skill_parameter_patch",
                        source: str = "failure_guided") -> Proposal:
        prop = Proposal(
            id=f"prop_{uuid.uuid4().hex[:8]}",
            source=source,
            task=task,
            target_skill_id=target_skill,
            hypothesis_statement=hypothesis_statement,
            patch_type=patch_type,
            search_space=search_space,
            required_gates=["sandbox_check", "multi_seed_eval", "regression_check"],
        )
        self._save("proposals", prop.id, prop.to_dict())
        if self.publisher:
            self.publisher.proposal_created(
                proposal_id=prop.id,
                task_id=task,
                target_skill_id=target_skill,
                hypothesis_statement=hypothesis_statement,
            )
        if self._seekdb is not None:
            with contextlib.suppress(Exception):
                self._seekdb.insert("auto_proposals", {
                    "id": prop.id, "task_id": task,
                    "target_skill": target_skill, "source": source,
                    "hypothesis": hypothesis_statement,
                    "search_space": search_space, "status": "open",
                    "created_at": datetime.now(UTC).isoformat(),
                })
        return prop

    def list_proposals(self, task: str | None = None) -> list[Proposal]:
        all_p = [Proposal.from_dict(d) for d in self._iterate("proposals")]
        if task:
            return [p for p in all_p if p.task == task]
        return all_p

    # ------------------------------------------------------------------
    # Patch
    # ------------------------------------------------------------------
    def create_patch(self, proposal_id: str, target_skill: str,
                     changes: list[dict], patch_type: str = "skill_parameter_patch",
                     rollback_plan: dict | None = None) -> Patch:
        patch = Patch(
            id=f"patch_{uuid.uuid4().hex[:8]}",
            proposal_id=proposal_id,
            patch_type=patch_type,
            target_skill=target_skill,
            changes=changes,
            rollback_plan=rollback_plan or {},
            human_approval_required=(patch_type == "code_patch"),
        )
        self._save("patches", patch.id, patch.to_dict())
        if self._seekdb is not None:
            with contextlib.suppress(Exception):
                self._seekdb.insert("auto_patches", {
                    "id": patch.id, "proposal_id": proposal_id,
                    "target_skill": target_skill, "patch_type": patch_type,
                    "changes": changes, "status": "created",
                    "created_at": datetime.now(UTC).isoformat(),
                })
        return patch

    # ------------------------------------------------------------------
    # Experiment (Sprint C: integrated runners)
    # ------------------------------------------------------------------
    def create_experiment(self, proposal_id: str, patch_id: str, task: str,
                          baseline_skill: str, candidate_skill: str,
                          episodes: int = 50, seeds: list[int] | None = None) -> ExperimentSpec:
        exp = ExperimentSpec(
            id=f"exp_{uuid.uuid4().hex[:8]}",
            proposal_id=proposal_id,
            patch_id=patch_id,
            task=task,
            baseline_skill_id=baseline_skill,
            candidate_skill_id=candidate_skill,
            evaluation={"episodes": episodes, "seeds": seeds or [0, 1, 2],
                        "metrics": ["success_rate", "collision_rate", "completion_time"]},
            safety={"sandbox_required": True, "max_collision": 0, "max_force": 15},
            promotion={"min_success_improvement": 0.05, "max_collision_increase": 0.0},
        )
        self._save("experiments", exp.id, exp.to_dict())
        return exp

    def run_experiment(self, experiment: ExperimentSpec, runner: str = "local") -> dict:
        patch_data = self._load("patches", experiment.patch_id)
        if patch_data:
            experiment.patch_context = {
                "changes": patch_data.get("changes", []),
                "patch_type": patch_data.get("patch_type", ""),
            }
        else:
            experiment.patch_context = {"changes": [], "patch_type": ""}

        r = {"local": self.local_runner, "sandbox": self.sandbox_runner,
             "darwin": self.darwin_runner}.get(runner, self.local_runner)
        result = r.run(experiment)
        experiment.status = "completed" if result.success else "failed"
        self._save("experiments", experiment.id, experiment.to_dict())
        return result.to_dict()

    # ------------------------------------------------------------------
    # Evaluation (Sprint D: integrated promotion gate)
    # ------------------------------------------------------------------
    def create_evaluation(self, experiment_id: str,
                          baseline_metrics: dict, candidate_metrics: dict,
                          per_seed: dict | None = None,
                          sandbox_risk_score: float = 0.0) -> EvaluationResult:
        delta = {}
        for k in set(baseline_metrics) | set(candidate_metrics):
            b = baseline_metrics.get(k, 0)
            c = candidate_metrics.get(k, 0)
            if isinstance(b, (int, float)) and isinstance(c, (int, float)):
                delta[k] = c - b
            else:
                delta[k] = "n/a"

        gate_result = self.promotion_gate.evaluate(
            baseline_metrics, candidate_metrics,
            current_level="baseline", per_seed=per_seed,
            sandbox_risk_score=sandbox_risk_score,
        )

        ev = EvaluationResult(
            id=f"eval_{uuid.uuid4().hex[:8]}",
            experiment_id=experiment_id,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            delta=delta,
            decision=gate_result.decision,
        )
        self._save("evaluations", ev.id, ev.to_dict())
        if self._seekdb is not None:
            with contextlib.suppress(Exception):
                self._seekdb.insert("auto_results", {
                    "id": ev.id, "experiment_id": experiment_id,
                    "decision": ev.decision, "delta": delta,
                    "created_at": datetime.now(UTC).isoformat(),
                })
        return ev

    # ------------------------------------------------------------------
    # Champion & DeadEnd (Sprint D: integrated store)
    # ------------------------------------------------------------------
    def promote_champion(self, skill_id: str, task_id: str, level: str,
                         metrics: dict, parent_skill: str = "", patch_id: str = "",
                         experiment_id: str = "") -> Champion:
        champ = Champion(
            id=f"champ_{uuid.uuid4().hex[:8]}",
            skill_id=skill_id,
            task_id=task_id,
            level=level,
            parent_skill_id=parent_skill,
            patch_id=patch_id,
            metrics=metrics,
            experiment_id=experiment_id,
        )
        self.champion_store.save_champion(champ)
        self.lineage.record(
            skill_id=skill_id, parent_skill=parent_skill,
            patch_id=patch_id, experiment_id=experiment_id,
            result="champion", metrics=metrics,
        )
        if self.publisher:
            self.publisher.champion_promoted(
                champion_id=champ.id, skill_id=skill_id,
                task_id=task_id, level=level, metrics=metrics,
            )
        if self._skill_registry is not None:
            try:
                from rosclaw.skill_manager.registry import SkillEntry
                self._skill_registry.register(SkillEntry(
                    name=skill_id,
                    description=f"Auto-evolved champion at level {level}",
                    skill_type="learned",
                    parameters=metrics,
                    metadata={
                        "parent_skill": parent_skill,
                        "patch_id": patch_id,
                        "experiment_id": experiment_id,
                        "level": level,
                    },
                ))
            except Exception:
                pass
        if self._seekdb is not None:
            with contextlib.suppress(Exception):
                self._seekdb.insert("champions", {
                    "id": champ.id, "skill_id": skill_id,
                    "task_id": task_id, "level": level,
                    "parent_skill": parent_skill, "metrics": metrics,
                    "promoted_at": datetime.now(UTC).isoformat(),
                })
        return champ

    def get_champion(self, task_id: str, level: str | None = None) -> Champion | None:
        return self.champion_store.get_champion(task_id, level)

    def list_champions(self, task_id: str | None = None) -> list[Champion]:
        return self.champion_store.list_champions(task_id)

    def register_deadend(self, task_id: str, direction: str,
                         rejection_reason: str, evidence: list[str] | None = None) -> DeadEnd:
        de = DeadEnd(
            id=f"de_{uuid.uuid4().hex[:8]}",
            task_id=task_id,
            direction=direction,
            rejection_reason=rejection_reason,
            evidence=evidence or [],
        )
        self._save("deadends", de.id, de.to_dict())
        if self.publisher:
            self.publisher.deadend_registered(
                deadend_id=de.id, task_id=task_id,
                direction=direction, rejection_reason=rejection_reason,
            )
        if self._seekdb is not None:
            with contextlib.suppress(Exception):
                self._seekdb.insert("dead_ends", {
                    "id": de.id, "task_id": task_id,
                    "direction": direction,
                    "rejection_reason": rejection_reason,
                    "registered_at": datetime.now(UTC).isoformat(),
                })
        return de

    def list_deadends(self, task_id: str | None = None) -> list[DeadEnd]:
        all_d = [DeadEnd.from_dict(d) for d in self._iterate("deadends")]
        if task_id:
            return [d for d in all_d if d.task_id == task_id]
        return all_d

    # ------------------------------------------------------------------
    # Rollback (Sprint D)
    # ------------------------------------------------------------------
    def rollback_skill(self, task_id: str, target_level: str | None = None) -> Champion | None:
        return self.rollback_manager.rollback(task_id, target_level)

    def get_lineage(self, skill_id: str) -> list:
        return self.lineage.get_lineage(skill_id)

    def render_lineage_tree(self, root_skill: str) -> str:
        return self.lineage.render_tree(root_skill)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def generate_report(self, task_id: str) -> EvolutionReport:
        task = self.get_task(task_id)
        task_name = task.name if task else task_id
        proposals = len(self.list_proposals(task_name))
        experiments = len([e for e in self._iterate("experiments")
                           if ExperimentSpec.from_dict(e).task in (task_id, task_name)])
        champs = len(self.champion_store.list_champions(task_id))
        deads = len(self.list_deadends(task_id))
        report = EvolutionReport(
            id=f"report_{uuid.uuid4().hex[:8]}",
            task_id=task_id,
            summary=f"Auto evolution for {task_id}",
            proposals_created=proposals,
            experiments_run=experiments,
            champions_promoted=champs,
            deadends_registered=deads,
        )
        self._save("reports", report.id, report.to_dict())
        return report

    # ------------------------------------------------------------------
    # High-level workflow (Sprint C+D integrated)
    # ------------------------------------------------------------------
    def run(self, task_id: str, rounds: int = 10, dry_run: bool = False) -> EvolutionReport:
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        for i in range(rounds):
            failures = self.list_failures(task_id)
            if not failures:
                break
            latest = failures[-1]

            from ..diagnosis.rule_diagnoser import RuleDiagnoser
            diagnoser = RuleDiagnoser()
            diag = diagnoser.diagnose(latest)
            search_space = diag.recommended_search_space if diag.recommended_search_space else {"param_range": [0.0, 1.0]}

            prop = self.create_proposal(
                latest.id, task.name, task.target_skill_id,
                f"Auto-generated repair for {latest.failure_mode}: {diag.root_cause_candidates}",
                search_space,
            )
            if dry_run:
                continue

            changes = []
            for param, range_vals in search_space.items():
                if isinstance(range_vals, (list, tuple)) and len(range_vals) == 2:
                    midpoint = (range_vals[0] + range_vals[1]) / 2
                    changes.append({
                        "path": f"/skill/{param}",
                        "old": None,
                        "new": round(midpoint, 4),
                    })

            patch = self.create_patch(prop.id, task.target_skill_id, changes)
            exp = self.create_experiment(prop.id, patch.id, task.name,
                                          task.target_skill_id, f"{task.target_skill_id}_candidate")

            raw_local = self.run_experiment(exp, runner="local")
            if not raw_local["success"]:
                self.register_deadend(task_id, f"round_{i}", "Local smoke test failed", [raw_local.get("error", "")])
                continue

            raw_sandbox = self.run_experiment(exp, runner="sandbox")
            if not raw_sandbox["success"]:
                self.register_deadend(task_id, f"round_{i}", "Sandbox safety check failed",
                                      raw_sandbox.get("safety_violations", []))
                continue

            raw_darwin = self.run_experiment(exp, runner="darwin")
            if not raw_darwin["success"]:
                self.register_deadend(task_id, f"round_{i}", "Darwin benchmark failed", [raw_darwin.get("error", "")])
                continue

            b_metrics = raw_darwin["metrics"].get("baseline", {})
            c_metrics = raw_darwin["metrics"].get("candidate", {})
            per_seed = raw_darwin["metrics"].get("per_seed")
            sandbox_risk = 0.0 if not raw_sandbox.get("safety_violations") else 0.5

            eval_res = self.create_evaluation(exp.id, b_metrics, c_metrics, per_seed, sandbox_risk)

            if eval_res.decision.startswith("promote"):
                level = eval_res.decision.replace("promote_to_", "")
                self.promote_champion(
                    f"{task.target_skill_id}_{level}", task_id, level,
                    eval_res.candidate_metrics, task.target_skill_id, patch.id, exp.id
                )
            elif eval_res.decision == "reject":
                self.register_deadend(task_id, f"round_{i}_{patch.id}",
                                      "Candidate worse than baseline",
                                      [str(eval_res.delta)])

        return self.generate_report(task_id)

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------
    def _save(self, namespace: str, key: str, data: dict) -> None:
        self.store.save(namespace, key, data)

    def _load(self, namespace: str, key: str) -> dict | None:
        return self.store.load(namespace, key)

    def _iterate(self, namespace: str):
        return self.store.iterate(namespace)
