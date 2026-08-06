"""Benchmark harness (PR-EV-080): scenarios, seeds, groups, metrics.

One scenario = one bounded task family. Runs are deterministic under a
seed (scripted gateways), or live against a real profile (integration).
Metrics follow 总纲 §17.2: mission success rate (per family), verification
coverage, unsupported-success-claim rate (target 0), tokens/cost, turns.
Baseline groups (§17.3): A = native-only, B = native + native workers.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rosclaw.agentd.config import load_agent_config
from rosclaw.agentd.models.gateway import MockModelGateway
from rosclaw.agentd.models.profiles import mock_profile
from rosclaw.agentd.service import AgentService
from rosclaw.contracts.agent.model_turn import ModelTurnResultV1

GROUP_A_NATIVE_ONLY = "A"
GROUP_B_NATIVE_WORKERS = "B"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    family: str
    goal: str
    user_turns: tuple[str, ...]
    script: tuple  # MockModelGateway script items (shared across seeds here)
    expect_state: str = "IDLE"
    expect_tool_rounds_min: int = 0
    expect_delegation: bool = False


@dataclass
class RunMetrics:
    scenario_id: str
    seed: int
    group_id: str
    success: bool
    final_state: str
    tool_rounds: int
    model_turns: int
    tokens_used: int
    cost_microunits: int
    unsupported_success_claim: bool
    delegation_accepted: bool | None = None
    notes: tuple[str, ...] = ()


_UNSUPPORTED_MARKERS = ("已完成真实动作", "已执行真实", "动作已完成", "receipt 已确认")


def _turn_from(content: str) -> ModelTurnResultV1:
    return ModelTurnResultV1(
        turn_id="bench_t",
        provider="mock",
        model="mock-model",
        content=content,
        assistant_message={"role": "assistant", "content": content},
        usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},  # type: ignore[arg-type]
    )


def answer_script(summary: str):
    def make(request) -> ModelTurnResultV1:
        decision = {
            "schema_version": "rosclaw.decision.v1",
            "decision_id": "dec_bench",
            "mission_id": request.mission_id,
            "context_id": request.context_id,
            "context_revision": request.context_revision,
            "next_intent": "ANSWER",
            "summary": summary,
            "evidence_refs": [],
        }
        return _turn_from(f"{summary}\n```json\n{json.dumps(decision)}\n```")

    return make


def hire_script(summary: str, goal: str):
    def make(request) -> ModelTurnResultV1:
        decision = {
            "schema_version": "rosclaw.decision.v1",
            "decision_id": "dec_bench_h",
            "mission_id": request.mission_id,
            "context_id": request.context_id,
            "context_revision": request.context_revision,
            "next_intent": "HIRE_WORKER",
            "summary": summary,
            "evidence_refs": ["artifact://bench/1"],
            "proposed_operation": {
                "type": "create_work_order",
                "payload": {"goal": goal, "capability": "analysis.text"},
            },
            "verification": {
                "schema_version": "rosclaw.decision_verification.v1",
                "verifiers": ["deterministic:schema"],
            },
        }
        return _turn_from(f"```json\n{json.dumps(decision)}\n```")

    return make


def worker_answer_script(text: str):
    def make(request) -> ModelTurnResultV1:
        return _turn_from(text)

    return make


def default_scenarios() -> list[Scenario]:
    return [
        Scenario(
            scenario_id="chat_answer",
            family="cognition",
            goal="解释身体与边界",
            user_turns=("你的身体是什么？不能做什么？",),
            script=(answer_script("我是绑定 sim/ur5e 的 ROSClaw 认知代理，不能直连硬件。"),),
        ),
        Scenario(
            scenario_id="delegation_analysis",
            family="delegation",
            goal="委派日志分析",
            user_turns=("把这段日志委派给 worker 分析",),
            script=(
                hire_script("委派分析", "分析失败日志"),
                worker_answer_script("根因：超时配置过短 [推断]"),
            ),
            expect_delegation=True,
        ),
    ]


class BenchmarkRunner:
    def __init__(self, home_factory, *, reporter_dir: Path | None = None) -> None:
        """``home_factory(seed) -> Path`` gives each run an isolated home."""
        self._home_factory = home_factory
        self._reporter_dir = reporter_dir

    async def run_once(self, scenario: Scenario, *, seed: int, group_id: str) -> RunMetrics:
        home = self._home_factory(seed)
        config = load_agent_config(home / "config.yaml")
        script = list(scenario.script)
        gateway = MockModelGateway(mock_profile(), script)
        service = AgentService(config, home, gateway=gateway)
        if group_id == GROUP_A_NATIVE_ONLY:
            service._handlers = None  # no worker channel in group A
            service._loops = {}
        try:
            mission = service.create_mission(scenario.goal)
            tool_rounds = 0
            model_turns = 0
            tokens = 0
            unsupported = False
            final_state = mission.state.value
            last_reply = ""
            for text in scenario.user_turns:
                result = await service.send_turn(mission.mission_id, text)
                tool_rounds += result.tool_rounds
                model_turns += result.model_turns
                tokens += result.tokens_used
                last_reply = result.reply
                final_state = result.state.value
                unsupported = unsupported or any(m in result.reply for m in _UNSUPPORTED_MARKERS)
            delegation_accepted = None
            if scenario.expect_delegation and group_id != GROUP_A_NATIVE_ONLY:
                orders = service._worker_manager.orders_for_mission(mission.mission_id)
                delegation_accepted = bool(orders) and orders[0].status == "ACCEPTED"
            usage = service.mission_usage(mission.mission_id)
            success = (
                final_state == scenario.expect_state
                and tool_rounds >= scenario.expect_tool_rounds_min
                and not unsupported
                and (not scenario.expect_delegation or delegation_accepted is True)
            )
            return RunMetrics(
                scenario_id=scenario.scenario_id,
                seed=seed,
                group_id=group_id,
                success=success,
                final_state=final_state,
                tool_rounds=tool_rounds,
                model_turns=model_turns,
                tokens_used=tokens,
                cost_microunits=usage["cost_microunits"],
                unsupported_success_claim=unsupported,
                delegation_accepted=delegation_accepted,
                notes=() if success else (f"reply: {last_reply[:120]}",),
            )
        finally:
            await service.close()

    async def run_matrix(
        self,
        scenarios: list[Scenario],
        *,
        seeds: list[int],
        groups: list[str],
    ) -> list[RunMetrics]:
        results: list[RunMetrics] = []
        for scenario in scenarios:
            for group in groups:
                for seed in seeds:
                    metrics = await self.run_once(scenario, seed=seed, group_id=group)
                    results.append(metrics)
                    self._persist(metrics)
        return results

    def _persist(self, metrics: RunMetrics) -> None:
        if self._reporter_dir is None:
            return
        self._reporter_dir.mkdir(parents=True, exist_ok=True)
        path = (
            self._reporter_dir / f"{metrics.scenario_id}_{metrics.group_id}_seed{metrics.seed}.json"
        )
        path.write_text(
            json.dumps(metrics.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def aggregate(results: list[RunMetrics]) -> dict:
    """Per (scenario, group): SR, tokens mean/stdev, unsupported rate."""
    by_key: dict[tuple[str, str], list[RunMetrics]] = {}
    for r in results:
        by_key.setdefault((r.scenario_id, r.group_id), []).append(r)
    report: dict[str, dict] = {}
    for (scenario, group), rows in sorted(by_key.items()):
        n = len(rows)
        tokens = [r.tokens_used for r in rows]
        report[f"{scenario}/{group}"] = {
            "runs": n,
            "success_rate": sum(r.success for r in rows) / n,
            "unsupported_claim_rate": sum(r.unsupported_success_claim for r in rows) / n,
            "tokens_mean": statistics.fmean(tokens),
            "tokens_stdev": statistics.stdev(tokens) if n > 1 else 0.0,
            "delegation_accept_rate": (
                sum(r.delegation_accepted for r in rows if r.delegation_accepted is not None)
                / max(sum(r.delegation_accepted is not None for r in rows), 1)
            ),
        }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "families": report,
        "total_runs": len(results),
    }
