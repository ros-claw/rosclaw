"""Team league E/F benchmark tests (PR-TF-075 功能核）.

断言（总纲 §10.9/§17.2 Team 指标）：
- 无故障：E/F 都完成全部任务
- 成员失联：F 经重公告完成 > E（E 的任务永久停滞）
- F 的双重分配恒为 0（DB 级 lease CAS）
- 恢复时间可度量且有限
"""

from __future__ import annotations

from pathlib import Path

from rosclaw.agentd.mission import MissionStore
from rosclaw.team.league import (
    LeagueMember,
    LeagueTask,
    run_group_e,
    run_group_f,
)

MEMBERS = [
    LeagueMember("r1", ("nav.local", "inspect.a"), speed=1.0),
    LeagueMember("r2", ("nav.local", "inspect.a"), speed=1.2),
    LeagueMember("r3", ("nav.local",), speed=0.8),
]
TASKS = [
    LeagueTask("t1", "nav.local", 1000),
    LeagueTask("t2", "inspect.a", 2000),
    LeagueTask("t3", "nav.local", 1500),
    LeagueTask("t4", "inspect.a", 800),
    LeagueTask("t5", "nav.local", 1200),
]


def _conn(tmp_path: Path, name: str = "league.db"):
    return MissionStore(tmp_path / name).connection


class TestLeague:
    def test_no_fault_both_complete(self, tmp_path: Path) -> None:
        e = run_group_e(MEMBERS, TASKS, seed=1)
        f = run_group_f(_conn(tmp_path), MEMBERS, TASKS, seed=1, team_id="lg1")
        assert e.tasks_completed == len(TASKS)
        assert f.tasks_completed == len(TASKS)
        assert f.double_assignments == 0
        assert f.role_conflicts == 0

    def test_member_loss_f_outperforms_e(self, tmp_path: Path) -> None:
        e = run_group_e(MEMBERS, TASKS, seed=2, lost_member="r1", lost_at_ms=500)
        f = run_group_f(
            _conn(tmp_path), MEMBERS, TASKS, seed=2, team_id="lg2",
            lost_member="r1", lost_at_ms=500,
        )
        # E：r1 名下的任务永久停滞；F：重公告后被其他成员接走。
        assert f.tasks_completed > e.tasks_completed
        assert f.double_assignments == 0
        assert f.recovery_ms is not None and f.recovery_ms >= 0

    def test_deterministic_per_seed(self, tmp_path: Path) -> None:
        f1 = run_group_f(_conn(tmp_path, "a.db"), MEMBERS, TASKS, seed=7, team_id="lg3")
        f2 = run_group_f(_conn(tmp_path, "b.db"), MEMBERS, TASKS, seed=7, team_id="lg3")
        assert f1.tasks_completed == f2.tasks_completed
        assert f1.makespan_ms == f2.makespan_ms

    def test_multi_seed_summary(self, tmp_path: Path) -> None:
        for seed in range(5):
            f = run_group_f(
                _conn(tmp_path, f"s{seed}.db"), MEMBERS, TASKS, seed=seed,
                team_id=f"lg{seed}", lost_member="r2" if seed % 2 else None,
                lost_at_ms=800,
            )
            assert f.double_assignments == 0
            assert f.tasks_completed >= len(TASKS) - 1
