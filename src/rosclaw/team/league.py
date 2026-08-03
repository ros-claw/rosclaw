"""Team league benchmark (PR-TF-075 精简版, 总纲 §10.9 T-SIM-3 功能核）.

E/F 基线对比（§17.3）：
- E：多机器人各自为政（静态分配，无 epoch/lease/重协调）
- F：Team Fabric（coordinator + role lease + requeue + 降级矩阵）

指标：任务完成数、makespan（模拟时钟）、双重分配（目标 0）、成员失联
恢复时间、分区期间安全退化。全部 local_sim 确定性，不宣称真实增益
——3v3 联赛（多 seed 固定场景集）留待 PR-TF-075 完整版。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rosclaw.contracts.team.member import MemberBody, TeamMemberCardV1
from rosclaw.team import TeamCoordinator
from rosclaw.team.allocator import Bid, TaskAnnouncement


@dataclass(frozen=True)
class LeagueTask:
    task_id: str
    required_capability: str
    duration_ms: int


@dataclass(frozen=True)
class LeagueMember:
    member_id: str
    capabilities: tuple[str, ...]
    speed: float = 1.0


@dataclass
class LeagueResult:
    group: str
    seed: int
    tasks_total: int
    tasks_completed: int
    makespan_ms: int
    double_assignments: int
    role_conflicts: int
    recovery_ms: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def _card(member: LeagueMember, team_id: str) -> TeamMemberCardV1:
    return TeamMemberCardV1(
        team_id=team_id,
        member_id=member.member_id,
        body=MemberBody(**{"body_id": member.member_id, "effective_body_hash": "h", "class": "mb"}),
        capabilities=list(member.capabilities),
    )


def run_group_e(
    members: list[LeagueMember],
    tasks: list[LeagueTask],
    *,
    seed: int,
    lost_member: str | None = None,
    lost_at_ms: int = 0,
) -> LeagueResult:
    """E 组：静态轮询分配；成员失联后其任务永久停滞（无重协调）。"""
    clock = 0
    completed = 0
    stalled = 0
    for index, task in enumerate(tasks):
        owner = members[index % len(members)]
        clock += task.duration_ms // owner.speed
        if lost_member is not None and owner.member_id == lost_member and clock >= lost_at_ms:
            stalled += 1
            continue
        if task.required_capability in owner.capabilities:
            completed += 1
        else:
            stalled += 1
    return LeagueResult(
        group="E",
        seed=seed,
        tasks_total=len(tasks),
        tasks_completed=completed,
        makespan_ms=int(clock),
        double_assignments=0,
        role_conflicts=0,
        notes=(f"{stalled} tasks stalled" if stalled else ()),
    )


def run_group_f(
    conn,
    members: list[LeagueMember],
    tasks: list[LeagueTask],
    *,
    seed: int,
    team_id: str,
    lost_member: str | None = None,
    lost_at_ms: int = 0,
) -> LeagueResult:
    """F 组：TeamCoordinator 分配；成员失联 → 无副作用任务重公告 → 重分配。"""
    coord = TeamCoordinator(conn, team_id=team_id, actor_id="league", policy_hash="league_pol")
    for member in members:
        coord.join_member(_card(member, team_id))
    clock = 0
    completed = 0
    double_assignments = 0
    recovery_ms: int | None = None
    holders: dict[str, str] = {}
    pending = list(tasks)
    index = 0
    while pending:
        task = pending.pop(0)
        epoch = coord.epoch()
        # 成员失联注入：先 sweep + requeue，再继续。
        if lost_member is not None and clock >= lost_at_ms and recovery_ms is None:
            conn.execute(
                "UPDATE team_members SET last_seen_at = ? WHERE member_id = ? AND team_id = ?",
                ("2000-01-01", lost_member, team_id),
            )
            coord.membership.sweep_ttl(suspect_after_ms=1, lost_after_ms=2)
            coord.member_lost(lost_member)
            recovery_start = clock
            alive = [m for m in members if m.member_id != lost_member]
        else:
            alive = [m for m in members if m.member_id != lost_member] if lost_member else members
            recovery_start = None
        bids = []
        for member in alive:
            fit = 1.0 if task.required_capability in member.capabilities else 0.0
            if fit > 0:
                bids.append(
                    Bid(
                        member_id=member.member_id,
                        eta_ms=task.duration_ms / member.speed,
                        energy_cost=task.duration_ms / 100.0,
                        capability_fit=fit,
                        reliability=0.9,
                        current_load=0.0,
                        comms_quality=1.0,
                    )
                )
        if not bids:
            break
        ann = TaskAnnouncement(
            task_id=task.task_id,
            team_id=team_id,
            team_epoch=epoch,
            required_capabilities=(task.required_capability,),
            idempotency_key=f"{team_id}:{seed}:{task.task_id}",
        )
        _, winner = coord.announce_and_award(ann, bids)
        if winner in holders.get(task.task_id, ""):
            double_assignments += 1
        holders[task.task_id] = winner
        coord.accept_task(task.task_id, winner)
        clock += int(task.duration_ms / next(m.speed for m in alive if m.member_id == winner))
        coord.complete_task(
            task.task_id, winner, evidence={"summary": "league sim", "receipt_ref": "sim://1"}
        )
        completed += 1
        if recovery_start is not None and recovery_ms is None:
            recovery_ms = clock - recovery_start
        index += 1
    # 角色双重持有检查（DB 约束应使其恒为 0）。
    active = coord.roles.active_leases()
    seen: set[str] = set()
    role_conflicts = 0
    for lease in active:
        if lease.conflict_key in seen:
            role_conflicts += 1
        seen.add(lease.conflict_key)
    return LeagueResult(
        group="F",
        seed=seed,
        tasks_total=len(tasks),
        tasks_completed=completed,
        makespan_ms=int(clock),
        double_assignments=double_assignments,
        role_conflicts=role_conflicts,
        recovery_ms=recovery_ms,
    )
