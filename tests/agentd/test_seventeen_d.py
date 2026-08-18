"""十六审 PR-17.6 红测试：F2 单任务卡（P1 TUI 整改）。

红测试先行——修复前必须红：
1. pi.task.executions 卡片折叠**全部** attempts（repair/escalate 的
   历史 attempt 不再是游离 Worker 卡）——linked_ids 覆盖 root+所有
   attempt；
2. 卡片带编译 profile 与验收结果（用户可见的诚实面）。
"""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

from tests.agentd.test_pi_tool_bridge import _setup


async def _executions_cards(service, mission_id: str) -> list[dict]:
    from rosclaw.agentd.pi_bridge.server import PiBridgeServer

    server = PiBridgeServer(service, Path(service._home) / "pi-bridge.sock")
    result = await server._dispatch(
        "owner", 0, "pi.task.executions",
        {"mission_id": mission_id, "token": service.control_token},
    )
    assert result.get("ok"), result
    return list(result["executions"])


class TestExecutionCardFolding:
    async def test_card_folds_repair_attempts(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """一次验收失败 + REPAIRING 同 session 修复 → 卡片 1 张、
        attempts 2 条折叠、linked_ids 覆盖两单、profile/verifier 在卡上。"""
        from rosclaw.agentd.workers import pi_managed

        service, mission = await _setup(tmp_path)
        counter = tmp_path / "runs.count"
        fake = tmp_path / "fake-repair"
        fake.write_text(
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            f'echo \'{{"kind":"session_persisted","session_file":"{tmp_path}/fs.jsonl"}}\'\n'
            f"N=$(cat {counter} 2>/dev/null || echo 0)\n"
            f"echo $((N+1)) > {counter}\n"
            'if [ "$N" -ge 1 ]; then\n'
            "  echo x > deliverable.txt\n"
            '  echo \'{"kind":"attempt_finished","report":"修复完成"}\'\n'
            "else\n"
            '  echo \'{"kind":"attempt_finished","report":"初版完成"}\'\n'
            "fi\n"
        )
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        (tmp_path / "fs.jsonl").write_text("{}\n")
        monkeypatch.setattr(
            pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake))
        )
        adapter = pi_managed.PiManagedAdapter(
            rosclaw_home=tmp_path, conn=service._store.connection
        )
        service._worker_manager._adapters["pi_managed"] = adapter
        adapter._manager_ref = service._worker_manager
        if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
            service._registry.set_status(
                "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="fake"
            )
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {"goal": "实现功能", "required_capabilities": ["code.implement"],
             "effects": "workspace_only",
             "acceptance": {"required_files": ["deliverable.txt"]}},
            idem="17d_fold",
        )
        row = None
        for _ in range(1800):
            row = plane._get(view["execution_id"])
            if row["state"] in ("SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"):
                break
            await asyncio.sleep(0.05)
        assert row is not None and row["state"] == "SUCCEEDED", row["summary"]

        cards = await _executions_cards(service, mission.mission_id)
        assert len(cards) == 1, f"一个任务一张卡: {len(cards)}"
        card = cards[0]
        assert len(card["attempts"]) == 2, (
            f"repair 的历史 attempt 必须折叠进卡: {card['attempts']}"
        )
        linked = set(card["linked_ids"])
        assert len(linked) >= 2
        assert {a["work_order_id"] for a in card["attempts"]} <= linked
        assert card["profile"] == "developer", card["profile"]
        assert card["verifier"].get("verdict") == "PASS", card["verifier"]
        assert card["verifier"].get("checks", 0) >= 1
        await service.close()

    async def test_card_shows_sim_executor_without_worker(
        self, tmp_path: Path
    ) -> None:
        """仿真执行任务：卡上 runtime=executor:simulation，无 Worker
        attempt（内置确定性链路不雇佣 Worker）。"""
        service, mission = await _setup(tmp_path)
        plane = service._task_control_plane
        view = await plane.submit(
            mission.mission_id,
            {"goal": "画五角星动力学仿真",
             "required_capabilities": ["simulation.planar_trajectory"],
             "effects": "simulation_only",
             "inputs": {"shape": "star5"},
             "acceptance": {"max_tracking_error_m": 0.10}},
            idem="17d_sim",
        )
        for _ in range(1800):
            row = plane._get(view["execution_id"])
            if row["state"] in ("SUCCEEDED", "FAILED", "BLOCKED"):
                break
            await asyncio.sleep(0.05)
        cards = await _executions_cards(service, mission.mission_id)
        assert len(cards) == 1
        assert cards[0]["runtime"] == "executor:simulation"
        assert cards[0]["attempts"] == [], (
            f"内置仿真链不得出现 Worker 卡: {cards[0]['attempts']}"
        )
        assert cards[0]["state"] == "SUCCEEDED", cards[0]["summary"]
        await service.close()


class TestFuzzyAttach:
    def test_goal_similarity(self) -> None:
        from rosclaw.agentd.control_plane import _goal_similarity

        assert _goal_similarity(
            "编写一个 Python 脚本计算斐波那契数列前 10 项",
            "编写一个名为 fib.py 的 Python 脚本，计算斐波那契数列前 10 项",
        ) >= 0.80
        assert _goal_similarity("画五角星仿真", "写一个贪吃蛇游戏") < 0.80

    async def test_reworded_same_goal_attaches(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """同一目标换措辞重提 → attach（不裂变）；不同目标 → 新任务。
        驱动打桩立即返回（execution 停留 PREFLIGHT 活跃态）——测的是
        submit 的 attach 判定，不烧 Worker。"""
        service, mission = await _setup(tmp_path)
        plane = service._task_control_plane

        async def _noop_drive(*args, **kwargs):
            return None

        monkeypatch.setattr(plane, "_drive", _noop_drive)
        first = await plane.submit(
            mission.mission_id,
            {"goal": "编写一个 Python 脚本计算斐波那契数列前 10 项并保存",
             "effects": "workspace_only", "acceptance": {}},
            idem="17d_fuzzy1",
        )
        second = await plane.submit(
            mission.mission_id,
            {"goal": "编写一个名为 fib.py 的 Python 脚本，计算斐波那契前 10 项并保存",
             "effects": "workspace_only", "acceptance": {}},
            idem="17d_fuzzy2",
        )
        assert second.get("attached"), "换措辞重提同目标必须 attach（防裂变）"
        assert second["execution_id"] == first["execution_id"]
        third = await plane.submit(
            mission.mission_id,
            {"goal": "写一个完全不同的贪吃蛇游戏",
             "effects": "workspace_only", "acceptance": {}},
            idem="17d_fuzzy3",
        )
        assert not third.get("attached"), "不同目标不得 attach"
        assert third["execution_id"] != first["execution_id"]
        assert len(plane.executions_for(mission.mission_id)) == 2
        await service.close()
