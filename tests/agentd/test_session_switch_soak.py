"""T-SESSION（二次审计 NA-FIX-2 收尾）：100 次绑定/lease 切换无 split-brain。

每一时刻只有一个 writer；旧 lease 正确释放；无孤儿 ACTIVE binding。
"""

from __future__ import annotations

from pathlib import Path

from rosclaw.agentd.mission import MissionStore
from rosclaw.agentd.pi_bridge.session_binding import BindingError, SessionBindingStore


def test_hundred_switches_single_writer(tmp_path: Path) -> None:
    store = MissionStore(tmp_path / "m.db")
    bindings = SessionBindingStore(store.connection)
    active_tokens: dict[str, str] = {}
    for i in range(100):
        session_id = f"pi_sess_{i % 4}"  # 4 个 session 轮转绑定两个 mission
        mission_id = f"mis_{i % 2}"
        binding = bindings.bind(
            pi_session_id=session_id, pi_session_path="", mission_id=mission_id,
            body_id="b", execution_mode="SIMULATION", created_by="user:local:1000",
        )
        assert binding.status == "ACTIVE"
        # 同 session 重绑：幂等返回或降旧；ACTIVE binding 全局最多 4 个（每 session 1 个）。
        lease, token = bindings.acquire_lease(
            mission_id=mission_id, pi_session_id=session_id, owner_pid=100 + i, owner_uid=1000
        )
        writer = bindings.writer_of(mission_id)
        assert writer is not None and writer.pi_session_id == session_id
        active_tokens[mission_id] = token
        # 换 session 抢同一 mission：WRITER_HELD。
        try:
            bindings.acquire_lease(
                mission_id=mission_id, pi_session_id=f"pi_rival_{i}", owner_pid=900 + i,
                owner_uid=1000,
            )
        except BindingError as exc:
            assert exc.code == "WRITER_HELD"
        else:
            raise AssertionError(f"iteration {i}: rival session took the writer lease")
        # 释放后新 session 可获得。
        assert bindings.release_lease(mission_id, session_id, token)
    # 终态：每个 session 至多一个 ACTIVE binding。
    rows = store.connection.execute(
        "SELECT pi_session_id, COUNT(*) c FROM pi_session_bindings "
        "WHERE status = 'ACTIVE' GROUP BY pi_session_id"
    ).fetchall()
    for row in rows:
        assert row["c"] == 1, f"split-brain: session {row['pi_session_id']} has {row['c']} ACTIVE bindings"
    store.close()
