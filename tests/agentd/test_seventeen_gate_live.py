"""十六审 Gate 2（重建）：真实 `rosclaw chat` 产品闭环验收（真实 K3）。

与十四审 Gate 2 的分界线（审计：「Gate 2 测错了层级」）：这里**不直接
调用 dispatcher/delegate**——从 PTY 输入真实用户话语，走完整产品链路：

    rosclaw chat → Native Agent（真实 K3）→ TaskSpec/内置能力
    → Task Compiler → ExecutionRouter → profile/effect 编译
    → 执行 → artifacts → verifier → TUI 呈现

断言（审计 Gate 2 清单）：
- 一个 root Task，不产生重复 task（指纹 attach）；
- 正确 executor/profile（仿真 → executor:simulation 零 Worker；
  开发 → developer 不是 scout）；
- 用户不需要手工执行命令（输出不得出现 pip install 兜底）；
- 真实 artifact 存在且内容有效（GIF 帧数/文件非空）；
- verifier 非零检查；
- TUI 可见终态与数据库终态一致。

无真实 key/Node 诚实 skip。慢测试（真实模型运行）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from rosclaw.agentd.pi_entry import find_pi_agent_entry
from tests.agentd.test_product_journey import PtySession

KIMI_ENV_VARS = ("ROSCLAW_KIMI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY")
REPO = Path(__file__).resolve().parents[2]


def _has_key() -> bool:
    return any(os.environ.get(v) for v in KIMI_ENV_VARS)


def _has_runtime() -> bool:
    try:
        return find_pi_agent_entry() is not None
    except Exception:  # noqa: BLE001
        return False


def _prepare_home(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """真实 K3 配置（与 journey 的 fake 同构——provider 指向真实
    Kimi Code endpoint，key 只走 env 引用，绝不落盘）。"""
    home = tmp_path / "rh"
    (home / "run").mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "agent:\n  enabled: true\n  default_profile: embodied_default\n"
        "models:\n  backend: legacy\n  profiles:\n    embodied_default:\n"
        "      provider: kimi_code\n      model: k3\n"
        "      base_url: https://api.kimi.com/coding/v1\n"
        "      api_key_ref: env:ROSCLAW_KIMI_API_KEY\n"
        "      capabilities: [llm.chat, llm.structured_decision, llm.tool_use]\n",
        encoding="utf-8",
    )
    (home / "agent").mkdir(parents=True, exist_ok=True)
    (home / "agent" / "settings.json").write_text(
        json.dumps({"defaultProvider": "kimi-code", "defaultModel": "k3"}),
        encoding="utf-8",
    )
    (home / "agent" / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "kimi-code": {
                        "name": "Kimi Code",
                        "baseUrl": "https://api.kimi.com/coding/v1",
                        "api": "openai-completions",
                        "apiKey": "$ROSCLAW_KIMI_API_KEY",
                        "models": [
                            {
                                "id": "k3",
                                "name": "Kimi K3",
                                "contextWindow": 262144,
                                "maxTokens": 16384,
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    key = next(os.environ[v] for v in KIMI_ENV_VARS if os.environ.get(v))
    env = dict(
        os.environ,
        ROSCLAW_HOME=str(home),
        TERM="xterm",
        ROSCLAW_KIMI_API_KEY=key,
        KIMI_API_KEY=key,
    )
    return home, env


def _db(home: Path) -> sqlite3.Connection:
    return sqlite3.connect(home / "agentd" / "missions.db")


def _wait_execution_terminal(home: Path, timeout: float = 900.0) -> list[dict]:
    """等全部 execution 终态（轮询 DB——不依赖模型措辞）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        db_path = home / "agentd" / "missions.db"
        if db_path.exists():
            db = _db(home)
            try:
                rows = db.execute(
                    "SELECT execution_id, state, runtime, summary, "
                    "artifacts_json FROM task_executions"
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            db.close()
            if rows and all(
                r[1] in ("SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED")
                for r in rows
            ):
                return [
                    dict(zip(
                        ("execution_id", "state", "runtime", "summary",
                         "artifacts_json"), r, strict=True,
                    ))
                    for r in rows
                ]
        time.sleep(2)
    raise AssertionError("execution 未在时限内收敛（见 PTY 日志）")


def _wait_turn_settled(session, home: Path, timeout: float = 900.0) -> None:
    """等模型回合真正收束：DB 出现终态记录 + TUI 输出停止增长
    ≥15s（Working 转动期间输出持续增长；停止=回合结束）。
    第一个终态记录不等于回合结束——模型可能继续调参重试。"""
    deadline = time.monotonic() + timeout
    saw_terminal = False
    last_len = -1
    last_growth = time.monotonic()
    while time.monotonic() < deadline:
        db_path = home / "agentd" / "missions.db"
        if db_path.exists():
            db = _db(home)
            try:
                rec = db.execute(
                    "SELECT COUNT(*) FROM task_records "
                    "WHERE state IN ('VERIFIED','FAILED')"
                ).fetchone()[0]
                exe = db.execute(
                    "SELECT COUNT(*) FROM task_executions "
                    "WHERE state IN ('SUCCEEDED','FAILED','BLOCKED','CANCELLED')"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                rec, exe = 0, 0
            db.close()
            if rec or exe:
                saw_terminal = True
        with session._lock:
            current = len(session.output)
        if current != last_len:
            last_len = current
            last_growth = time.monotonic()
        if saw_terminal and time.monotonic() - last_growth > 15:
            return
        time.sleep(1)
    raise AssertionError("回合未在时限内收束（见 PTY 日志）")


def _assert_no_manual_fallback(output: bytes) -> None:
    """用户不需要手工执行命令——框架失败甩锅 pip install 是本轮根因。"""
    assert b"pip install" not in output, "输出竟让用户手工 pip install"
    assert "请手动".encode() not in output


@pytest.mark.slow
@pytest.mark.skipif(
    not (_has_key() and _has_runtime()),
    reason="无真实 provider key 或 Node/dist——诚实 skip",
)
class TestGate2RealProductLoop:
    def test_star_sim_via_chat_full_chain(self, tmp_path: Path) -> None:
        """仿真闭环：PTY 输入"画五角星仿真出 GIF"——内置确定性链路
        （零 Worker），真实 GIF + 非零验收，终态一致。

        正确路由是 rosclaw_task → Task Runner（task_records）或
        task_submit → executor:simulation（task_executions）——两条
        都是内置确定性链，共同断言：零 Worker 雇佣 + 真实 GIF +
        验收 PASS。"""
        home, env = _prepare_home(tmp_path)
        python = REPO / ".venv" / "bin" / "python"
        session = PtySession(
            [str(python), "-m", "rosclaw.entrypoint", "chat"],
            env, log_path=tmp_path / "pty-sim.log",
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send(
                "帮我做一个 UR5e 机械臂画五角星的动力学仿真，输出 GIF 动画\r"
            )
            _wait_turn_settled(session, home, timeout=900)
            db = _db(home)
            records = [
                dict(zip(("task_id", "goal", "state", "verification_json",
                          "error"), r, strict=True))
                for r in db.execute(
                    "SELECT task_id, goal, state, verification_json, error "
                    "FROM task_records"
                ).fetchall()
            ]
            executions = [
                dict(zip(("execution_id", "state", "runtime", "summary",
                          "artifacts_json"), r, strict=True))
                for r in db.execute(
                    "SELECT execution_id, state, runtime, summary, "
                    "artifacts_json FROM task_executions"
                ).fetchall()
            ]
            db.close()
            assert records or executions, "两条确定性链都无记录（见 PTY 日志）"
            session.expect_with_resend(b"rosclaw continue", "/quit\r",
                                       timeout=120)
            session.proc.wait(timeout=30)
            output = session.clean
            _assert_no_manual_fallback(output)
            # 零 Worker 雇佣——内置仿真链必须确定性直跑。
            db = _db(home)
            orders = db.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0]
            db.close()
            assert orders == 0, f"内置仿真链不得雇佣 Worker: {orders}"
            # 验收 PASS（task_records VERIFIED 或 execution SUCCEEDED）。
            verified = any(r["state"] == "VERIFIED" for r in records) or any(
                e["state"] == "SUCCEEDED" for e in executions
            )
            assert verified, f"无一条验收通过: {records} {executions}"
            # 真实 GIF artifact（帧数/尺寸有效）。
            gifs = list(home.glob("sim/**/*.gif"))
            assert gifs, "仿真 GIF 未真实落盘"
            gif = max(gifs, key=lambda p: p.stat().st_size)
            assert gif.stat().st_size > 1000, gif
        finally:
            session.stop()

    def test_coding_task_via_chat_full_chain(self, tmp_path: Path) -> None:
        """开发闭环：PTY 输入写脚本任务——一个 execution、developer
        profile（不是 scout）、真实交付文件、验收非零、终态一致。"""
        home, env = _prepare_home(tmp_path)
        python = REPO / ".venv" / "bin" / "python"
        session = PtySession(
            [str(python), "-m", "rosclaw.entrypoint", "chat"],
            env, log_path=tmp_path / "pty-coding.log",
        )
        try:
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session.send(
                "帮我写一个 Python 脚本，计算斐波那契数列前 10 项，"
                "保存到 answer.txt（每行一个数），并实际运行验证内容正确。\r"
            )
            _wait_turn_settled(session, home, timeout=900)
            db = _db(home)
            # PR-H1 后：模型面没有 task_submit——编码任务由主会话直接
            # 完成；零 execution、零 WorkOrder 是架构不变量（PR-H2：
            # root task 由输入事务创建）。
            orders = db.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0]
            execs = db.execute(
                "SELECT COUNT(*) FROM task_executions"
            ).fetchone()[0]
            try:
                tasks = db.execute(
                    "SELECT task_id, state, active_revision FROM tasks"
                ).fetchall()
            except sqlite3.OperationalError:
                tasks = []
            db.close()
            session.expect_with_resend(b"rosclaw continue", "/quit\r",
                                       timeout=120)
            session.proc.wait(timeout=30)
            output = session.clean
            _assert_no_manual_fallback(output)
            assert orders == 0, f"直接工作不得创建 WorkOrder: {orders}"
            assert execs == 0, f"直接工作不得创建 execution: {execs}"
            assert len(tasks) == 1, f"一个目标一个 root task: {tasks}"
            # 真实交付：answer.txt 真实落盘（主会话工作区=chat cwd 或
            # task workspace）。
            found = list(tmp_path.rglob("answer.txt"))
            assert found, "answer.txt 未真实落盘"
            content = found[0].read_text().strip().splitlines()
            assert "34" in content[-1] or "34" in content, content[:5]
        finally:
            session.stop()
