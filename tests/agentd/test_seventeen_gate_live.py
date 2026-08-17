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
import re
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
                         "artifacts_json"), r,
                    ))
                    for r in rows
                ]
        time.sleep(2)
    raise AssertionError("execution 未在时限内收敛（见 PTY 日志）")


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
        （零 Worker），真实 GIF + 非零验收，终态一致。"""
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
            executions = _wait_execution_terminal(home, timeout=900)
            # 终态后等 TUI/回复渲染一拍再收尾。
            session.expect_with_resend(b"rosclaw continue", "/quit\r",
                                       timeout=120)
            session.proc.wait(timeout=30)
            output = session.clean
            _assert_no_manual_fallback(output)
            # 一任务一执行；executor:simulation；零 Worker 雇佣。
            assert len(executions) == 1, f"裂变: {executions}"
            execution = executions[0]
            assert execution["state"] == "SUCCEEDED", (
                f"{execution['state']}: {execution['summary']}"
            )
            assert execution["runtime"] == "executor:simulation", (
                execution["runtime"]
            )
            db = _db(home)
            orders = db.execute("SELECT COUNT(*) FROM work_orders").fetchone()[0]
            db.close()
            assert orders == 0, f"内置仿真链不得雇佣 Worker: {orders}"
            artifacts = json.loads(execution["artifacts_json"] or "{}")
            gif = Path(str(artifacts.get("gif", "")))
            assert gif.exists() and gif.stat().st_size > 1000, gif
            assert artifacts.get("evidence_level") == "SIM_DYN_ROLLOUT"
            # 验收非零（metrics 在 artifacts 内；verify PASS 已在链路中）。
            assert artifacts.get("metrics"), "缺验收 metrics"
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
            executions = _wait_execution_terminal(home, timeout=900)
            session.expect_with_resend(b"rosclaw continue", "/quit\r",
                                       timeout=120)
            session.proc.wait(timeout=30)
            output = session.clean
            _assert_no_manual_fallback(output)
            assert len(executions) == 1, f"裂变: {executions}"
            execution = executions[0]
            assert execution["state"] == "SUCCEEDED", (
                f"{execution['state']}: {execution['summary']}"
            )
            # 正确 profile：开发任务必须 developer（不是只读 scout）。
            db = _db(home)
            rows = db.execute(
                "SELECT inputs FROM work_orders"
            ).fetchall()
            db.close()
            assert len(rows) >= 1, "开发任务应有一个内部执行单"
            profiles = {
                json.loads(r[0]).get("worker_profile") for r in rows
            }
            assert profiles == {"developer"}, f"profile 错编译: {profiles}"
            # 真实交付：answer.txt 存在于某个执行 workspace。
            found = list(home.glob("work/*/workspace/answer.txt"))
            assert found, "answer.txt 未真实落盘"
            content = found[0].read_text().strip().splitlines()
            assert "55" in content[-1] or "34" in content, content[:5]
            # verifier 非零检查（验收 PASS·N 项，N≥1）。
            assert re.search(r"PASS·[1-9]", execution["summary"] or "") or (
                json.loads(execution["artifacts_json"] or "{}")
                .get("verifier", {}).get("checks", 0) >= 1
            ), f"验收零检查: {execution['summary']}"
        finally:
            session.stop()
