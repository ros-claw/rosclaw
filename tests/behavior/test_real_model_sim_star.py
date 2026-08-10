"""PR-EIGHT-1（八审 §4 P0-1）：真实模型 Behavior Journey。

与 Protocol Journey（假模型硬编码工具链）并存、互不替代：

- 测试端只给一句自然语言目标——不写死 tool call ID/顺序/参数；
- 默认 provider 用真实 Kimi K3（env: ROSCLAW_KIMI_API_KEY /
  KIMI_API_KEY / MOONSHOT_API_KEY——无 key 时诚实 skip，绝不假绿）；
- 行为判定走 behavior_judge 唯一权威（与失败回归同一口径）；
- 会话指标（模型请求/工具调用/重试分类/token/延迟）落盘为
  artifact——清洗后无密钥，仅有结构化计数。

八审 §7.3 发布阻断：fake scripted journey 绿但真实 provider journey
红 → Gate 必须红。本测试默认 **不** 加 pytest.mark.slow 以外的豁免；
在 CI 中由 nightly/required 矩阵驱动。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from tests.agentd.test_product_journey import (
    PtySession,
    _build_and_install,
    _hidden_source_checkout,
)

KIMI_ENV_VARS = ("ROSCLAW_KIMI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY")


def _provider_key() -> tuple[str, str] | None:
    for var in KIMI_ENV_VARS:
        value = os.environ.get(var, "")
        if value:
            return var, value
    return None


def _write_real_home(home: Path, key_var: str) -> None:
    """真实 Kimi K3 配置——api_key 只用 env 引用（红线：绝不落盘）。"""
    (home / "run").mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "agent:\n  enabled: true\n  default_profile: kimi_k3\n"
        "models:\n  backend: legacy\n  profiles:\n    kimi_k3:\n"
        "      provider: kimi_code\n      model: k3\n"
        "      base_url: https://api.kimi.com/coding/v1\n"
        f"      api_key_ref: env:{key_var}\n"
        "      capabilities: [llm.chat, llm.structured_decision, llm.tool_use]\n",
        encoding="utf-8",
    )
    (home / "agent").mkdir(parents=True, exist_ok=True)
    (home / "agent" / "settings.json").write_text(
        json.dumps({"defaultProvider": "kimi-coding", "defaultModel": "k3"}),
        encoding="utf-8",
    )
    (home / "agent" / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "kimi-coding": {
                        "name": "Kimi Coding",
                        "baseUrl": "https://api.kimi.com/coding/v1",
                        "api": "openai-completions",
                        "apiKey": f"${key_var}",
                        "models": [
                            {
                                "id": "k3",
                                "name": "Kimi K3",
                                "contextWindow": 262144,
                                "maxTokens": 8192,
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _collect_metrics(home: Path, session: PtySession) -> dict:
    """从 missions.db + PTY 文本收集行为指标（结构化计数，无正文）。"""
    db = sqlite3.connect(home / "agentd" / "missions.db")
    try:
        model_requests = db.execute(
            "SELECT COUNT(*) FROM model_usage"
        ).fetchone()[0]
        action_proposals = db.execute(
            "SELECT COUNT(*) FROM operator_requests"
        ).fetchone()[0]
        events = [
            r[0]
            for r in db.execute(
                "SELECT type FROM agent_events ORDER BY rowid"
            ).fetchall()
        ]
        receipts = db.execute(
            "SELECT payload_json FROM agent_events WHERE type='receipt.received'"
        ).fetchall()
    finally:
        db.close()
    with session._lock:
        clean = session.clean.decode("utf-8", "replace")
    verifier_pass = "几何验证" in clean and (
        "PASS" in clean or "通过" in clean or "验证通过" in clean
    )
    return {
        "goal": "draw star",
        "user_messages": 1,
        "user_confirmations": clean.count("授权请求"),
        "model_requests": model_requests,
        "action_proposals": action_proposals,
        "observation_calls": events.count("tool.proposed"),  # 粗计数见 artifact
        "capability_queries": 0,
        "context_not_fresh_retries": clean.count("CONTEXT_NOT_FRESH"),
        "schema_retries": clean.count("INVALID_ARGUMENTS")
        + clean.count("trajectory required"),
        "hash_retries": clean.count("hash"),
        "lease_retries": clean.count("lease"),
        "task_completed": "完成" in clean,
        "verifier_pass": verifier_pass,
        "conflict_with_kernel": "冲突" in clean,
        "_event_counts": {e: events.count(e) for e in sorted(set(events))},
        "_receipt_count": len(receipts),
    }


@pytest.mark.slow
class TestRealModelBehavior:
    def test_sim_star_real_model(self, tmp_path: Path) -> None:
        key = _provider_key()
        if key is None:
            pytest.skip(
                "无真实 provider key（ROSCLAW_KIMI_API_KEY/KIMI_API_KEY/"
                "MOONSHOT_API_KEY）——Behavior Gate 诚实跳过，不假绿"
            )
        key_var, _ = key
        prefix, _root = _build_and_install(tmp_path)
        home = tmp_path / "rh"
        _write_real_home(home, key_var)
        rosclaw = prefix / "bin" / "rosclaw"
        env = dict(
            os.environ,
            ROSCLAW_HOME=str(home),
            TERM="xterm",
            ROSCLAW_UI_LOCALE="zh-CN",
            PATH=f"{prefix / 'bin'}:{os.environ['PATH']}",
        )
        session = PtySession(
            [str(rosclaw), "chat"], env, log_path=tmp_path / "pty-behavior.log"
        )
        started = time.monotonic()
        try:
            with _hidden_source_checkout():
                session.expect(b"ROSClaw Native Agent", timeout=120)
                session.send("我想用机械臂画个五角星\r")
                # 真实模型延迟不可控——等完成证据或失败信号，最长 10 分钟。
                deadline = time.monotonic() + 600
                done = False
                while time.monotonic() < deadline and not done:
                    with session._lock:
                        clean = session.clean
                    if (
                        "几何验证".encode() in clean
                        or "验证通过".encode() in clean
                        or "无法完成".encode() in clean
                        or "未能完成".encode() in clean
                    ):
                        done = True
                        break
                    if session.proc.poll() is not None:
                        break
                    time.sleep(2.0)
                time.sleep(2.0)  # 让最终回合落盘
                metrics = _collect_metrics(home, session)
                metrics["wall_seconds"] = round(time.monotonic() - started, 1)
                metrics["settled"] = done
                session.send("/quit\r")
        finally:
            session.stop()
        # artifact：脱敏指标（结构化计数，无密钥无正文）。
        (tmp_path / "behavior-metrics.json").write_text(
            json.dumps(metrics, indent=1, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        from rosclaw.agentd.behavior_judge import judge_session

        verdict = judge_session(metrics)
        (tmp_path / "behavior-verdict.json").write_text(
            json.dumps(verdict, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        assert verdict["verdict"] == "PASS", (
            f"真实模型 Behavior Gate FAIL: {verdict['violations']} "
            f"metrics={metrics}"
        )
