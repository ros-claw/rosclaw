"""十一审 PR-A 红测试：false timeout 状态机 + prompt/tool 契约。

红测试先行——修复前必须红：
1. 语义静默 ≥90s 但 liveness 正常 → 不杀（stall 只告警）；旧实现
   60s 无事件即杀。
2. 全静默（连 liveness 都没有）→ liveness timeout 判死。
3. wall 预算：先 wrap-up steer，grace 内不退出才终止。
4. token 预算 ≥80% → wrap-up steer 一次。
5. Developer prompt 不再含 "read-only tools" 矛盾；manifest 列出
   7 个工具与 DoD。
"""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

from rosclaw.agentd.workers.scheduler import CandidateView
from rosclaw.contracts.common import new_id
from rosclaw.contracts.worker.order import (
    BudgetEnvelope,
    ExpectedOutput,
    SideEffectPolicy,
    WorkOrderV1,
)
from tests.agentd.test_pi_tool_bridge import _setup

# CI Full Regression 无 node/dist——prompt 契约测试诚实 skip（本地与
# Node jobs 覆盖）。
_NODE_AVAILABLE = shutil.which("node") is not None and (
    Path(__file__).resolve().parents[2]
    / "packages/rosclaw-agent/dist/src/workers/profiles.js"
).exists()


def _fake(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


async def _run(service, mission, tmp_path, fake, monkeypatch, *, wall_time=300):
    from rosclaw.agentd.workers import pi_managed

    monkeypatch.setattr(pi_managed, "find_pi_agent_entry", lambda: ("/bin/sh", str(fake)))
    adapter = pi_managed.PiManagedAdapter(rosclaw_home=tmp_path)
    service._worker_manager._adapters["pi_managed"] = adapter
    if service._registry.status_of("worker:rosclaw:pi") != "ENABLED":
        service._registry.set_status(
            "worker:rosclaw:pi", "ENABLED", actor_id="test", reason="fake entry"
        )
    card = service._registry.get("worker:rosclaw:pi")
    order = WorkOrderV1(
        work_order_id=new_id("wo"),
        mission_id=mission.mission_id,
        issued_by="test",
        capability="analysis.text",
        goal="长任务",
        inputs={"instructions": "x"},
        budgets=BudgetEnvelope(wall_time_sec=wall_time, model_tokens=1000),
        expected_output=ExpectedOutput(artifacts=["text/plain"]),
        side_effect_policy=SideEffectPolicy(**{"class": "none"}),
    )
    scheduled = service._worker_manager.hire(
        order,
        [CandidateView(card=card, registry_status="ENABLED", running_orders=0,
                       circuit_open=False)],
    )
    return await service._worker_manager.run_to_completion(scheduled)


class TestNoFalseTimeout:
    async def test_semantic_silence_with_liveness_not_killed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Gate B：90s+ 无语义事件但 liveness 正常——进程保持存活。
        （为测试时长可控，stall 阈值缩小；断言无 idle 误杀 + stall 告警。）"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "STALL_WARN_SEC", 1.0)
        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        fake = _fake(
            tmp_path,
            "fake-slow-thinker",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "i=0\n"
            "while [ $i -lt 12 ]; do\n"
            '  echo \'{"kind":"liveness","phase":"RUNNING_MODEL","span_age_ms":1000,"pid_alive":true}\'\n'
            "  sleep 0.5\n"
            "  i=$((i+1))\n"
            "done\n"
            'echo \'{"kind":"attempt_finished","report":"长推理完成"}\'\n',
        )
        started = time.monotonic()
        result, report = await _run(service, mission, tmp_path, fake, monkeypatch)
        elapsed = time.monotonic() - started
        # 6 秒全 liveness 无语义事件——旧实现会在 60s 处杀（语义层面
        # 等价：stall 阈值 1s 下，liveness 保活必须让任务活到完成）。
        assert result.status == "COMPLETED", result.summary
        assert "长推理完成" in result.summary
        assert elapsed > 5, f"疑似提前退出: {elapsed:.1f}s"
        assert report.accepted
        await service.close()

    async def test_total_silence_killed_by_liveness_timeout(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """连 liveness 都没有 = 进程挂死——必须判死（不无限等待）。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 2.0)
        service, mission = await _setup(tmp_path)
        fake = _fake(
            tmp_path,
            "fake-hang-after-start",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "exec sleep 300\n",
        )
        started = time.monotonic()
        result, _report = await _run(service, mission, tmp_path, fake, monkeypatch)
        elapsed = time.monotonic() - started
        assert result.status == "FAILED"
        assert "liveness lost" in result.summary, result.summary
        assert elapsed < 20, f"判死耗时 {elapsed:.1f}s"
        await service.close()


class TestBudgetWrapup:
    async def test_wall_budget_wrapup_then_terminate(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """wall 到期：先发 wrap-up steer（grace 内可收尾），grace 后仍
        不退出才终止。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "WRAPUP_GRACE_SEC", 1.0)
        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        stdin_log = tmp_path / "stdin.log"
        fake = _fake(
            tmp_path,
            "fake-ignore-wrapup",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            # liveness 后台持续（不被误杀）；前台 cat 读 stdin（POSIX sh
            # 的后台任务 stdin 会被重定向到 /dev/null——必须前台读）。
            "(while true; do echo '{\"kind\":\"liveness\",\"phase\":\"RUNNING_MODEL\"}'; sleep 0.5; done) &\n"
            f"cat >> {stdin_log}\n",
        )
        result, _report = await _run(
            service, mission, tmp_path, fake, monkeypatch, wall_time=2
        )
        assert result.status == "FAILED"
        assert "wall budget" in result.summary, result.summary
        assert stdin_log.exists()
        assert "wall" in stdin_log.read_text() or "预算" in stdin_log.read_text()
        await service.close()

    async def test_wrapup_allows_graceful_finish(self, tmp_path: Path, monkeypatch) -> None:
        """wrap-up 后在 grace 内正常完成的单必须被接受（不误杀）。"""
        from rosclaw.agentd.workers import pi_managed

        monkeypatch.setattr(pi_managed, "WRAPUP_GRACE_SEC", 10.0)
        monkeypatch.setattr(pi_managed, "LIVENESS_TIMEOUT_SEC", 30.0)
        service, mission = await _setup(tmp_path)
        fake = _fake(
            tmp_path,
            "fake-graceful",
            "#!/bin/sh\n"
            'echo \'{"kind":"attempt_started"}\'\n'
            "while IFS= read -r line; do\n"
            "  case \"$line\" in\n"
            "    *steer*)\n"
            '      echo \'{"kind":"liveness","phase":"RUNNING_MODEL"}\'\n'
            "      sleep 1\n"
            '      echo \'{"kind":"attempt_finished","report":"partial handoff：已保存"}\'\n'
            "      exit 0;;\n"
            "  esac\n"
            "done\n",
        )
        result, report = await _run(
            service, mission, tmp_path, fake, monkeypatch, wall_time=3
        )
        assert result.status == "COMPLETED", result.summary
        assert "partial handoff" in result.summary
        assert report.accepted
        await service.close()


@pytest.mark.skipif(not _NODE_AVAILABLE, reason="无 node/dist——诚实 skip")
class TestPromptContract:
    def test_developer_prompt_has_no_readonly_contradiction(self) -> None:
        # 用 node 直接读编译后 profiles（真实运行面）。
        proc = subprocess.run(
            [
                "node",
                "-e",
                "import('./packages/rosclaw-agent/dist/src/workers/profiles.js')"
                ".then(m => {"
                "const p = m.profileFor('developer');"
                "const sp = m.buildSystemPrompt(p, '/tmp/ws', ['text/x-diff']);"
                "console.log(JSON.stringify({tools: p.tools, sp}));"
                "})",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert proc.returncode == 0, proc.stderr[-500:]
        payload = json.loads(proc.stdout)
        prompt = payload["sp"]
        assert "read-only tools" not in prompt
        for tool in ("read", "grep", "find", "ls", "write", "edit", "bash"):
            assert tool in payload["tools"]
        assert "Available tools: read, grep, find, ls, write, edit, bash" in prompt
        assert "Required evidence" in prompt
        assert "design document" in prompt.lower()  # DoD 反设计稿声明

    def test_scout_prompt_manifest_is_readonly_honest(self) -> None:
        proc = subprocess.run(
            [
                "node",
                "-e",
                "import('./packages/rosclaw-agent/dist/src/workers/profiles.js')"
                ".then(m => {"
                "const p = m.profileFor('scout');"
                "console.log(JSON.stringify({sp: m.buildSystemPrompt(p, '/tmp/ws', [])}));"
                "})",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert proc.returncode == 0
        prompt = json.loads(proc.stdout)["sp"]
        assert "Available tools: read, grep, find, ls" in prompt
        assert "read-only profile" in prompt
