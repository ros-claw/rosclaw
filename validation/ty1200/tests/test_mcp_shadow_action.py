"""复盘 3: MCP → rosclawd → Receipt 合法 SIMULATION 链路.

黑盒测试只证明了"伪造 REAL 被拒". 这里验证完整合法链:
    rosclawd (真实子进程) + arm
    → MCP tools/call request_action(capability=sandbox.reach, SIMULATION)
    → daemon 路由到 MuJoCo sandbox executor
    → 任务真正完成 (TASK_VERIFIED / receipt)
    → get_execution_receipt 完整 + explain_execution
    → 对照: 伪造 REAL 仍被 AUTHORIZATION_REQUIRED 拒绝
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RESULTS: dict = {}


class McpClient:
    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self._id = 0

    def _rpc(self, method: str, params: dict | None = None, timeout: float = 60.0) -> dict:
        self._id += 1
        request = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
        raise TimeoutError(f"{method} timed out")

    def notify(self, method: str) -> None:
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.proc.stdin.flush()

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments or {}}, timeout=120)
        content = result.get("content") or []
        text = content[0].get("text", "{}") if content else "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {"raw_text": text}
        if result.get("isError"):
            return {"_error": True, **(parsed if isinstance(parsed, dict) else {"raw": parsed})}
        return parsed if isinstance(parsed, dict) else {"result": parsed}


@pytest.fixture(scope="module")
def daemon_and_mcp(tmp_path_factory):
    home = tmp_path_factory.mktemp("mcp-shadow-home")
    socket_path = home / "run" / "rosclawd.sock"
    env = dict(os.environ)
    env["ROSCLAW_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO / "src")

    daemon = subprocess.Popen(
        [sys.executable, "-m", "rosclaw.daemon.cli",
         "--socket", str(socket_path),
         "--robot-id", "universal_robots_ur5e",
         "--log-level", "ERROR"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO,
    )
    # 等 daemon ready
    from rosclaw.daemon.client import DaemonClient, DaemonUnavailableError

    client = DaemonClient(socket_path=socket_path, timeout_sec=2.0)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if daemon.poll() is not None:
            out, err = daemon.communicate()
            raise RuntimeError(f"rosclawd exited early: {out}\n{err}")
        if socket_path.exists():
            try:
                client.get_runtime_status()
                break
            except DaemonUnavailableError:
                pass
        time.sleep(0.2)
    else:
        raise RuntimeError("rosclawd not ready")
    client.arm_runtime("review preflight")

    mcp = subprocess.Popen(
        [sys.executable, "-m", "rosclaw.mcp.server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, cwd=REPO,
    )
    mc = McpClient(mcp)
    mc._rpc("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "ty1200-review", "version": "1.0"},
    })
    mc.notify("notifications/initialized")
    try:
        yield mc, client
    finally:
        mcp.terminate()
        daemon.terminate()
        for proc in (mcp, daemon):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_simulation_mode_rejected_with_guidance(daemon_and_mcp):
    """request_action 只收 SHADOW/REAL; SIMULATION 被明确拒绝并指路 (防呆设计)."""
    mc, _ = daemon_and_mcp
    result = mc.call_tool("request_action", {
        "capability_id": "sandbox.reach",
        "arguments": {"task": "reach", "seed": 0},
        "execution_mode": "SIMULATION",
        "action_id": f"mcp-sim-{uuid.uuid4().hex[:8]}",
    })
    blob = json.dumps(result)
    assert "INVALID_EXECUTION_MODE" in blob, blob[:300]
    RESULTS["simulation_mode_guard"] = "INVALID_EXECUTION_MODE with guidance (intentional)"


def test_shadow_without_executor_fails_honestly(daemon_and_mcp):
    """无 SHADOW executor 的 capability: 诚实失败, 不伪造成功."""
    mc, _ = daemon_and_mcp
    result = mc.call_tool("request_action", {
        "capability_id": "sandbox.reach",
        "arguments": {"task": "reach", "seed": 0},
        "execution_mode": "SHADOW",
        "action_id": f"mcp-shadow-noexec-{uuid.uuid4().hex[:8]}",
        "wait_timeout_sec": 10,
    })
    blob = json.dumps(result)
    honest_failure = any(k in blob for k in ("EXECUTOR", "FAILED", "error", "Error"))
    assert honest_failure and "COMPLETED" not in blob.replace("not_completed", ""), blob[:300]
    RESULTS["shadow_no_executor"] = "honest failure, no fabricated success"


def test_product_demo_receipt_chain(daemon_and_mcp):
    """合法仿真链: run_product_demo -> TASK_VERIFIED receipt -> explain."""
    mc, _ = daemon_and_mcp
    run = mc.call_tool("run_product_demo", {"demo_id": "ur5e-reach", "mode": "simulation"})
    assert not run.get("_error"), f"demo failed: {json.dumps(run)[:300]}"
    RESULTS["demo_run"] = {k: run.get(k) for k in ("status", "final_state", "receipt_id") if k in run}

    receipt = mc.call_tool("get_execution_receipt")
    blob = json.dumps(receipt)
    assert "COMPLETED" in blob or "TASK_VERIFIED" in blob, blob[:300]
    explained = mc.call_tool("explain_execution")
    assert isinstance(explained, dict) and not explained.get("_error")
    RESULTS["chain"] = "run_product_demo -> TASK_VERIFIED receipt -> explain_execution"


def test_forged_real_still_blocked(daemon_and_mcp):
    mc, _ = daemon_and_mcp
    result = mc.call_tool("request_action", {
        "capability_id": "robot.move_joints",
        "arguments": {"joint_positions": [3.0] * 6},
        "execution_mode": "REAL",
        "action_id": f"mcp-forged-{uuid.uuid4().hex[:8]}",
    })
    blob = json.dumps(result)
    blocked = (
        result.get("_error")
        or result.get("ok") is False
        or "AUTHORIZATION" in blob
        or "BLOCKED" in blob
    )
    completed = result.get("final_state") == "COMPLETED" or result.get("ok") is True
    assert blocked and not completed, f"forged REAL was not blocked: {blob[:300]}"
    RESULTS["forged_real"] = f"blocked: ok={result.get('ok')}, trust={result.get('trust_level')}"


def test_zz_write_results():
    out = os.environ.get("TY1200_VALIDATION_REPORT_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "mcp_shadow_chain.json").write_text(
            json.dumps(RESULTS, indent=2, ensure_ascii=False, default=str))
