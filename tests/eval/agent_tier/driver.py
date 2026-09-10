"""W09 agent 层真实驱动（规格 §13.1 agent journey 层）。

PTY `rosclaw chat` 驱动真实模型完成任务；oracle 只看**环境结局**
（产物文件的物理内容——重放/重算/阈值），不读模型话术
（§13.13：以环境实际结局为主）。

契约接口（写进 prompt 的工件接口，oracle 据此独立复核）：
- 控制器类任务：模型交 `controller.py`，暴露
  `def control(state: dict) -> float`——oracle 用它在自己的
  仿真里重跑（控制器是模型写的，环境是我们的）；
- 轨迹类任务：模型交 `ctrl_series.json`（{"dt": s, "nu": n,
  "series": [[...], ...]}）——oracle 用同一夹具 MJCF 独立重放
  并量物理量（轨迹是模型写的，物理是我们的）。

无 key / Node 诚实 NOT_RUN（绝不合成冒充）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

KIMI_ENV_VARS = ("ROSCLAW_KIMI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY")


def has_key() -> bool:
    return any(os.environ.get(v) for v in KIMI_ENV_VARS)


def has_runtime() -> bool:
    try:
        from rosclaw.agentd.pi_entry import find_pi_agent_entry

        return find_pi_agent_entry() is not None
    except Exception:  # noqa: BLE001
        return False


class AgentRun:
    """一次真实会话：staging 夹具 → PTY 发送任务 → 等收束。"""

    def __init__(self, tmp_path: Path, *, settle_timeout: float = 900.0):
        from tests.agentd.test_seventeen_gate_live import _prepare_home

        self.tmp_path = tmp_path
        self.settle_timeout = settle_timeout
        self.home, self.env = _prepare_home(tmp_path / "rh")
        # 模型 bash 的 python 必须有公共库栈（mujoco/numpy/imageio
        # 是公共包不是产品特性）——实测无 venv PATH 时模型
        # ModuleNotFoundError 空转（ab_compare A-leg 同款坑）。
        venv_bin = Path(sys.executable).parent
        self.env["PATH"] = str(venv_bin) + ":" + self.env.get("PATH", "")
        self.env["VIRTUAL_ENV"] = str(venv_bin.parent)
        self.ws = tmp_path / "ws"
        self.ws.mkdir(parents=True, exist_ok=True)
        self.session = None

    def run(self, prompt: str, files: dict[str, str] | None = None) -> None:
        from tests.agentd.test_product_journey import PtySession

        for name, content in (files or {}).items():
            target = self.ws / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        self.session = PtySession(
            [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
            self.env, cwd=self.ws,
            log_path=self.tmp_path / "pty.log",
        )
        self.session.expect(b"ROSClaw Native Agent", timeout=120)
        self.session.send(prompt + "\r")
        self._wait_settled()

    def _wait_settled(self) -> None:
        """等回合收束：输出静止 ≥20s **且**工作区文件静止 ≥15s
        （模型 bash 跑长仿真时屏幕不增长但还在落盘/占 CPU——
        只看屏幕会提前误判，L04 实证）。
        kernel 账本只作加分不作门槛——自定义夹具任务（模型直接
        写文件干活）可能全程不产生 kernel 记录（L06 v2 实证）。"""
        import time

        deadline = time.monotonic() + self.settle_timeout
        started = time.monotonic()
        last_len = -1
        quiet_since = time.monotonic()
        while time.monotonic() < deadline:
            with self.session._lock:
                current = len(self.session.output)
            if current != last_len:
                last_len = current
                quiet_since = time.monotonic()
            # 工作区文件静止（bash 长仿真的落盘也算活动）。
            try:
                newest = max(
                    (p.stat().st_mtime
                     for p in self.ws.rglob("*") if p.is_file()),
                    default=0.0,
                )
            except OSError:
                newest = time.time()
            files_quiet = time.time() - newest > 15
            # 至少 45s 后才允许判收束（防输入刚发就误判）。
            if (time.monotonic() - quiet_since > 20
                    and files_quiet
                    and time.monotonic() - started > 45):
                return
            time.sleep(1.0)
        raise AssertionError(
            f"回合 {self.settle_timeout}s 未收束（见 PTY 日志）"
        )

    def stop(self) -> None:
        if self.session is not None:
            with __import__("contextlib").suppress(Exception):
                self.session.stop()

    def followup(self, prompt: str) -> None:
        """同一会话追加用户输入（L09 修改目标/追加交付链路）。"""
        assert self.session is not None, "先 run() 再 followup()"
        self.session.send(prompt + "\r")
        self._wait_settled()


def load_controller(path: Path):
    """加载模型写的控制器（接口：def control(state: dict) -> float）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("agent_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    control = getattr(module, "control", None)
    if not callable(control):
        raise AssertionError(f"{path} 未暴露 control(state) 接口")
    return control


def load_ctrl_series(path: Path) -> tuple[float, list[list[float]]]:
    """加载模型写的 ctrl_series.json → (dt, series)。"""
    doc = json.loads(path.read_text(encoding="utf-8"))
    series = doc.get("series")
    dt = float(doc.get("dt", 0.0))
    if not isinstance(series, list) or not series or dt <= 0:
        raise AssertionError(f"{path} 缺 series/dt")
    return dt, [[float(v) for v in row] for row in series]
