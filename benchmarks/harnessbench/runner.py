"""HarnessBench 运行器（MH11，0916 优化 §七/§十）。

一次运行 = 独立 workspace（task.md + model/ + allowed_assets）+
独立 HOME；Agent 只能看到 workspace 与工具面——看不到 tests/
oracle/expected patch/golden answer（§7.1 答案零泄漏）。

两侧条件（§十）：
- A = 原生 coding agent（pi CLI，mujoco+python+bash，无 ROSClaw
  sim 工具）；
- B = ROSClaw Physical Harness（rosclaw chat + `rosclaw sim` CLI）。

同模型/同 prompt/同任务/同 settle 判据。指标：task_success/
verified_success/false_success + wall_time/tool_calls/glue_bytes/
python LOC/XML LOC（§10.1——verified_success↑、false_success→0、
glue_code↓ 三个最关键）。
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.harnessbench import oracle
from benchmarks.harnessbench.tasks import TASKS

REPO = Path(__file__).resolve().parents[2]


class RunInfraError(AssertionError):
    """infra 层失败（stall/未收束/启动失败），携带部分记录。

    live 实证（2026-09-23）：stall 跑真实烧了 1200s+ 却落
    wall_time_s=0.0，聚合 median/P95 被假零污染。异常必须带着
    真实耗时与已收集指标走。
    """

    def __init__(self, message: str, partial: dict[str, Any]) -> None:
        super().__init__(message)
        self.partial = partial


def error_record(
    exc: Exception, leg: str, task_id: str, run_idx: int, *, model: str | None = None
) -> dict[str, Any]:
    """单次失败记录：RunInfraError 保留部分记录（真实 wall_time 等），
    其他异常从零起——verdict/oracle 恒为诚实 ERROR/runner_error。"""
    record: dict[str, Any] = dict(exc.partial) if isinstance(exc, RunInfraError) else {}
    record.update(
        {
            "leg": leg,
            "task_id": task_id,
            "run": run_idx,
            "verdict": "ERROR",
            "error": str(exc)[:300],
            "oracle": {
                "task_success": False,
                "verified_success": False,
                "false_success": False,
                "reason": "runner_error",
            },
        }
    )
    if model is not None:
        record.setdefault("model", model)
    for key, default in (
        ("wall_time_s", 0.0),
        ("tool_calls", 0),
        ("glue_bytes", 0),
        ("bash_python_loc", 0),
        ("python_loc", 0),
        ("xml_loc", 0),
        ("infra_retries", 0),
    ):
        record.setdefault(key, default)
    return record

#: B 侧提示追加（A/B 公平：双方都知道自己有什么工具）。
_B_TOOL_HINT = (
    "\n\n环境里装有 ROSClaw 仿真工具链：rosclaw sim --help 查看"
    "（capabilities/load/inspect/patch/rollout/observe/audit/"
    "branch-experiment/compare/compile-world/render，JSON 接口，"
    "默认以当前目录为任务根）。"
    "修复类任务：fixed_model_ref 必须填 patch 血缘链上的模型引用"
    "（rosclaw sim patch/branch-experiment 返回的 model_ref）——"
    "另写文件再 load 的模型不在血缘内，oracle 不采信。"
)
_A_TOOL_HINT = "\n\n环境里有 Python（含 mujoco/numpy）与 bash。没有专用仿真工具链——一切自己动手。"

#: A 侧干净 venv（无 rosclaw 包——A 条件的本质就是没有 Harness；
#: 首轮 pilot 实证：venv python 自带 rosclaw 时 A 组会自己发现
#: `rosclaw sim` 并用 113 次 patch——那不是 A 条件）。
_A_LEG_VENV = REPO / ".venv-harnessbench-a"


def _a_leg_python() -> str:
    """A 侧干净 python（mujoco/numpy/imageio，**无 rosclaw**）。

    不存在则创建（gitignored .venv-*）；干净性断言失败即诚实
    报错——不把含 rosclaw 的解释器当 A 条件冒充。
    """
    import subprocess
    import venv

    python = _A_LEG_VENV / "bin" / "python"
    if not python.exists():
        venv.create(_A_LEG_VENV, with_pip=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "-q", "mujoco", "numpy", "imageio"],
            check=True,
            timeout=600,
        )
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.util, sys; sys.exit(1 if importlib.util.find_spec('rosclaw') else 0)",
        ],
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError("A_LEG_ENV_CONTAMINATED: clean venv 里能 import rosclaw")
    # pi 启动依赖 fd/ripgrep——干净 PATH 里缺失时 pi 会现场从
    # GitHub 下载（本机 GitHub 直连不可达 → startup 永远卡
    # "still in progress"，prompt 被启动竞态吞掉；v2/v3/v4
    # 三连 45s 假 FAIL 的真根因）。从系统 PATH 预置 symlink。
    import shutil as _shutil

    for tool in ("fd", "rg"):
        source = _shutil.which(tool)
        if source:
            link = _A_LEG_VENV / "bin" / tool
            if not link.exists():
                link.symlink_to(source)
    return str(python)


def _prepare_a_leg_env(workdir: Path, profile: dict[str, Any] | None = None) -> dict:
    """A 组原生 pi：同一模型同一 key（apiKey 只写 $ENV 引用——key
    绝不落盘）；PATH = 干净 python（无 rosclaw）+ node + 系统。
    profile=None → kimi-k3（与 B 侧默认同参）；否则本地档案。"""
    import shutil

    agent_dir = workdir / ".pi-agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    if profile is None:
        provider_block = {
            "name": "kimi-code",
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
        default_provider, default_model = "kimi-code", "k3"
    else:
        provider_block = {
            "name": profile["provider"],
            "baseUrl": profile["base_url"],
            "api": "openai-completions",
            "apiKey": profile["api_key"],
            "models": [
                {
                    "id": profile["model"],
                    "name": profile["model"],
                    "contextWindow": profile["context_window"],
                    "maxTokens": profile["max_tokens"],
                }
            ],
        }
        default_provider, default_model = profile["provider"], profile["model"]
    (agent_dir / "settings.json").write_text(
        json.dumps({"defaultProvider": default_provider, "defaultModel": default_model}),
        encoding="utf-8",
    )
    (agent_dir / "models.json").write_text(
        json.dumps({"providers": {default_provider: provider_block}}),
        encoding="utf-8",
    )
    _a_leg_python()
    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    node_path = shutil.which("node")
    node_bin = str(Path(node_path).parent) if node_path else "/usr/local/bin"
    env["PATH"] = f"{_A_LEG_VENV / 'bin'}:{node_bin}:/usr/local/bin:/usr/bin:/bin"
    return env


def _count_session_stats(session_dir: Path) -> tuple[int, int, int]:
    """pi session JSONL → (工具调用数, write/edit 字节数, bash 内联
    python 行数)——heredoc 胶水也是胶水（首轮 pilot 实证：A 组用
    bash heredoc 跑 python 时 write/edit 计数为 0）。"""
    tool_calls = 0
    glue_bytes = 0
    bash_python_loc = 0
    if not session_dir.is_dir():
        return 0, 0, 0
    for f in session_dir.glob("**/*.jsonl"):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = entry.get("message") or {}
            for block in msg.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "toolCall":
                    tool_calls += 1
                    args = block.get("arguments") or block.get("input") or {}
                    if block.get("name") in ("write", "edit"):
                        glue_bytes += len(str(args.get("content", args.get("newText", ""))))
                    if block.get("name") == "bash":
                        command = str(args.get("command", ""))
                        if "python" in command or "mujoco" in command:
                            bash_python_loc += len(command.splitlines())
    return tool_calls, glue_bytes, bash_python_loc


#: 模型档案（0916 §十一：资格认证至少两个模型，防 prompt
#: overfit——Harness 只对一个模型有效就不是真价值）。
#: None = 默认 kimi-k3（走 _prepare_home 既有路径）。
MODEL_PROFILES: dict[str, dict[str, Any] | None] = {
    "kimi-k3": None,
    "deepseekv4": {
        "provider": "local-vllm",
        "model": "deepseekv4",
        "base_url": "http://10.10.217.108:30456/v1",
        "api_key": "EMPTY",  # 本地 vllm 不校验；非真实 secret
        "context_window": 131072,
        "max_tokens": 16384,
    },
}


def has_model_key(model: str = "kimi-k3") -> bool:
    profile = MODEL_PROFILES.get(model)
    if profile is not None:
        return True  # 本地档案无需远端 key
    return any(
        os.environ.get(v) for v in ("ROSCLAW_KIMI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY")
    )


def _prepare_home_with_profile(home: Path, profile: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    """本地模型档案的 B 侧 HOME（与 _prepare_home 同构，provider
    指向本地 OpenAI 兼容端点；无 key 要求）。"""
    (home / "run").mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "agent:\n  enabled: true\n  default_profile: embodied_default\n"
        "models:\n  backend: legacy\n  profiles:\n    embodied_default:\n"
        f"      provider: local\n      model: {profile['model']}\n"
        f"      base_url: {profile['base_url']}\n"
        '      api_key_ref: ""\n'
        "      capabilities: [llm.chat, llm.structured_decision, llm.tool_use]\n",
        encoding="utf-8",
    )
    (home / "agent").mkdir(parents=True, exist_ok=True)
    (home / "agent" / "settings.json").write_text(
        json.dumps({"defaultProvider": profile["provider"], "defaultModel": profile["model"]}),
        encoding="utf-8",
    )
    (home / "agent" / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    profile["provider"]: {
                        "name": profile["provider"],
                        "baseUrl": profile["base_url"],
                        "api": "openai-completions",
                        "apiKey": profile["api_key"],
                        "models": [
                            {
                                "id": profile["model"],
                                "name": profile["model"],
                                "contextWindow": profile["context_window"],
                                "maxTokens": profile["max_tokens"],
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ, ROSCLAW_HOME=str(home), TERM="xterm")
    venv_bin = Path(sys.executable).parent
    env["PATH"] = str(venv_bin) + ":" + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(venv_bin.parent)
    return home, env


def _code_loc(workspace: Path) -> dict[str, int]:
    """模型写的 Python/XML 行数（glue code 指标；staged model/ 除外）。"""
    python_loc = 0
    xml_loc = 0
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace)
        if rel.parts[0] in ("model", "sim", "task.md", "answer.json"):
            continue
        if path.suffix == ".py":
            python_loc += len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        elif path.suffix in (".xml", ".mjcf"):
            xml_loc += len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    return {"python_loc": python_loc, "xml_loc": xml_loc}


def stage_workspace(base: Path, task_id: str) -> Path:
    """独立 task workspace（§7.1）：task.md + staged files，别无他物。"""
    task = TASKS[task_id]
    base.mkdir(parents=True, exist_ok=True)
    (base / "task.md").write_text(task.prompt, encoding="utf-8")
    for rel, content in task.staged_files.items():
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return base


def _wait_settled(
    session, workspace: Path, settle_timeout: float, *, prompt: str | None = None
) -> int:
    """等回合收束：输出静止 ≥20s 且工作区文件静止 ≥15s
    （driver.AgentRun 同款判据——两侧同标准）。

    provider 首 token 30s 无响应会自动取消请求（实测两例）——
    检测到取消标记且传了 prompt 就自动重发（最多 2 次，
    返回重发次数计入 infra_retries；这是产品自身的"可重发"语义，
    不把 API 瞬时故障算成 Agent 能力失败）。"""
    deadline = time.monotonic() + settle_timeout
    started = time.monotonic()
    with session._lock:
        baseline_len = len(session.output)
    last_len = baseline_len
    quiet_since = time.monotonic()
    retries = 0
    scanned = baseline_len
    activity_seen = False
    activity_baseline = baseline_len  # 8s 宽限后再定基线（排除 prompt 回显）
    while time.monotonic() < deadline:
        now = time.monotonic()
        with session._lock:
            current = len(session.output)
            output = bytes(session.output)
        if now - started > 8 and activity_baseline == baseline_len and current > baseline_len:
            # 宽限后仍有增长：把 8s 处的长度当活动基线（之前的多半是回显）。
            activity_baseline = current
        if current != last_len:
            last_len = current
            quiet_since = now
            if now - started > 8 and current > activity_baseline:
                activity_seen = True
                activity_baseline = current
        # provider 自动取消检测（增量扫描新输出）。
        if retries < 2 and len(output) > scanned:
            chunk = output[scanned:]
            scanned = len(output)
            if prompt is not None and (
                b"Operation aborted" in chunk or "已取消本次请求".encode() in chunk
            ):
                retries += 1
                quiet_since = time.monotonic()
                session.send(prompt + "\r")
                time.sleep(2.0)
                continue
        else:
            scanned = len(output)
        # 零活动 stall：发送后 120s 无任何输出增长 = 请求卡死在
        # API 慢波次（实测 native pi 无 30s 看门狗，45s 安静地板会
        # 把"还没收到首 token"误判成收束）——重发而非判失败。
        if prompt is not None and not activity_seen and time.monotonic() - started > 120:
            if retries >= 2:
                raise AssertionError("INFRA_STALL: 两次重发后仍无模型活动（API 不可用波次）")
            retries += 1
            started = time.monotonic()
            quiet_since = time.monotonic()
            session.send(prompt + "\r")
            time.sleep(2.0)
            continue
        try:
            newest = max(
                (p.stat().st_mtime for p in workspace.rglob("*") if p.is_file()),
                default=0.0,
            )
        except OSError:
            newest = time.time()
        files_quiet = time.time() - newest > 15
        if (
            activity_seen
            and time.monotonic() - quiet_since > 20
            and files_quiet
            and time.monotonic() - started > 45
        ):
            return retries
        time.sleep(1.0)
    raise AssertionError(f"回合 {settle_timeout}s 未收束（见 PTY 日志）")


def run_leg(
    leg: str,
    task_id: str,
    out_root: Path,
    run_idx: int,
    *,
    settle_timeout: float = 1200.0,
    model: str = "kimi-k3",
) -> dict[str, Any]:
    """跑一侧一次：A=pi CLI（无扩展）；B=rosclaw chat（sim CLI）。
    model 取 MODEL_PROFILES 键（§十一 多模型资格认证）。"""
    sys.path.insert(0, str(REPO))
    from tests.agentd.test_product_journey import PtySession

    if model not in MODEL_PROFILES:
        raise ValueError(f"BENCH_MODEL_UNKNOWN: {model!r}（支持 {sorted(MODEL_PROFILES)}）")
    profile = MODEL_PROFILES[model]

    task = TASKS[task_id]
    work = out_root / f"{leg.lower()}_{task_id.lower()}_{model}_run{run_idx}"
    stage_workspace(work, task_id)
    hint = _B_TOOL_HINT if leg == "B" else _A_TOOL_HINT
    prompt = task.prompt + hint

    started = time.monotonic()
    record: dict[str, Any] = {
        "leg": leg,
        "task_id": task_id,
        "category": task.category,
        "model": model,
        "run": run_idx,
        "wall_time_s": 0.0,
        "tool_calls": 0,
        "glue_bytes": 0,
        "verdict": "ERROR",
    }

    session = None
    session_dir: Path | None = None
    try:
        if leg == "A":
            from scripts.ab_compare import _find_native_pi_cli

            pi_entry = _find_native_pi_cli()
            assert pi_entry is not None, "A 组需要原生 pi CLI"
            session = PtySession(
                [pi_entry],
                _prepare_a_leg_env(work, profile),
                cwd=work,
                log_path=work / "pty.log",
            )
            session.expect(b"ctrl+o to show full startup help", timeout=120)
            session_dir = work / ".pi-agent" / "sessions"
        elif profile is not None:
            home, home_env = _prepare_home_with_profile(work / "rh", profile)
            session = PtySession(
                [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
                home_env,
                cwd=work,
                log_path=work / "pty.log",
            )
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session_dir = home / "agent" / "sessions"
        else:
            from tests.eval.agent_tier import driver

            run = driver.AgentRun(work, settle_timeout=settle_timeout)
            home_env = run.env
            session = PtySession(
                [sys.executable, "-m", "rosclaw.entrypoint", "chat"],
                home_env,
                cwd=work,
                log_path=work / "pty.log",
            )
            session.expect(b"ROSClaw Native Agent", timeout=120)
            session_dir = run.home / "agent" / "sessions"

        session.send(prompt + "\r")
        record["infra_retries"] = _wait_settled(session, work, settle_timeout, prompt=prompt)
    except Exception as exc:
        # stall/未收束/启动失败：保留已收集指标（真实 wall_time 由
        # finally 填入），包成 RunInfraError 让调用方落诚实 ERROR 记录。
        if session_dir is not None:
            with contextlib.suppress(Exception):
                (
                    record["tool_calls"],
                    record["glue_bytes"],
                    record["bash_python_loc"],
                ) = _count_session_stats(session_dir)
        with contextlib.suppress(Exception):
            record.update(_code_loc(work))
        raise RunInfraError(str(exc), record) from exc
    finally:
        record["wall_time_s"] = round(time.monotonic() - started, 1)
        if session is not None:
            with contextlib.suppress(Exception):
                session.stop()

    if session_dir is not None:
        (
            record["tool_calls"],
            record["glue_bytes"],
            record["bash_python_loc"],
        ) = _count_session_stats(session_dir)
    record.update(_code_loc(work))

    # Oracle 独立判定（环境结局，不信自报；A/B 证据通道分侧）。
    verdict = oracle.judge(task_id, work, leg=leg)
    record["oracle"] = verdict
    record["verdict"] = (
        "VERIFIED"
        if verdict.get("verified_success")
        else "FALSE_SUCCESS"
        if verdict.get("false_success")
        else "DONE"
        if verdict.get("task_success")
        else "FAIL"
    )
    return record


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval（§31 95% CI——小样本更宽，k=0/n 夹紧）。"""
    if n <= 0:
        return 0.0, 1.0
    if k <= 0:
        return 0.0, 1 - (0.05 ** (1 / n))
    if k >= n:
        return 0.05 ** (1 / n), 1.0
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """按侧聚合（§31 统计输出：rate + CI + median/P95 + infra retry）。

    LLM 是随机系统——1 run 不算数：verified/false-success rate 带
    95% Wilson CI；wall time 给 median 与 P95；infra retry 单列
    （API 波次故障不算能力失败）。
    """
    summary: dict[str, Any] = {}
    for leg in ("A", "B"):
        rows = [r for r in records if r["leg"] == leg]
        if not rows:
            continue
        total = len(rows)
        verified = sum(1 for r in rows if r["oracle"].get("verified_success"))
        done = sum(1 for r in rows if r["oracle"].get("task_success"))
        false_success = sum(1 for r in rows if r["oracle"].get("false_success"))
        retries = sum(1 for r in rows if r.get("infra_retries", 0))
        walls = sorted(r["wall_time_s"] for r in rows)
        summary[leg] = {
            "runs": total,
            "task_success_rate": done / total,
            "verified_success_rate": verified / total,
            "verified_ci95": wilson_interval(verified, total),
            "false_success_rate": false_success / total,
            "false_success_ci95": wilson_interval(false_success, total),
            "infra_retry_rate": retries / total,
            "wall_time_s_median": walls[total // 2],
            "wall_time_s_p95": walls[min(total - 1, int(0.95 * total) if total > 1 else 0)],
            "tool_calls_median": sorted(r["tool_calls"] for r in rows)[total // 2],
            "glue_bytes_median": sorted(r["glue_bytes"] for r in rows)[total // 2],
            "bash_python_loc_median": sorted(r.get("bash_python_loc", 0) for r in rows)[total // 2],
            "python_loc_median": sorted(r["python_loc"] for r in rows)[total // 2],
            "xml_loc_median": sorted(r["xml_loc"] for r in rows)[total // 2],
        }
    return summary
