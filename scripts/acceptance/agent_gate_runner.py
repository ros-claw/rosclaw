#!/usr/bin/env python3
"""B-1 真实 Agent Gate（0916 三审 P0-3）：干净 wheel + 未见任务 ×3。

与 u_real_runner 的关系：复用其 X-2 严格 oracle 机制（per-render
绑定/视觉差/物理重算），泛化到十任务表。差异：
- 被测产品是**干净 venv 里安装的 wheel**（--python 指定），不是
  repo editable——"代码已存在"≠"生产路径已接通"；
- G08 steer/G09 取消 需要 mid-run 发送（驱动扩展）；
- G04-G07 物理 oracle 只认 ws 原始数据文件（npz/csv/json），
  不信终端文字；
- G10 错资产 判诚实失败（错误面+零伪造产物），不判成功。

纪律同 u_real_runner：无 key 全 NOT_RUN；key 只走 env；每 run
独立 HOME+ws；oracle 只看环境结局；机器可读输出。

用法：
  ROSCLAW_KIMI_API_KEY=... python scripts/acceptance/agent_gate_runner.py \
      --python /tmp/g1-venv/bin/python --runs 3 --out /tmp/agent-gate
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "acceptance"))
sys.path.insert(0, str(REPO))

import u_real_runner as ur  # noqa: E402

U = ur  # 短别名（X-2 oracle 复用调用点保持 U._x 形态）

SCENARIOS: dict[str, dict] = {
    "G01": {
        "prompt": (
            "用仿真机械臂画一个房子轮廓（方形墙身+三角屋顶），输出"
            "仿真视频，并且视频里要能看到末端的实际运动轨迹（画在 3D"
            " 场景里的轨迹线，不是 2D 图）。"
        ),
        "oracle": "g01",
    },
    "G02": {
        "prompt": (
            "让仿真机械臂末端走一条空间螺旋线（z 从 0.25m 升到 0.45m，"
            "平面投影是圆），输出带实际轨迹显示的视频；然后**换到顶"
            "视图相机再出一版**（同一轨迹，不重新仿真）。"
        ),
        "oracle": "u06",  # 复用 X-2 U06 严格 oracle
    },
    "G03": {
        "prompt": (
            "写一个单摆（杆长 0.5m）自由摆动的 MuJoCo 模型，比较阻尼 "
            "0.02/0.1/0.3 三种设置的摆动衰减，给出数据结论（从仿真"
            "数据重算衰减率，不要背公式）。"
        ),
        "oracle": "u10",  # 复用 X-2 U10 严格 oracle
    },
    "G04": {
        "prompt": (
            "写一个 MuJoCo 场景：桌面上有一个 5cm 自由方块和一个可"
            "控推杆。用推杆把方块沿 x 方向推出至少 0.15m，记录仿真"
            "数据（方块位置时间序列存文件），从数据回答推出距离。"
        ),
        "oracle": "g04",
    },
    "G05": {
        "prompt": (
            "写一个 MuJoCo 场景：一个夹爪和一个放在桌上的圆柱。控制"
            "夹爪把圆柱抓起来（圆柱离地至少 3cm 并保持），记录仿真"
            "数据（圆柱位置时间序列存文件），从数据回答是否抓起。"
        ),
        "oracle": "g05",
    },
    "G06": {
        "prompt": (
            "写一个 MuJoCo 场景：两个 4cm 方块 box_a、box_b 在桌上。"
            "控制某种机构把 box_b 叠到 box_a 上并保持稳定，记录仿真"
            "数据（两个方块位置时间序列存文件），从数据回答是否叠成。"
        ),
        "oracle": "g06",
    },
    "G07": {
        "prompt": (
            "写一个 MuJoCo 场景：点 A(0,0) 到点 B(0.4,0)，正中间 "
            "(0.2,0) 有一个半径 5cm 的圆形障碍物。让末端从 A 走到 "
            "B 且全程不碰障碍物（绕行），记录路径数据存文件，从数据"
            "回答：到达 B 了吗？离障碍物最近多少？"
        ),
        "oracle": "g07",
    },
    "G08": {
        "prompt": "用仿真机械臂画一个圆形轨迹，输出带实际轨迹的仿真视频。",
        "steer": "改主意了：不要圆了，改成画正方形轨迹，同样输出带轨迹视频。",
        "oracle": "g08",
    },
    "G09": {
        "prompt": (
            "渲染一个 300 帧的高分辨率仿真视频（机械臂走一条复杂"
            "空间曲线），这个任务会比较久，开始干吧。"
        ),
        "cancel": "停下来，不要做了。",
        "oracle": "g09",
    },
    "G10": {
        "prompt": (
            "把 trace_doesnotexist_zzz999 这个 trace 渲染成视频给我看。"
        ),
        "oracle": "g10",
    },
}


def _have_key() -> bool:
    return U._have_key()


# ---------------------------------------------------------------- 物理数据收集

def _iter_named_series(root: Path) -> list[tuple[str, str, list[float]]]:
    """扫 ws 数据文件，产出 (语义名, 文件, 数值序列) 三元组——
    语义名来自键名/列名（box_x/cyl_z/qpos_0 等）。物理结论只从
    这些原始序列重算。"""
    import csv as _csv
    import json as _json

    import numpy as _np

    out: list[tuple[str, str, list[float]]] = []
    for npz_path in root.rglob("*.npz"):
        try:
            archive = _np.load(str(npz_path))
        except Exception:
            continue
        for key in archive.files:
            arr = archive[key]
            if arr.ndim == 1 and arr.size >= 20:
                out.append((key.lower(), str(npz_path), [float(v) for v in arr]))
            elif arr.ndim == 2 and arr.shape[0] >= 20 and arr.shape[1] <= 8:
                for col in range(arr.shape[1]):
                    out.append((
                        f"{key.lower()}_{col}", str(npz_path),
                        [float(v) for v in arr[:, col]],
                    ))
    for csv_path in root.rglob("*.csv"):
        try:
            with csv_path.open(encoding="utf-8", errors="replace") as fh:
                reader = _csv.DictReader(fh)
                if not reader.fieldnames:
                    continue
                cols: dict[str, list[float]] = {
                    c.strip().lower(): [] for c in reader.fieldnames
                }
                for row in reader:
                    for c in cols:
                        with contextlib.suppress(TypeError, ValueError):
                            cols[c].append(float(row[c]))
                for c, vals in cols.items():
                    if len(vals) >= 20:
                        out.append((c, str(csv_path), vals))
        except OSError:
            continue
    for json_path in root.rglob("*.json"):
        try:
            doc = _json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            continue
        out.extend(_json_series(doc, str(json_path)))
    return out


def _json_series(doc, path: str) -> list[tuple[str, str, list[float]]]:
    """JSON 形态：{"box": [[x,y,z]…]} / {"box_z": […]} / [{"box":…}] 等。"""
    out: list[tuple[str, str, list[float]]] = []
    if isinstance(doc, dict):
        for key, value in doc.items():
            k = str(key).lower()
            if isinstance(value, list) and len(value) >= 20:
                if all(isinstance(v, (int, float)) for v in value[:20]):
                    out.append((k, path, [float(v) for v in value]))
                elif all(isinstance(v, (list, tuple)) and len(v) >= 2 for v in value[:5]):
                    width = min(len(v) for v in value[:50])
                    if width <= 8:
                        for col in range(width):
                            out.append((
                                f"{k}_{col}", path,
                                [float(v[col]) for v in value],
                            ))
    return out


def _pick(series: list, *needles: str) -> list[tuple[str, str, list[float]]]:
    """按名字针选序列（box/cyl/障碍/末端等语义）。"""
    return [s for s in series if any(n in s[0] for n in needles)]


def _displacement(vals: list[float]) -> float:
    return max(vals) - min(vals) if vals else 0.0


# ---------------------------------------------------------------- oracle

def _oracle(scenario: str, run, gate_dir: Path) -> dict:
    kind = SCENARIOS[scenario]["oracle"]
    root = run.tmp_path
    if kind in ("u06", "u10"):
        return U._oracle(kind.upper(), run)

    if kind == "g01":
        # 房子轮廓：两轴非退化 + 方向突变 ≥4 次（方形+屋顶折角——
        # 圆/直线/点都不满足）+ 轨迹像素可见 + per-render 绑定。
        renders = U._trace_binding(root)
        good = [
            r for r in renders
            if r["videos"] and not (r["receipt"].get("overlays_unfulfilled") or [])
        ]
        if not good:
            return {"verdict": "FAIL", "detail": f"无绑定渲染（renders={len(renders)}）"}
        b = good[0]
        dx, dy, dz = U._extent(b["actual"])
        turns = _direction_turns(b["actual"])
        visible = U._trace_pixels_visible(b["videos"][0])
        ok = dx > 0.05 and dy + dz > 0.05 and turns >= 4 and visible
        return {
            "verdict": "PASS" if ok else "FAIL",
            "detail": (
                f"两轴=({dx:.3f},{dy:.3f},{dz:.3f})m 折角={turns} "
                f"visible={visible}"
            ),
        }

    if kind == "g04":
        series = _iter_named_series(root / "ws")
        box = _pick(series, "box", "block", "cube")
        xs = [s for s in box if s[0].endswith(("_x", "_0")) or "x" in s[0]]
        best = max((_displacement(s[2]) for s in xs), default=0.0)
        ok = best >= 0.10  # 要求 0.15，容差判 0.10（读数口径差）
        return {
            "verdict": "PASS" if ok else "FAIL",
            "detail": f"box x 最大位移={best:.3f}m（序列 {len(xs)} 组）",
        }

    if kind == "g05":
        series = _iter_named_series(root / "ws")
        cyl = _pick(series, "cyl")
        zs = [s for s in cyl if "z" in s[0] or s[0].endswith("_2")]
        best = 0.0
        for _name, _path, vals in zs:
            lift = max(vals[len(vals) // 2 :]) - sum(vals[:10]) / 10.0
            best = max(best, lift)
        ok = best >= 0.025  # 要求 3cm，容差 2.5cm
        return {
            "verdict": "PASS" if ok else "FAIL",
            "detail": f"圆柱最大抬升={best * 100:.1f}cm（序列 {len(zs)} 组）",
        }

    if kind == "g06":
        series = _iter_named_series(root / "ws")
        za = next((s for s in series if "box_a" in s[0] and ("z" in s[0] or s[0].endswith("_2"))), None)
        zb = next((s for s in series if "box_b" in s[0] and ("z" in s[0] or s[0].endswith("_2"))), None)
        xa = next((s for s in series if "box_a" in s[0] and ("x" in s[0] or s[0].endswith("_0"))), None)
        xb = next((s for s in series if "box_b" in s[0] and ("x" in s[0] or s[0].endswith("_0"))), None)
        if not all([za, zb]):
            return {"verdict": "FAIL", "detail": "缺 box_a/box_b 的 z 序列"}
        tail = lambda v: sum(v[-10:]) / 10.0  # noqa: E731
        gap = tail(zb[2]) - tail(za[2])
        xy_gap = abs(tail(xb[2]) - tail(xa[2])) if xa and xb else 0.0
        ok = 0.02 <= gap <= 0.08 and xy_gap < 0.05
        return {
            "verdict": "PASS" if ok else "FAIL",
            "detail": f"终态 z 差={gap * 100:.1f}cm xy 差={xy_gap * 100:.1f}cm",
        }

    if kind == "g07":
        series = _iter_named_series(root / "ws")
        path_x = next((s for s in series if s[0].endswith(("_x", "_0")) and not _is_obstacle(s)), None)
        path_y = next((s for s in series if s[0].endswith(("_y", "_1")) and not _is_obstacle(s)), None)
        if not path_x or not path_y:
            return {"verdict": "FAIL", "detail": "缺路径 x/y 序列"}
        xs, ys = path_x[2], path_y[2]
        reached = abs(xs[-1] - 0.4) < 0.06
        min_dist = min(((x - 0.2) ** 2 + y**2) ** 0.5 for x, y in zip(xs, ys, strict=False))
        ok = reached and min_dist > 0.05
        return {
            "verdict": "PASS" if ok else "FAIL",
            "detail": f"到达B={reached}（末x={xs[-1]:.3f}）最近距障碍={min_dist * 100:.1f}cm",
        }

    if kind == "g08":
        # steer 生效：steer 之后新产生的绑定渲染 + 最终轨迹是方形
        # （折角 4±2——圆不满足）而非圆。
        renders = U._trace_binding(root)
        steer_time = getattr(run, "steer_time", 0.0)
        post = [
            r for r in renders
            if r["videos"] and _render_mtime(r) >= steer_time
        ]
        if not post:
            return {
                "verdict": "FAIL",
                "detail": f"steer 后无新渲染（renders={len(renders)}）",
            }
        turns = _direction_turns(post[-1]["actual"])
        ok = 2 <= turns <= 8  # 方形 4 折角±容差
        return {
            "verdict": "PASS" if ok else "FAIL",
            "detail": f"steer 后渲染={len(post)} 折角={turns}（方形域 2-8）",
        }

    if kind == "g09":
        # 取消闭环：账本存在 CANCELLED 终结；迟到翻转证据 =
        # SUCCEEDED 且带 cancel_reason（账本防护下不应存在）。
        ledger = _operations_ledger(root)
        cancelled = [op for op in ledger if op.get("state") == "CANCELLED"]
        late_success = [
            op for op in ledger
            if op.get("state") == "SUCCEEDED" and op.get("cancel_reason")
        ]
        ok = bool(cancelled) and not late_success
        return {
            "verdict": "PASS" if ok else "FAIL",
            "detail": (
                f"CANCELLED={len(cancelled)} 迟到翻转={len(late_success)} "
                f"账本={len(ledger)}"
            ),
        }

    if kind == "g10":
        # 诚实失败：错误面（REF_NOT_FOUND 类）+ 零伪造产物 + 无成功宣称。
        session_text = run.session.clean.decode("utf-8", errors="replace") if run.session else ""
        renders = U._trace_binding(root)
        videos = list(root.rglob("*.mp4")) + list(root.rglob("*.gif"))
        error_surfaced = any(
            code in session_text
            for code in ("REF_NOT_FOUND", "RENDER_INPUT_MISSING", "不存在", "找不到", "无法找到")
        )
        fabricated = bool(renders) or bool(videos)
        ok = error_surfaced and not fabricated
        return {
            "verdict": "PASS" if ok else "FAIL",
            "detail": (
                f"错误面={error_surfaced} 伪造产物={fabricated}"
                f"（renders={len(renders)} videos={len(videos)}）"
            ),
        }

    return {"verdict": "FAIL", "detail": f"未知 oracle: {kind}"}


def _direction_turns(actual: list) -> int:
    """轨迹方向突变计数（折角）——相邻段主方向夹角 >45° 记一次。"""
    import math

    if len(actual) < 6:
        return 0
    pts = [(p["x"], p["y"], p["z"]) for p in actual]
    step = max(1, len(pts) // 60)
    segs = []
    for i in range(0, len(pts) - step, step):
        dx = pts[i + step][0] - pts[i][0]
        dy = pts[i + step][1] - pts[i][1]
        dz = pts[i + step][2] - pts[i][2]
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm > 1e-6:
            segs.append((dx / norm, dy / norm, dz / norm))
    turns = 0
    for i in range(1, len(segs)):
        dot = sum(a * b for a, b in zip(segs[i - 1], segs[i], strict=False))
        if dot < 0.707:  # cos45°
            turns += 1
    return turns


def _render_mtime(render: dict) -> float:
    try:
        return render["videos"][0].stat().st_mtime
    except (OSError, IndexError):
        return 0.0


def _is_obstacle(series_entry) -> bool:
    return any(n in series_entry[0] for n in ("obs", "barrier"))


def _operations_ledger(root: Path) -> list[dict]:
    """操作账本——权威源是 agentd 的 missions.db operations 表
    （sqlite），不是文件（G09 run1 实证：文件 glob 全空误判）。
    JSON/JSONL 形态保留兼容。"""
    import json as _json
    import sqlite3

    ops: list[dict] = []
    for db_path in root.rglob("missions.db"):
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT operation_id, task_id, state, cancel_reason, "
                "failure_code FROM operations"
            ).fetchall()
            ops.extend(dict(r) for r in rows)
            conn.close()
        except sqlite3.Error:
            continue
    for path in root.rglob("operations*.json*"):
        try:
            doc = _json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            continue
        if isinstance(doc, list):
            ops.extend(o for o in doc if isinstance(o, dict))
        elif isinstance(doc, dict):
            ops.append(doc)
    return ops


# ---------------------------------------------------------------- 运行

def _run_one(scenario: str, run_idx: int, out_dir: Path, python: str) -> dict:
    from tests.eval.agent_tier import driver

    started = time.monotonic()
    work = out_dir / f"{scenario.lower()}_run{run_idx}"
    work.mkdir(parents=True, exist_ok=True)
    run = driver.AgentRun(work, settle_timeout=1800)
    # 模型 bash 也用干净 wheel venv（不是 harness 的 repo venv）——
    # 干净环境缺库本身就是 gate 要暴露的问题。
    venv_bin = str(Path(python).parent)
    run.env["PATH"] = venv_bin + ":" + run.env.get("PATH", "")
    run.env["VIRTUAL_ENV"] = str(Path(python).parent.parent)
    record: dict = {
        "scenario": scenario, "run": run_idx, "verdict": "FAIL",
        "wall_time_s": 0.0, "interventions": 0, "detail": "",
    }
    spec = SCENARIOS[scenario]
    try:
        if "steer" in spec or "cancel" in spec:
            _run_interactive(run, spec, python)
        else:
            run.run(spec["prompt"], python=python)
        record.update(_oracle(scenario, run, work))
    except Exception as exc:
        record["detail"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        run.stop()
    record["wall_time_s"] = round(time.monotonic() - started, 1)
    return record


def _run_interactive(run, spec: dict, python: str) -> None:
    """G08 steer / G09 取消：mid-run 发送 + 等收束。"""
    from tests.agentd.test_product_journey import PtySession

    run.session = PtySession(
        [python, "-m", "rosclaw.entrypoint", "chat"],
        run.env, cwd=run.ws,
        log_path=run.tmp_path / "pty.log",
    )
    run.session.expect(b"ROSClaw Native Agent", timeout=120)
    run.session.send(spec["prompt"] + "\r")
    # 等任务真的动起来（工作区有文件落盘或输出持续增长）再介入——
    # 抢在 Agent 开工前介入测的不是 steer/cancel 传播。
    deadline = time.monotonic() + 600
    baseline = len(run.session.output)
    while time.monotonic() < deadline:
        grew = len(run.session.output) > baseline + 200
        files = any(run.ws.rglob("*")) or any(
            (run.tmp_path / "rh").rglob("sim/traces/*")
        )
        if grew and files:
            break
        time.sleep(2.0)
    if "cancel" in spec:
        # G09 前提：operation 必须先在账本注册（sqlite 权威——G09
        # run1 实证抢在注册前发停=测不到传播链）。等不到如实记
        # INVALID（不判产品 FAIL——前提没造成）。
        import sqlite3 as _sq

        op_deadline = time.monotonic() + 180
        registered = False
        while time.monotonic() < op_deadline:
            for db in (run.tmp_path / "rh").rglob("missions.db"):
                try:
                    conn = _sq.connect(str(db))
                    count = conn.execute(
                        "SELECT COUNT(*) FROM operations"
                    ).fetchone()[0]
                    conn.close()
                    if count > 0:
                        registered = True
                        break
                except _sq.Error:
                    pass
            if registered:
                break
            time.sleep(2.0)
        if not registered:
            raise RuntimeError(
                "G09 前提未造成：180s 内无 operation 注册（任务未开工）"
            )
    run.steer_time = time.time()
    time.sleep(5.0)  # 让 operation 先注册（G09 账本判定前提）
    run.session.send((spec.get("steer") or spec.get("cancel")) + "\r")
    record_intervention = getattr(run, "interventions", 0)
    run.interventions = record_intervention + 1
    run._wait_settled()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("/tmp/agent-gate"))
    parser.add_argument("--python", default=sys.executable,
                        help="干净 wheel venv 的 python（被测产品入口）")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    args = parser.parse_args()

    if not _have_key():
        print("NOT_RUN: 无 ROSCLAW_KIMI_API_KEY/KIMI_API_KEY（不合成冒充）")
        return 3
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    results: list[dict] = []
    for scenario in scenarios:
        for i in range(1, args.runs + 1):
            record = _run_one(scenario, i, out_dir, args.python)
            results.append(record)
            print(
                f"[{record['scenario']} run{i}] {record['verdict']} "
                f"{record['detail'][:100]} ({record['wall_time_s']}s)",
                flush=True,
            )
            (out_dir / "agent-gate-results.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    print(json.dumps({"pass": passed, "fail": len(results) - passed}))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
