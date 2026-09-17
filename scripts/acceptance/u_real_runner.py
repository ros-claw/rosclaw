#!/usr/bin/env python3
"""U05/U06/U10 真实模型验收 runner（0914 PR-5，审计 §7）。

任务泛化场景必须真实模型（审计：任务泛化场景补真实模型；每类
≥3 次运行；报告分母/失败原因/人工介入/耗时-token）。

纪律：
- 无 ROSCLAW_KIMI_API_KEY/KIMI_API_KEY → 全部 NOT_RUN（不合成冒充）；
- key 只读环境变量；日志绝不打印 key；
- 每次运行独立 HOME+workspace；oracle 只看环境结局（产物/账本/
  文件证据），不信模型自报；
- 输出机器可读 u-real-results.json（并入 U 矩阵）。

用法：
  ROSCLAW_KIMI_API_KEY=... python scripts/acceptance/u_real_runner.py \
      --runs 3 --out /tmp/u-real
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SCENARIOS = {
    "U05": {
        "prompt": (
            "用仿真机械臂画一个 L 形折线（两段：先向右再向上），"
            "输出仿真视频，并且**视频里要能看到末端的实际运动轨迹**"
            "（画在 3D 场景里的轨迹线，不是 2D 图）。"
        ),
        # oracle：场景视频存在 + receipt overlays_applied 含 actual_eef_trace
        "oracle": "u05",
    },
    "U06": {
        "prompt": (
            "让仿真机械臂末端走一条空间螺旋线（z 从 0.25m 升到 0.45m，"
            "平面投影是圆），输出带实际轨迹显示的视频；然后**不换相机"
            "参数再出一版顶视图**（同一轨迹，不重新仿真）。"
        ),
        "oracle": "u06",
    },
    "U10": {
        "prompt": (
            "写一个单摆（杆长 0.5m）自由摆动的 MuJoCo 模型，比较阻尼 "
            "0.02/0.1/0.3 三种设置的摆动衰减，给出数据结论（从仿真数据"
            "重算衰减率，不要背公式）。"
        ),
        "oracle": "u10",
    },
}


def _have_key() -> bool:
    return bool(
        os.environ.get("ROSCLAW_KIMI_API_KEY") or os.environ.get("KIMI_API_KEY")
    )


def _run_one(scenario: str, run_idx: int, out_dir: Path) -> dict:
    """单次真实运行：PTY rosclaw chat（agent_tier driver 复用）。"""
    sys.path.insert(0, str(REPO))
    from tests.eval.agent_tier import driver

    started = time.monotonic()
    work = out_dir / f"{scenario.lower()}_run{run_idx}"
    work.mkdir(parents=True, exist_ok=True)
    run = driver.AgentRun(work, settle_timeout=1800)
    record: dict = {
        "scenario": scenario,
        "run": run_idx,
        "verdict": "FAIL",
        "wall_time_s": 0.0,
        "interventions": 0,
        "detail": "",
    }
    try:
        run.run(SCENARIOS[scenario]["prompt"])
        record.update(_oracle(scenario, run))
    except Exception as exc:  # noqa: BLE001 — 失败如实记录（含原因）
        record["detail"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        run.stop()
    record["wall_time_s"] = round(time.monotonic() - started, 1)
    return record


def _trace_binding(root: Path) -> list[dict]:
    """收集 per-render 绑定：视频↔render_key↔receipt↔trace（W04 keyed
    命名构造绑定）。

    返回 [{trace_id, render_key, receipt, videos, actual}]——每次渲染
    一条。receipt 优先取 per-render 证据 render_receipt-{key}.json
    （0916 起产品侧逐渲染落盘；旧现场回退单文件 receipt——其 camera
    可能被后渲染覆盖，故相机差异另有像素级视觉判定）。states_digest
    必须非空 sha256（None 永不得通过一致性）。
    """
    import json as _json
    import re as _re

    renders: list[dict] = []
    for trace_dir in sorted(root.rglob("sim/traces/*")):
        if not trace_dir.is_dir():
            continue
        trace_id = trace_dir.name
        legacy_path = trace_dir / "render_receipt.json"
        legacy = (
            _json.loads(legacy_path.read_text(encoding="utf-8"))
            if legacy_path.exists() else None
        )
        trace_json = trace_dir / "trace.json"
        actual: list = []
        if trace_json.exists():
            actual = _json.loads(trace_json.read_text(encoding="utf-8")).get("actual") or []
        by_key: dict[str, list[Path]] = {}
        for p in sorted(trace_dir.iterdir()):
            match = _re.match(
                rf"{_re.escape(trace_id)}-([0-9a-f]+)-scene\.(mp4|gif)$", p.name,
            )
            if match:
                by_key.setdefault(match.group(1), []).append(p)
        for render_key, videos in by_key.items():
            per_path = trace_dir / f"render_receipt-{render_key}.json"
            receipt = (
                _json.loads(per_path.read_text(encoding="utf-8"))
                if per_path.exists() else legacy
            )
            if not receipt:
                continue
            states_digest = receipt.get("states_digest")
            if not states_digest or not str(states_digest).startswith("sha256:"):
                continue  # digest 缺失/None——此渲染不参与通过
            renders.append({
                "trace_id": trace_id,
                "render_key": render_key,
                "receipt": receipt,
                "per_render_receipt": per_path.exists(),
                "states_digest": str(states_digest),
                "camera": str(receipt.get("camera", "")),
                "videos": videos,
                "actual": actual,
            })
    return renders


def _extent(actual: list) -> tuple[float, float, float]:
    """轨迹 xyz 三轴跨度（米）。"""
    if not actual:
        return (0.0, 0.0, 0.0)
    xs = [p["x"] for p in actual]
    ys = [p["y"] for p in actual]
    zs = [p["z"] for p in actual]
    return (
        max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs),
    )


def _trace_pixels_visible(video: Path) -> bool:
    """overlay 轨迹色（红系 rgba≈(1.0,0.2,0.2)）在视频中段的像素
    证据——宣称画了≠画面有（0914 mjv 米单位教训）。"""
    import imageio.v3 as iio
    import numpy as np

    frames = [np.asarray(f) for f in iio.imiter(str(video))]
    if len(frames) < 2:
        return False
    hits = 0
    for frame in frames[len(frames) // 3 :]:
        r = frame[:, :, 0].astype(int)
        g = frame[:, :, 1].astype(int)
        b = frame[:, :, 2].astype(int)
        hits += int(((r > 150) & (r > g + 50) & (r > b + 50)).sum())
    return hits > 200


def _visual_difference(video_a: Path, video_b: Path) -> float:
    """两视频中段帧平均像素差——相机差异的环境真相（不信 receipt
    的 camera 字段：旧现场单文件 receipt 被后渲染覆盖）。同相机
    确定性重渲染 ≈0；换相机（follow vs top）远大于阈值。"""
    import imageio.v3 as iio
    import numpy as np

    frames_a = [np.asarray(f) for f in iio.imiter(str(video_a))]
    frames_b = [np.asarray(f) for f in iio.imiter(str(video_b))]
    count = min(len(frames_a), len(frames_b))
    if count < 2:
        return 0.0
    diffs: list[float] = []
    for i in (count // 3, count // 2, (2 * count) // 3):
        a = frames_a[i][:, :, :3].astype(int)
        b = frames_b[i][:, :, :3].astype(int)
        if a.shape != b.shape:
            return float("inf")  # 分辨率都不同——必为不同渲染
        diffs.append(float(np.abs(a - b).mean()))
    return sum(diffs) / len(diffs)


def _oracle(scenario: str, run) -> dict:
    """环境结局核验（不信模型自报，不信终端文字——0916 三审重写）。

    三审稿指摘的假绿根治：
    - 一切一致性判定要求非空 states_digest（None 永远不得通过）；
    - 视频必须经 W04 keyed 文件名与 trace 构造绑定；
    - 物理语义从原始状态/轨迹数据重算，不从 session 文本判定；
    - run root 是每 run 新建 mkdtemp 隔离目录——产物构造上属于
      本 run（无历史 artifact 冒充面）。
    """
    root = run.tmp_path
    renders = _trace_binding(root)
    # 注：run root 是本 run 新建的 mkdtemp 隔离目录（含隔离 home）——
    # 其中一切产物构造上属于本 run，无历史 artifact 冒充面。

    if scenario == "U05":
        # L 形折线 + 视频内实际轨迹：绑定视频的 receipt 画了
        # actual_eef_trace（零 unfulfilled）；轨迹本身两轴非退化
        # （L 不是点也不是单轴线段）；帧内有轨迹色像素。
        good = [
            r for r in renders
            if "actual_eef_trace" in (r["receipt"].get("overlays_applied") or [])
            and not (r["receipt"].get("overlays_unfulfilled") or [])
            and r["videos"]
        ]
        if not good:
            return {
                "verdict": "FAIL",
                "detail": f"无绑定且画轨迹的渲染（renders={len(renders)}）",
            }
        b = good[0]
        dx, dy, _dz = _extent(b["actual"])
        shape_ok = dx > 0.02 and dy > 0.02 and len(b["actual"]) >= 10
        visible = _trace_pixels_visible(b["videos"][0])
        ok = shape_ok and visible
        return {
            "verdict": "PASS" if ok else "FAIL",
            "detail": (
                f"trace={b['trace_id']} 两轴=({dx:.3f},{dy:.3f})m "
                f"visible={visible} videos={len(b['videos'])}"
            ),
        }

    if scenario == "U06":
        # 同一 TraceRef 双渲染（换相机不重仿真）：同 trace ≥2 个
        # render_key、states_digest 全部非空且一致（构造绑定=同
        # trace 目录内渲染读同一份 states）；螺旋语义=z 跨度≥0.1m
        # 且 xy 跨度≥0.05m；相机差异以像素级视觉差判定（不信
        # receipt camera——旧现场单文件 receipt 被后渲染覆盖；
        # per-render receipt 存在时额外核 camera 字段互异）；
        # 至少一视频轨迹色像素可见。
        by_trace: dict[str, list[dict]] = {}
        for r in renders:
            by_trace.setdefault(r["trace_id"], []).append(r)
        for trace_id, trace_renders in by_trace.items():
            if len(trace_renders) < 2:
                continue
            digests = {r["states_digest"] for r in trace_renders}
            actual = trace_renders[0]["actual"]
            dx, dy, dz = _extent(actual)
            helix_ok = dz >= 0.1 and (dx >= 0.05 or dy >= 0.05)
            if len(digests) != 1 or not helix_ok:
                continue
            pair = trace_renders[0], trace_renders[1]
            video_a = next(
                (v for v in pair[0]["videos"] if v.suffix == ".gif"),
                pair[0]["videos"][0],
            )
            video_b = next(
                (v for v in pair[1]["videos"] if v.suffix == ".gif"),
                pair[1]["videos"][0],
            )
            visual = _visual_difference(video_a, video_b)
            if visual < 10.0:
                continue  # 视觉无差异——同相机重渲染不算换相机
            if (
                pair[0]["per_render_receipt"] and pair[1]["per_render_receipt"]
                and pair[0]["camera"] == pair[1]["camera"]
            ):
                continue  # per-render 证据在场时 camera 字段必须互异
            if not _trace_pixels_visible(video_a):
                continue
            return {
                "verdict": "PASS",
                "detail": (
                    f"trace={trace_id} 同 digest 双渲染 "
                    f"visual_diff={visual:.1f} dz={dz:.3f}m "
                    f"cameras={sorted({r['camera'] for r in trace_renders})}"
                ),
            }
        return {
            "verdict": "FAIL",
            "detail": (
                f"无'同 trace 双相机'证据（traces={len(by_trace)} "
                f"renders={len(renders)}）"
            ),
        }

    # U10：三个独立 rollout（不同阻尼）从原始状态序列重算衰减——
    # 终端文字（含提示词自带的 0.02/0.1/0.3/衰减字样）一律不作证据。
    series = _collect_damping_series(root)
    if len(series) < 3:
        return {
            "verdict": "FAIL",
            "detail": f"可用阻尼序列仅 {len(series)} 组（需 3 组独立 rollout）",
        }
    decays: dict[float, float] = {}
    for damping, angles in series.items():
        decays[damping] = _amplitude_decay(angles)
    ordered = sorted(decays.items())
    # 物理单调：阻尼越大衰减率越小（峰值比衰减更快）。
    monotone = all(
        ordered[i][1] < ordered[i - 1][1] for i in range(1, len(ordered))
    )
    distinct = len({round(v, 6) for v in decays.values()}) == len(decays)
    ok = monotone and distinct
    return {
        "verdict": "PASS" if ok else "FAIL",
        "detail": (
            f"衰减率(小→大阻尼)={[f'{d}:{r:.4f}' for d, r in ordered]} "
            f"单调={monotone} 互异={distinct}"
        ),
    }


def _collect_damping_series(root: Path) -> dict[float, list[float]]:
    """从运行现场收集 (阻尼→摆角时间序列)——只认原始物理数据。

    两形态：内核 trajectory_states.json（qpos[0] 摆角 + 同 trace
    MJCF/元数据里的 damping 值）；模型自写 CSV/JSON（列含 damping
    或文件名标阻尼 + 时间+角度列）。三者独立 rollout 才计数——
    同一文件切三段/同一 damping 重复只算一组。
    """
    import json as _json

    series: dict[float, list[float]] = {}
    # 内核形态：trajectory_states.json + 邻近 MJCF/元数据 damping。
    for states_path in root.rglob("trajectory_states.json"):
        payload = _json.loads(states_path.read_text(encoding="utf-8"))
        states = payload.get("states") or []
        if len(states) < 50:
            continue
        damping = _find_damping_near(states_path)
        if damping is None:
            continue
        angles = [float(s["qpos"][0]) for s in states]
        if damping not in series:
            series[damping] = angles
    # 模型自写形态：CSV（damping 列或文件名）与 JSON（damping 键）。
    for csv_path in root.rglob("*.csv"):
        for damping, angles in _angles_from_csv(csv_path):
            if damping is not None and len(angles) >= 50 and damping not in series:
                series[damping] = angles
    for json_path in root.rglob("*.json"):
        damping, angles = _angles_from_json(json_path)
        if damping is not None and len(angles) >= 50 and damping not in series:
            series[damping] = angles
    # npz 形态：单文件多键（q_<阻尼>）或每阻尼一文件（theta 数组）。
    for npz_path in root.rglob("*.npz"):
        for damping, angles in _series_from_npz(npz_path):
            if damping is not None and len(angles) >= 50 and damping not in series:
                series[damping] = angles
    return series


def _series_from_npz(path: Path) -> list[tuple[float | None, list[float]]]:
    import re as _re

    import numpy as np

    name_match = _re.search(
        r"(?:damping|damp|阻尼|_d)[_=-]?([0-9]+(?:[.p][0-9]+)?)", path.name,
    )
    file_damping = (
        float(name_match.group(1).replace("p", ".")) if name_match else None
    )
    found: list[tuple[float | None, list[float]]] = []
    try:
        archive = np.load(str(path))
    except Exception:
        return found  # 坏文件不当证据也不炸 oracle
    for key in archive.files:
        key_match = _re.match(
            r"(?:q|theta|angle|qpos)(?:_([0-9]+(?:\.[0-9]+)?))?$", key,
        )
        if not key_match:
            continue
        array = archive[key]
        if array.ndim != 1 or array.size < 50:
            continue
        damping = float(key_match.group(1)) if key_match.group(1) else file_damping
        found.append((damping, [float(v) for v in array]))
    return found


def _find_damping_near(states_path: Path) -> float | None:
    """trace 邻近文件（MJCF/trace.json/元数据）里的 damping 值——
    从数据声明取，不从提示词取。"""
    import json as _json
    import re

    candidates = [states_path.parent / "trace.json", *sorted(states_path.parent.glob("*.xml"))]
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r'damping["\s:=]+([0-9]+(?:\.[0-9]+)?)', text)
        if match:
            return float(match.group(1))
        # trace.json 的 spec/参数段。
        try:
            doc = _json.loads(text)
        except ValueError:
            continue
        for key in ("damping", "joint_damping"):
            if key in doc:
                return float(doc[key])
        spec = doc.get("spec") or {}
        if "damping" in spec:
            return float(spec["damping"])
    return None


def _angles_from_csv(path: Path) -> list[tuple[float | None, list[float]]]:
    """CSV → [(damping, 摆角序列)]。

    形态：每阻尼一文件（文件名 d0.02/d0p02 或 damping 列单值）；
    单文件多阻尼（damping 列多值——按列值拆分成独立 rollout，
    G03 run3 实证）。"""
    import contextlib
    import csv
    import re as _re

    damping: float | None = None
    # 文件名阻尼形态：d0.02 / d0p02（p=小数点，G03 run1 实证）。
    name_match = _re.search(
        r"(?:damping|damp|阻尼|_d)[_=-]?([0-9]+(?:[.p][0-9]+)?)", path.name,
    )
    if name_match:
        damping = float(name_match.group(1).replace("p", "."))
    angles: list[float] = []
    by_damping: dict[float, list[float]] = {}
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return []
            angle_col = next(
                (c for c in reader.fieldnames
                 if c.strip().lower().startswith(("theta", "angle"))
                 or "qpos" in c.strip().lower()
                 or c.strip().lower() == "q"),
                None,
            )
            damp_col = next(
                (c for c in reader.fieldnames if "damp" in c.lower()), None,
            )
            for row in reader:
                if angle_col is None:
                    break
                try:
                    value = float(row[angle_col])
                except (TypeError, ValueError):
                    continue
                angles.append(value)
                if damp_col:
                    with contextlib.suppress(TypeError, ValueError):
                        row_damping = float(row[damp_col])
                        by_damping.setdefault(row_damping, []).append(value)
    except (OSError, csv.Error):
        return []
    # 多阻尼单文件：拆分优先于整文件单值（每个阻尼一组独立序列）。
    if len(by_damping) >= 2:
        return sorted(by_damping.items())
    if damping is None and by_damping:
        damping = next(iter(by_damping))
    return [(damping, angles)]


def _angles_from_json(path: Path) -> tuple[float | None, list[float]]:
    import json as _json

    try:
        doc = _json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return (None, [])
    if not isinstance(doc, dict):
        return (None, [])
    damping = doc.get("damping")
    series_raw = None
    for key in ("theta", "angle", "angles", "qpos", "series"):
        value = doc.get(key)
        if isinstance(value, list) and value:
            series_raw = value
            break
    if series_raw is None and isinstance(doc.get("states"), list):
        series_raw = [
            s.get("qpos", [None])[0] if isinstance(s, dict) else None
            for s in doc["states"]
        ]
    if damping is None and "params" in doc and isinstance(doc["params"], dict):
        damping = doc["params"].get("damping")
    if series_raw is None or damping is None:
        return (None, [])
    try:
        return (float(damping), [float(v) for v in series_raw if v is not None])
    except (TypeError, ValueError):
        return (None, [])


def _amplitude_decay(angles: list[float]) -> float:
    """衰减率 = 第三峰/第二峰（从序列重算——不读理论值）。"""
    peaks: list[float] = []
    prev_vel = 0.0
    for i in range(1, len(angles)):
        vel = angles[i] - angles[i - 1]
        if prev_vel > 0 >= vel or prev_vel < 0 <= vel:
            peaks.append(abs(angles[i]))
        prev_vel = vel
    if len(peaks) >= 3 and peaks[1] > 1e-9:
        return float(peaks[2] / peaks[1])
    return 1.0


def _rescore(out_dir: Path) -> list[dict]:
    """对已完成的运行现场重判（oracle 修正后）——不重跑、不烧配额。

    首轮实证：oracle 只搜 ws 把真实成功误判 FAIL（内核渲染产物在
    home 侧）。重判仍只看环境结局（现场文件 + pty.log）。
    """
    import json as _json
    from dataclasses import dataclass

    @dataclass
    class _Session:
        clean: bytes

    @dataclass
    class _Run:
        tmp_path: Path
        session: _Session

    results: list[dict] = []
    for run_dir in sorted(out_dir.glob("u*_run*")):
        parts = run_dir.name.split("_run")
        scenario, run_idx = parts[0].upper(), int(parts[1])
        pty = run_dir / "pty.log"
        run = _Run(
            tmp_path=run_dir,
            session=_Session(clean=pty.read_bytes() if pty.exists() else b""),
        )
        record = _oracle(scenario, run)
        record.update({
            "scenario": scenario,
            "run": run_idx,
            "rescored": True,
            "wall_time_s": None,
            "interventions": 0,
        })
        results.append(record)
        print(
            f"[rescore] {scenario} run {run_idx}: {record['verdict']} "
            f"{record['detail'][:120]}",
            flush=True,
        )
    (out_dir / "u-real-rescore.json").write_text(
        _json.dumps({"results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("/tmp/u-real"))
    parser.add_argument(
        "--scenarios", default="U05,U06,U10",
        help="逗号分隔（默认全部真实模型场景）",
    )
    parser.add_argument(
        "--rescore", action="store_true",
        help="对 --out 下已完成运行现场重判（不重跑、不烧配额）",
    )
    args = parser.parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rescore:
        results = _rescore(out_dir)
        passes = sum(1 for r in results if r["verdict"] == "PASS")
        fails = sum(1 for r in results if r["verdict"] == "FAIL")
        print(json.dumps({"pass": passes, "fail": fails}, ensure_ascii=False))
        return 0 if fails == 0 else 1

    results: list[dict] = []
    if not _have_key():
        for scenario in args.scenarios.split(","):
            for i in range(1, args.runs + 1):
                results.append({
                    "scenario": scenario, "run": i, "verdict": "NOT_RUN",
                    "wall_time_s": 0.0, "interventions": 0,
                    "detail": "无真实模型 key——NOT_RUN 不合成冒充",
                })
    else:
        for scenario in args.scenarios.split(","):
            for i in range(1, args.runs + 1):
                print(f"[u-real] {scenario} run {i}/{args.runs} …", flush=True)
                record = _run_one(scenario, i, out_dir)
                results.append(record)
                print(
                    f"[u-real] {scenario} run {i}: {record['verdict']} "
                    f"({record['wall_time_s']}s) {record['detail'][:120]}",
                    flush=True,
                )

    payload = {
        "schema_version": "rosclaw.u_real.v1",
        "runs_per_scenario": args.runs,
        "results": results,
        "pass": sum(1 for r in results if r["verdict"] == "PASS"),
        "fail": sum(1 for r in results if r["verdict"] == "FAIL"),
        "not_run": sum(1 for r in results if r["verdict"] == "NOT_RUN"),
    }
    (out_dir / "u-real-results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(
        {k: payload[k] for k in ("pass", "fail", "not_run")}, ensure_ascii=False,
    ))
    return 0 if payload["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
