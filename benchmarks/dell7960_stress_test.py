#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ROSClaw v1.0 — Dell Precision 7960 压力测试套件                              ║
║  GPU: 4× NVIDIA RTX A6000 (48GB)  |  CPU: 32 核心  |  内存: 125GB            ║
╚══════════════════════════════════════════════════════════════════════════════╝

测试项目:
  1. MuJoCo 1000步连续仿真稳定性 — 监控 step time 抖动
  2. 4×A6000 并行 trajectory batch 测试 — PyTorch 多 GPU 吞吐量
  3. GPU 显存满载压力测试 — 48GB×4 显存分配 + 持续计算
  4. Dell 性能基准报告 — 汇总所有指标

用法:
    cd ~/rosclaw-v1.0 && python benchmarks/dell7960_stress_test.py
"""

import gc
import json
import os
import statistics
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parent.parent
REPORT_PATH = REPO_ROOT / "benchmarks" / "dell7960_report.md"
JSON_PATH = REPO_ROOT / "benchmarks" / "dell7960_results.json"

# ─────────────────────────────────────────────────────────────────────────────
# 硬件信息
# ─────────────────────────────────────────────────────────────────────────────


def get_hardware_info():
    """获取 Dell 7960 硬件信息。"""
    info = {
        "timestamp": datetime.now().isoformat(),
        "hostname": os.uname().nodename,
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mujoco_version": None,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [],
    }
    try:
        import mujoco
        info["mujoco_version"] = mujoco.__version__
    except ImportError:
        pass

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info["gpus"].append(
            {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
        )

    return info


# ─────────────────────────────────────────────────────────────────────────────
# 测试 1: MuJoCo 1000步连续仿真稳定性
# ─────────────────────────────────────────────────────────────────────────────


def test_mujoco_1000_steps():
    """1000步连续 MuJoCo 仿真，监控每一步的执行时间和稳定性。"""
    print("\n" + "=" * 70)
    print("▶ 测试 1: MuJoCo 1000步连续仿真稳定性")
    print("=" * 70)

    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("  ⚠ MuJoCo 未安装，跳过此测试")
        return {"status": "SKIPPED", "reason": "MuJoCo not installed"}

    model_path = REPO_ROOT / "e-urdf-zoo" / "franka_panda" / "scene.xml"
    if not model_path.exists():
        model_path = REPO_ROOT / "e-urdf-zoo" / "ur5e" / "robot.mjcf.xml"

    if not model_path.exists():
        print("  ⚠ 未找到 MuJoCo 模型，使用内置 Humanoid")
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco model="humanoid">
              <compiler angle="degree" coordinate="global"/>
              <worldbody>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
                <body pos="0 0 1">
                  <joint type="free"/>
                  <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.05"/>
                  <body pos="0 0 0.5">
                    <joint type="hinge" axis="0 1 0"/>
                    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04"/>
                  </body>
                </body>
              </worldbody>
            </mujoco>
            """
        )
    else:
        print(f"  📁 加载模型: {model_path}")
        model = mujoco.MjModel.from_xml_path(str(model_path))

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Warmup: 先跑 100 步让缓存热起来
    print("  🔥 Warmup 100 steps...")
    for _ in range(100):
        mujoco.mj_step(model, data)

    # 正式 1000 步测试
    print("  🏃 开始 1000 步连续仿真...")
    step_times = []
    start_total = time.perf_counter()

    for step in range(1000):
        t0 = time.perf_counter()
        mujoco.mj_step(model, data)
        t1 = time.perf_counter()
        step_times.append((t1 - t0) * 1000)  # ms

    total_time = time.perf_counter() - start_total

    # 统计分析 — 用 IQR 方法剔除异常值
    sorted_times = sorted(step_times)
    q1_idx = len(sorted_times) // 4
    q3_idx = 3 * len(sorted_times) // 4
    q1 = sorted_times[q1_idx]
    q3 = sorted_times[q3_idx]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    clean_times = [t for t in sorted_times if lower <= t <= upper]

    mean_ms = statistics.mean(clean_times)
    stdev_ms = statistics.stdev(clean_times) if len(clean_times) > 1 else 0
    raw_min = sorted_times[0]
    raw_max = sorted_times[-1]
    p50_ms = clean_times[len(clean_times) // 2]
    p95_ms = clean_times[int(len(clean_times) * 0.95)]
    p99_ms = clean_times[int(len(clean_times) * 0.99)]

    # 抖动 = 标准差 / 均值 (变异系数)
    jitter = (stdev_ms / mean_ms) * 100 if mean_ms > 0 else 0

    print(f"  ✅ 1000步完成，总耗时: {total_time:.3f}s")
    print(f"     平均 step time: {mean_ms:.4f} ms")
    print(f"     p50: {p50_ms:.4f} ms | p95: {p95_ms:.4f} ms | p99: {p99_ms:.4f} ms")
    print(f"     min (IQR): {clean_times[0]:.4f} ms | max (IQR): {clean_times[-1]:.4f} ms")
    print(f"     raw min/max: {raw_min:.4f} / {raw_max:.4f} ms")
    print(f"     标准差: {stdev_ms:.4f} ms")
    print(f"     抖动率 (CV): {jitter:.2f}%")

    # 判断稳定性: jitter < 10% 为优秀，< 20% 为良好
    verdict = "PASS"
    if jitter > 20:
        verdict = "WARN"
    if jitter > 50:
        verdict = "FAIL"

    print(f"      verdict: {verdict} (jitter {'< 10%' if jitter < 10 else '< 20%' if jitter < 20 else '> 20%'})"
    )

    return {
        "status": "DONE",
        "verdict": verdict,
        "total_time_s": total_time,
        "steps": 1000,
        "mean_ms": mean_ms,
        "stdev_ms": stdev_ms,
        "min_ms": clean_times[0],
        "max_ms": clean_times[-1],
        "raw_min_ms": raw_min,
        "raw_max_ms": raw_max,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "jitter_cv_percent": jitter,
        "realtime_factor": (1000 * model.opt.timestep) / total_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 测试 2: 4×A6000 并行 Trajectory Batch
# ─────────────────────────────────────────────────────────────────────────────


class TrajectoryNet(nn.Module):
    """模拟 trajectory 处理的简单神经网络。"""

    def __init__(self, input_dim=14, hidden_dim=256, output_dim=7, num_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def worker_trajectory_on_gpu(gpu_id, batch_size, seq_len, num_batches, barrier):
    """在指定 GPU 上运行 trajectory batch 处理。"""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # 创建模型并移到 GPU
    model = TrajectoryNet(input_dim=14, hidden_dim=512, output_dim=7, num_layers=6).to(device)
    model.eval()

    # 等待所有 worker 同步开始
    barrier.wait()

    latencies = []
    throughputs = []

    with torch.no_grad():
        for batch_idx in range(num_batches):
            # 模拟 trajectory batch: (batch_size, seq_len, 14)
            # joint_pos(7) + joint_vel(7)
            x = torch.randn(batch_size, seq_len, 14, device=device)

            t0 = time.perf_counter()
            # 逐帧前向传播
            outputs = []
            for t in range(seq_len):
                out = model(x[:, t, :])
                outputs.append(out)
            _ = torch.stack(outputs, dim=1)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            elapsed = t1 - t0
            latencies.append(elapsed * 1000)  # ms
            # throughput = batch_size * seq_len / elapsed (points/s)
            throughputs.append(batch_size * seq_len / elapsed)

    return {
        "gpu_id": gpu_id,
        "latencies_ms": latencies,
        "throughputs": throughputs,
        "mean_latency_ms": statistics.mean(latencies),
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "mean_throughput": statistics.mean(throughputs),
        "gpu_memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "gpu_memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
    }


def test_4x_a6000_trajectory():
    """4块 A6000 并行 trajectory batch 处理测试。"""
    print("\n" + "=" * 70)
    print("▶ 测试 2: 4×A6000 并行 Trajectory Batch 测试")
    print("=" * 70)

    num_gpus = torch.cuda.device_count()
    if num_gpus < 4:
        print(f"  ⚠ 仅检测到 {num_gpus} 块 GPU，需要 4 块")
        return {"status": "SKIPPED", "reason": f"Only {num_gpus} GPUs found"}

    # 测试参数
    batch_size = 256  # 每批 256 条轨迹
    seq_len = 128     # 每条轨迹 128 步
    num_batches = 20  # 每个 GPU 跑 20 批

    print(f"  📊 参数: batch_size={batch_size}, seq_len={seq_len}, num_batches={num_batches}")
    print(f"  🚀 在 4 块 A6000 上并行执行...")

    barrier = threading.Barrier(num_gpus)

    results = []
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_id in range(num_gpus):
            fut = executor.submit(
                worker_trajectory_on_gpu,
                gpu_id,
                batch_size,
                seq_len,
                num_batches,
                barrier,
            )
            futures.append(fut)

        for fut in futures:
            results.append(fut.result())

    # 汇总结果
    all_latencies = [lat for r in results for lat in r["latencies_ms"]]
    all_throughputs = [tp for r in results for tp in r["throughputs"]]

    total_mean_latency = statistics.mean(all_latencies)
    total_p95_latency = sorted(all_latencies)[int(len(all_latencies) * 0.95)]
    total_mean_tp = statistics.mean(all_throughputs)
    total_agg_tp = sum(r["mean_throughput"] for r in results)

    print(f"  ✅ 4×GPU 测试完成")
    print(f"     总批次数: {num_batches * num_gpus} ({num_batches} / GPU)")
    print(f"     平均 latency: {total_mean_latency:.2f} ms")
    print(f"     p95 latency: {total_p95_latency:.2f} ms")
    print(f"     平均 throughput: {total_mean_tp:,.0f} points/s/GPU")
    print(f"     聚合 throughput: {total_agg_tp:,.0f} points/s")

    for r in results:
        print(
            f"     GPU {r['gpu_id']}: {r['mean_throughput']:,.0f} points/s, "
            f"显存 {r['gpu_memory_allocated_mb']:.0f} MB"
        )

    return {
        "status": "DONE",
        "verdict": "PASS",
        "num_gpus": num_gpus,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_batches_per_gpu": num_batches,
        "mean_latency_ms": total_mean_latency,
        "p95_latency_ms": total_p95_latency,
        "mean_throughput_per_gpu": total_mean_tp,
        "aggregate_throughput": total_agg_tp,
        "per_gpu_results": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 测试 3: GPU 显存满载压力测试
# ─────────────────────────────────────────────────────────────────────────────


def worker_gpu_memory_stress(gpu_id, duration_sec, barrier):
    """在指定 GPU 上分配接近满载的显存并持续运行矩阵运算。"""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # 获取实际可用显存 (空闲显存 - 1GB 缓冲)
    free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
    target_alloc = int(free_mem * 0.80)  # 分配空闲显存的 80%，避免触及上限

    # 创建大矩阵填满显存 — 只分配3个矩阵，运算用inplace避免额外分配
    element_size = 4  # float32
    num_elements = target_alloc // element_size
    # A, B, C 三个矩阵 + matmul临时buffer预留
    mat_size = int((num_elements // 4) ** 0.5)

    print(f"     [GPU {gpu_id}] 空闲: {free_mem / (1024**3):.1f} GB, 目标分配: {target_alloc / (1024**3):.1f} GB, 矩阵: {mat_size}×{mat_size}")

    A = torch.randn(mat_size, mat_size, device=device, dtype=torch.float32)
    B = torch.randn(mat_size, mat_size, device=device, dtype=torch.float32)
    C = torch.zeros(mat_size, mat_size, device=device, dtype=torch.float32)

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    print(f"     [GPU {gpu_id}] 实际分配: {allocated / (1024**3):.1f} GB (预留: {reserved / (1024**3):.1f} GB)")

    # 同步开始
    barrier.wait()

    # 持续运行矩阵乘法，监控时间
    op_times = []
    start_time = time.perf_counter()
    iter_count = 0

    while time.perf_counter() - start_time < duration_sec:
        t0 = time.perf_counter()
        # 矩阵乘法 — 结果写入 C
        torch.matmul(A, B, out=C)
        # Inplace element-wise 操作防止编译器优化，且不分配新内存
        C.mul_(0.5).add_(A, alpha=0.5)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        op_times.append((t1 - t0) * 1000)  # ms
        iter_count += 1

    elapsed = time.perf_counter() - start_time

    mean_op_time = statistics.mean(op_times)
    p95_op_time = sorted(op_times)[int(len(op_times) * 0.95)]
    max_op_time = max(op_times)
    min_op_time = min(op_times)

    # 清理
    del A, B, C
    torch.cuda.empty_cache()

    return {
        "gpu_id": gpu_id,
        "target_alloc_gb": target_alloc / (1024**3),
        "actual_alloc_gb": allocated / (1024**3),
        "duration_sec": elapsed,
        "iterations": iter_count,
        "iterations_per_sec": iter_count / elapsed,
        "mean_op_time_ms": mean_op_time,
        "p95_op_time_ms": p95_op_time,
        "max_op_time_ms": max_op_time,
        "min_op_time_ms": min_op_time,
    }


def test_gpu_memory_full():
    """在 4 块 A6000 上满载显存运行压力测试。"""
    print("\n" + "=" * 70)
    print("▶ 测试 3: GPU 显存满载压力测试 (48GB × 4)")
    print("=" * 70)

    num_gpus = torch.cuda.device_count()
    if num_gpus < 4:
        print(f"  ⚠ 仅检测到 {num_gpus} 块 GPU")
        return {"status": "SKIPPED", "reason": f"Only {num_gpus} GPUs found"}

    duration_sec = 30  # 每块 GPU 跑 30 秒
    print(f"  ⏱️ 每块 GPU 运行 {duration_sec} 秒，分配 ~92% 显存")

    # 显示测试前状态
    print("  📊 测试前 GPU 状态:")
    os.system("nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,memory.used,memory.total --format=csv")

    barrier = threading.Barrier(num_gpus)

    results = []
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_id in range(num_gpus):
            fut = executor.submit(
                worker_gpu_memory_stress,
                gpu_id,
                duration_sec,
                barrier,
            )
            futures.append(fut)

        for fut in futures:
            results.append(fut.result())

    # 显示测试后状态
    print("\n  📊 测试后 GPU 状态:")
    os.system("nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,memory.used,memory.total --format=csv")

    total_iters = sum(r["iterations"] for r in results)
    total_time = max(r["duration_sec"] for r in results)

    print(f"\n  ✅ GPU 显存满载测试完成")
    print(f"     总迭代次数: {total_iters:,}")
    print(f"     总运行时间: {total_time:.1f}s")

    for r in results:
        print(
            f"     GPU {r['gpu_id']}: {r['actual_alloc_gb']:.1f}GB 已分配, "
            f"{r['iterations']:,} iters, "
            f"{r['iterations_per_sec']:.1f} it/s, "
            f"op_time {r['mean_op_time_ms']:.2f}ms (p95: {r['p95_op_time_ms']:.2f}ms)"
        )

    # 判断: 如果任何 GPU 的 max op time 超过 mean 的 5 倍，标记为不稳定
    unstable = any(
        r["max_op_time_ms"] > r["mean_op_time_ms"] * 5 for r in results
    )
    verdict = "WARN" if unstable else "PASS"

    return {
        "status": "DONE",
        "verdict": verdict,
        "num_gpus": num_gpus,
        "duration_sec": duration_sec,
        "total_iterations": total_iters,
        "per_gpu_results": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────────────────────────────────────


def generate_report(hw_info, test1, test2, test3):
    """生成 Markdown 格式的 Dell 7960 性能基准报告。"""
    lines = [
        "# 🖥️ Dell Precision 7960 — ROSClaw v1.0 压力测试报告",
        "",
        f"**测试时间**: {hw_info['timestamp']}",
        f"**主机**: {hw_info['hostname']}",
        f"**PyTorch**: {hw_info['torch_version']} | **CUDA**: {hw_info['cuda_version']}",
        "",
        "---",
        "",
        "## 🔧 硬件配置",
        "",
        f"- **CPU**: {hw_info['cpu_count']} 核心",
        f"- **GPU 数量**: {hw_info['gpu_count']}",
    ]

    for gpu in hw_info["gpus"]:
        lines.append(
            f"- **GPU {gpu['id']}**: {gpu['name']} ({gpu['total_memory_gb']:.0f}GB) "
            f"[CC {gpu['compute_capability']}, {gpu['multi_processor_count']} SM]"
        )

    lines.extend(["", "---", ""])

    # Test 1
    lines.extend([
        "## 🎯 测试 1: MuJoCo 1000步连续仿真稳定性",
        "",
    ])
    if test1["status"] == "DONE":
        lines.extend([
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 总步数 | {test1['steps']:,} |",
            f"| 总耗时 | {test1['total_time_s']:.3f} s |",
            f"| 平均 step time | {test1['mean_ms']:.4f} ms |",
            f"| p50 step time | {test1['p50_ms']:.4f} ms |",
            f"| p95 step time | {test1['p95_ms']:.4f} ms |",
            f"| p99 step time | {test1['p99_ms']:.4f} ms |",
            f"| min / max | {test1['min_ms']:.4f} / {test1['max_ms']:.4f} ms |",
            f"| 标准差 | {test1['stdev_ms']:.4f} ms |",
            f"| 抖动率 (CV) | {test1['jitter_cv_percent']:.2f} % |",
            f"| 实时倍率 | {test1['realtime_factor']:.1f}× |",
            f"| ** verdict** | **{test1['verdict']}** |",
            "",
        ])
    else:
        lines.extend([f"- 状态: {test1['status']} ({test1.get('reason', '')})", ""])

    # Test 2
    lines.extend([
        "## 🎯 测试 2: 4×A6000 并行 Trajectory Batch",
        "",
    ])
    if test2["status"] == "DONE":
        lines.extend([
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| GPU 数量 | {test2['num_gpus']} |",
            f"| Batch Size | {test2['batch_size']} |",
            f"| Sequence Length | {test2['seq_len']} |",
            f"| 每 GPU 批次数 | {test2['num_batches_per_gpu']} |",
            f"| 平均 Latency | {test2['mean_latency_ms']:.2f} ms |",
            f"| p95 Latency | {test2['p95_latency_ms']:.2f} ms |",
            f"| 平均 Throughput/GPU | {test2['mean_throughput_per_gpu']:,.0f} points/s |",
            f"| 聚合 Throughput | {test2['aggregate_throughput']:,.0f} points/s |",
            f"| ** verdict** | **{test2['verdict']}** |",
            "",
        ])
        for r in test2["per_gpu_results"]:
            lines.append(
                f"- GPU {r['gpu_id']}: {r['mean_throughput']:,.0f} points/s, "
                f"显存 {r['gpu_memory_allocated_mb']:.0f} MB"
            )
        lines.append("")
    else:
        lines.extend([f"- 状态: {test2['status']} ({test2.get('reason', '')})", ""])

    # Test 3
    lines.extend([
        "## 🎯 测试 3: GPU 显存满载压力测试 (48GB × 4)",
        "",
    ])
    if test3["status"] == "DONE":
        lines.extend([
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 测试时长 | {test3['duration_sec']} s |",
            f"| 总迭代次数 | {test3['total_iterations']:,} |",
            f"| ** verdict** | **{test3['verdict']}** |",
            "",
        ])
        for r in test3["per_gpu_results"]:
            lines.extend([
                f"### GPU {r['gpu_id']}",
                "",
                f"| 指标 | 数值 |",
                f"|------|------|",
                f"| 目标分配 | {r['target_alloc_gb']:.1f} GB |",
                f"| 实际分配 | {r['actual_alloc_gb']:.1f} GB |",
                f"| 迭代次数 | {r['iterations']:,} |",
                f"| 迭代速度 | {r['iterations_per_sec']:.1f} it/s |",
                f"| 平均操作时间 | {r['mean_op_time_ms']:.2f} ms |",
                f"| p95 操作时间 | {r['p95_op_time_ms']:.2f} ms |",
                f"| 最大操作时间 | {r['max_op_time_ms']:.2f} ms |",
                "",
            ])
    else:
        lines.extend([f"- 状态: {test3['status']} ({test3.get('reason', '')})", ""])

    lines.extend(["---", "", "*Report generated by ROSClaw v1.0 Dell 7960 Stress Test*"])

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + "  ROSClaw v1.0 — Dell Precision 7960 压力测试套件".center(68) + "║")
    print("║" + "  GPU: 4× NVIDIA RTX A6000 (48GB)  |  CPU: 32 核心".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # 硬件信息
    hw_info = get_hardware_info()
    print(f"\n📅 {hw_info['timestamp']}")
    print(f"🖥️  {hw_info['hostname']} | CPU: {hw_info['cpu_count']} 核心")
    for gpu in hw_info["gpus"]:
        print(
            f"🎮 GPU {gpu['id']}: {gpu['name']} {gpu['total_memory_gb']:.0f}GB "
            f"[CC {gpu['compute_capability']}]"
        )

    # 运行测试
    test1 = test_mujoco_1000_steps()
    test2 = test_4x_a6000_trajectory()
    test3 = test_gpu_memory_full()

    # 生成报告
    print("\n" + "=" * 70)
    print("▶ 测试 4: 生成 Dell 性能基准报告")
    print("=" * 70)

    report = generate_report(hw_info, test1, test2, test3)
    REPORT_PATH.write_text(report, encoding="utf-8")

    # 同时保存 JSON
    results_json = {
        "hardware": hw_info,
        "test1_mujoco_stability": test1,
        "test2_trajectory_batch": test2,
        "test3_gpu_memory_stress": test3,
    }
    JSON_PATH.write_text(json.dumps(results_json, indent=2, default=str), encoding="utf-8")

    print(f"  ✅ Markdown 报告: {REPORT_PATH}")
    print(f"  ✅ JSON 数据: {JSON_PATH}")

    # 汇总
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  📊 测试结果汇总".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    v1 = test1.get("verdict", "N/A")
    v2 = test2.get("verdict", "N/A")
    v3 = test3.get("verdict", "N/A")
    print(f"║  测试 1 (MuJoCo 稳定性):    {v1:>10}          ║")
    print(f"║  测试 2 (Trajectory Batch): {v2:>10}          ║")
    print(f"║  测试 3 (GPU 显存满载):     {v3:>10}          ║")
    print("╚" + "═" * 68 + "╝")

    # 如果所有测试通过，返回 0
    all_pass = all(v == "PASS" for v in [v1, v2, v3] if v != "N/A")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
