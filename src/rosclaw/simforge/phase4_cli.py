"""Product CLI for the G1 GoalForge Phase 4 workflows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rosclaw.auto.g1_kick.continual_runner import (
    run_goalforge_continual_screening,
)
from rosclaw.robot_pack.g1.chaos import run_g1_executor_chaos
from rosclaw.robot_pack.g1.dds_adapter import run_unitree_dds_loopback
from rosclaw.simforge.backends.unitree_mujoco_backend import qualify_g1_assets
from rosclaw.simforge.g1_cpu_gpu_agreement import run_cpu_gpu_label_agreement
from rosclaw.simforge.g1_doctor import doctor_goalforge, write_doctor_report
from rosclaw.simforge.g1_four_gpu import run_goalforge_four_gpu
from rosclaw.simforge.g1_memory_ablation import run_memory_ablation
from rosclaw.simforge.g1_promotion_run import evaluate_goalforge_promotion
from rosclaw.simforge.g1_proof_replay import replay_goalforge_proof_bundle
from rosclaw.simforge.g1_proofs import build_goalforge_e5_proof_bundle
from rosclaw.simforge.g1_recovery_validation import (
    run_physical_recovery_validation,
)
from rosclaw.simforge.g1_success_validation import (
    run_nominal_success_validation,
)
from rosclaw.simforge.g1_video import render_goalforge_video
from rosclaw.simforge.phase4_run import (
    run_goalforge_demo,
    run_goalforge_practice_flywheel,
)
from rosclaw.simforge.tasks.g1_goalforge.showcase import render_showcase_html


def dispatch_phase4_argv(argv: list[str]) -> int | None:
    if len(argv) >= 3 and argv[:3] == ["simforge", "doctor", "g1-goalforge"]:
        return _doctor(argv)
    if len(argv) >= 3 and argv[:3] == ["simforge", "validate", "g1-goalforge"]:
        return _recovery_validation(argv)
    if len(argv) >= 3 and argv[:3] == ["demo", "run", "g1-goalforge"]:
        return _demo(argv)
    if _is_goalforge_practice(argv):
        return _practice(argv)
    if _is_goalforge_dataset(argv):
        return _dataset(argv)
    if _is_goalforge_twin(argv):
        return _twin(argv)
    if _is_goalforge_evolution(argv):
        return _evolution(argv)
    if len(argv) >= 3 and argv[:3] == ["chaos", "run", "g1-goalforge"]:
        return _chaos(argv)
    if len(argv) >= 3 and argv[:3] == ["proof", "build", "g1-goalforge"]:
        return _proof_build(argv)
    if len(argv) >= 3 and argv[:3] == [
        "promotion",
        "evaluate",
        "g1-goalforge",
    ]:
        return _promotion(argv)
    if len(argv) >= 3 and argv[:2] == ["proof", "replay"]:
        return _proof_replay(argv)
    if _is_goalforge_export(argv):
        return _export(argv)
    return None


def _doctor(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw simforge doctor g1-goalforge")
    parser.add_argument("simforge")
    parser.add_argument("doctor")
    parser.add_argument("name")
    parser.add_argument("--all", action="store_true")
    _add_reference_arguments(parser)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = doctor_goalforge(
        asset_root=args.asset_root,
        mjlab_root=args.mjlab_root,
        unitree_mujoco_root=args.unitree_mujoco_root,
        isaaclab_root=args.isaaclab_root if args.all else None,
    )
    if args.output:
        write_doctor_report(report, args.output)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 2


def _recovery_validation(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw simforge validate g1-goalforge")
    parser.add_argument("simforge")
    parser.add_argument("validate")
    parser.add_argument("name")
    parser.add_argument(
        "--profile",
        choices=("recovery", "nominal-success"),
        default="recovery",
    )
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--asset-root", type=Path, default=_default_robonaldo())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.profile == "nominal-success":
        result = run_nominal_success_validation(
            asset_root=args.asset_root,
            output_path=args.output,
            source_checkout=_source_checkout(),
            workers=args.workers,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.passed else 2
    result = run_physical_recovery_validation(
        asset_root=args.asset_root,
        output_path=args.output,
        pair_count=args.pairs,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.passed else 2


def _demo(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw demo run g1-goalforge")
    parser.add_argument("demo")
    parser.add_argument("run")
    parser.add_argument("name")
    parser.add_argument("--target-zone", choices=("random",), default="random")
    parser.add_argument("--failure-to-success", action="store_true")
    parser.add_argument("--live-dashboard", action="store_true")
    parser.add_argument("--asset-root", type=Path, default=_default_robonaldo())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_goalforge_demo(
        asset_root=args.asset_root,
        output_dir=args.output_dir,
        source_checkout=_source_checkout(),
    )
    value = result.to_dict()
    value["live_dashboard_manifest"] = (
        str(args.output_dir.resolve() / "showcase.json") if args.live_dashboard else None
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if result.passed else 2


def _practice(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw practice start")
    parser.add_argument("practice")
    parser.add_argument("start")
    parser.add_argument("--task", choices=("g1_penalty_kick",), required=True)
    parser.add_argument(
        "--sources",
        default="dds,imu,contact,ball,agent,sandbox,runtime",
    )
    parser.add_argument("--generation", type=int, default=3)
    parser.add_argument("--asset-root", type=Path, default=_default_robonaldo())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    required = {"dds", "imu", "contact", "ball", "agent", "sandbox", "runtime"}
    if {item.strip() for item in args.sources.split(",")} != required:
        raise SystemExit("GoalForge Practice requires dds,imu,contact,ball,agent,sandbox,runtime")
    result = run_goalforge_practice_flywheel(
        asset_root=args.asset_root,
        output_dir=args.output_dir,
        source_checkout=_source_checkout(),
        generation=args.generation,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.passed else 2


def _dataset(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw practice dataset build")
    parser.add_argument("practice")
    parser.add_argument("dataset")
    parser.add_argument("build")
    parser.add_argument("--task", choices=("g1_penalty_kick",), required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--asset-root", type=Path, default=_default_robonaldo())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_goalforge_practice_flywheel(
        asset_root=args.asset_root,
        output_dir=args.output_dir,
        source_checkout=_source_checkout(),
        generation=args.generation,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.passed else 2


def _twin(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw twin calibrate")
    parser.add_argument("twin")
    parser.add_argument("calibrate")
    parser.add_argument("--task", choices=("g1_penalty_kick",), required=True)
    parser.add_argument("--practice-id", type=Path, required=True)
    args = parser.parse_args(argv)
    path = args.practice_id.expanduser().resolve()
    if path.is_dir():
        path = path / "goalforge-demo.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    twin = value.get("twin")
    if not isinstance(twin, dict) or not twin.get("update"):
        raise SystemExit("Practice evidence does not contain a GoalForge Twin update")
    print(json.dumps(twin, indent=2, sort_keys=True))
    return 0


def _evolution(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw evolution run")
    parser.add_argument("evolution")
    parser.add_argument("run")
    parser.add_argument("--task", choices=("g1_penalty_kick",), required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--count-per-gpu", type=int, default=250)
    parser.add_argument("--asset-root", type=Path, default=_default_robonaldo())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.gpus != "0,1,2,3":
        raise SystemExit("GoalForge acceptance requires physical GPUs 0,1,2,3")
    root = args.output_dir.expanduser().resolve()
    result = run_goalforge_four_gpu(
        output_dir=root,
        source_checkout=_source_checkout(),
        count_per_gpu=args.count_per_gpu,
        root_seed=20260723 + args.generation,
    )
    assets = qualify_g1_assets(args.asset_root)
    assets.require_eligible()
    memory = run_memory_ablation(
        output_path=root / "memory-ablation-100.json",
        body_hash=assets.body_hash,
    )
    continual = run_goalforge_continual_screening(
        four_gpu_root=root,
        output_path=root / "continual-g0-g10.json",
    )
    agreement = run_cpu_gpu_label_agreement(
        asset_root=args.asset_root,
        four_gpu_root=root,
        output_path=root / "cpu-gpu-label-agreement.json",
    )
    summary = {
        "four_gpu": result.to_dict(),
        "memory_ablation": memory.to_dict(),
        "continual": continual.to_dict(),
        "cpu_gpu_label_agreement": agreement.to_dict(),
        "passed": (result.passed and memory.passed and continual.passed and agreement.passed),
    }
    (root / "goalforge-evolution.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


def _chaos(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw chaos run g1-goalforge")
    parser.add_argument("chaos")
    parser.add_argument("run")
    parser.add_argument("name")
    parser.add_argument(
        "--faults",
        default="agent-kill,worker-crash,dds-loss,state-stale",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    _add_reference_arguments(parser)
    parser.add_argument("--domain-id", type=int, default=73)
    args = parser.parse_args(argv)
    required = {"agent-kill", "worker-crash", "dds-loss", "state-stale"}
    if not required.issubset({item.strip() for item in args.faults.split(",")}):
        raise SystemExit("GoalForge chaos requires agent-kill,worker-crash,dds-loss,state-stale")
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    assets = qualify_g1_assets(args.asset_root)
    assets.require_eligible()
    dds = run_unitree_dds_loopback(
        unitree_mujoco_root=args.unitree_mujoco_root,
        output_dir=root / "dds",
        source_checkout=_source_checkout(),
        domain_id=args.domain_id,
    )
    chaos = run_g1_executor_chaos(
        output_dir=root / "executor",
        source_checkout=_source_checkout(),
        body_hash=assets.body_hash,
    )
    value = {
        "dds": dds.to_dict(),
        "executor": chaos.to_dict(),
        "passed": dds.passed and chaos.passed,
    }
    (root / "goalforge-chaos.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["passed"] else 2


def _proof_replay(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw proof replay")
    parser.add_argument("proof")
    parser.add_argument("replay")
    parser.add_argument("evolution_id", type=Path)
    parser.add_argument("--modules", default="body,memory,know,how,auto")
    args = parser.parse_args(argv)
    root = args.evolution_id.expanduser().resolve()
    requested = tuple(item.strip() for item in args.modules.split(",") if item.strip())
    result = replay_goalforge_proof_bundle(
        root,
        requested_modules=requested,
    )
    value = result.to_dict()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if result.passed else 2


def _proof_build(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw proof build g1-goalforge")
    parser.add_argument("proof")
    parser.add_argument("build")
    parser.add_argument("name")
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--flywheel", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--four-gpu", type=Path, required=True)
    parser.add_argument("--agreement", type=Path, required=True)
    parser.add_argument("--continual", type=Path, required=True)
    parser.add_argument("--chaos", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle = build_goalforge_e5_proof_bundle(
        demo_path=args.demo,
        recovery_path=args.recovery,
        flywheel_path=args.flywheel,
        memory_path=args.memory,
        four_gpu_root=args.four_gpu,
        agreement_path=args.agreement,
        continual_path=args.continual,
        chaos_path=args.chaos,
        output_dir=args.output_dir,
        source_checkout=_source_checkout(),
    )
    value = bundle.to_dict()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _promotion(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw promotion evaluate g1-goalforge")
    parser.add_argument("promotion")
    parser.add_argument("evaluate")
    parser.add_argument("name")
    parser.add_argument("--doctor", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--flywheel", type=Path, required=True)
    parser.add_argument("--four-gpu", type=Path, required=True)
    parser.add_argument("--continual", type=Path, required=True)
    parser.add_argument("--chaos", type=Path, required=True)
    parser.add_argument("--proofs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_goalforge_promotion(
        doctor_path=args.doctor,
        recovery_path=args.recovery,
        flywheel_path=args.flywheel,
        four_gpu_root=args.four_gpu,
        continual_path=args.continual,
        chaos_path=args.chaos,
        proof_root=args.proofs,
        output_path=args.output,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.decision.value == "SIM_CHAMPION" else 2


def _export(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw evolution export")
    parser.add_argument("evolution")
    parser.add_argument("export")
    parser.add_argument("evolution_id", type=Path)
    parser.add_argument("--format", choices=("showcase", "video"), required=True)
    parser.add_argument("--asset-root", type=Path, default=_default_robonaldo())
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.evolution_id.expanduser().resolve()
    if args.format == "video":
        demo = source / "goalforge-demo.json" if source.is_dir() else source
        result = render_goalforge_video(
            demo_path=demo,
            asset_root=args.asset_root,
            output_path=args.output,
            source_checkout=_source_checkout(),
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if source.is_dir():
        source = source / "showcase.json"
    if not source.is_file():
        raise SystemExit(f"GoalForge showcase not found: {source}")
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / "showcase.json"
    target.write_bytes(source.read_bytes())
    manifest = json.loads(source.read_text(encoding="utf-8"))
    dashboard = args.output / "index.html"
    dashboard.write_text(
        render_showcase_html(manifest),
        encoding="utf-8",
    )
    value = {
        "format": "showcase",
        "manifest": str(target),
        "dashboard": str(dashboard),
        "source": str(source),
    }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _is_goalforge_practice(argv: list[str]) -> bool:
    return len(argv) >= 2 and argv[:2] == ["practice", "start"] and _task_is_goalforge(argv)


def _is_goalforge_dataset(argv: list[str]) -> bool:
    return (
        len(argv) >= 3 and argv[:3] == ["practice", "dataset", "build"] and _task_is_goalforge(argv)
    )


def _is_goalforge_twin(argv: list[str]) -> bool:
    return len(argv) >= 2 and argv[:2] == ["twin", "calibrate"] and _task_is_goalforge(argv)


def _is_goalforge_evolution(argv: list[str]) -> bool:
    return len(argv) >= 2 and argv[:2] == ["evolution", "run"] and _task_is_goalforge(argv)


def _is_goalforge_export(argv: list[str]) -> bool:
    if len(argv) < 3 or argv[:2] != ["evolution", "export"]:
        return False
    source = Path(argv[2]).expanduser()
    return source.name.startswith("goalforge") or (
        source.is_dir()
        and ((source / "showcase.json").is_file() or (source / "goalforge-demo.json").is_file())
    )


def _task_is_goalforge(argv: list[str]) -> bool:
    if "--task" not in argv:
        return False
    index = argv.index("--task")
    return index + 1 < len(argv) and argv[index + 1] == "g1_penalty_kick"


def _add_reference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--asset-root", type=Path, default=_default_robonaldo())
    parser.add_argument("--mjlab-root", type=Path, default=_reference_root() / "unitree_rl_mjlab")
    parser.add_argument(
        "--unitree-mujoco-root",
        type=Path,
        default=_reference_root() / "unitree_mujoco",
    )
    parser.add_argument(
        "--isaaclab-root",
        type=Path,
        default=_reference_root() / "unitree_sim_isaaclab",
    )


def _default_robonaldo() -> Path:
    configured = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    return Path(configured) if configured else _reference_root() / "RoboNaldo/RoboNaldo_Deploy"


def _reference_root() -> Path:
    configured = os.environ.get("ROSCLAW_SIMFORGE_REFERENCE_ROOT")
    return Path(configured) if configured else Path("/code/rosclaw/phase4_references")


def _source_checkout() -> Path:
    return Path(__file__).resolve().parents[3]


__all__ = ["dispatch_phase4_argv"]
