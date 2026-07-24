"""Product CLI entry points for the Phase 3 Failure-to-Success Arena."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosclaw.simforge.evolution_arena import export_evolution_showcase
from rosclaw.simforge.gazebo_guarded_base import run_gazebo_guarded_base
from rosclaw.simforge.phase3_run import (
    Phase3RunProfile,
    run_contact_push_phase3,
)
from rosclaw.simforge.proof import ModuleEvidenceLevel


def dispatch_phase3_argv(argv: list[str]) -> int | None:
    if _is_demo_command(argv):
        return _run_demo(argv)
    if _is_proof_show(argv):
        return _show_proof(argv)
    if _is_proof_run(argv):
        return _run_proof_suite(argv)
    if _is_evolution_export(argv):
        return _export_evolution(argv)
    if _is_gazebo_chaos(argv):
        return _run_gazebo_chaos(argv)
    return None


def _is_demo_command(argv: list[str]) -> bool:
    return len(argv) >= 3 and argv[:3] == ["demo", "run", "failure-to-success"]


def _is_proof_show(argv: list[str]) -> bool:
    return len(argv) >= 2 and argv[:2] == ["proof", "show"]


def _is_proof_run(argv: list[str]) -> bool:
    return len(argv) >= 3 and argv[:3] == ["proof", "run", "module-causal-v1"]


def _is_evolution_export(argv: list[str]) -> bool:
    return len(argv) >= 2 and argv[:2] == ["evolution", "export"]


def _is_gazebo_chaos(argv: list[str]) -> bool:
    return len(argv) >= 3 and argv[:3] == ["chaos", "run", "gazebo-guarded-base"]


def _run_demo(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw demo run failure-to-success")
    parser.add_argument("demo")
    parser.add_argument("run")
    parser.add_argument("name")
    parser.add_argument("--task", choices=("contact_push",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--root-seed", type=int, default=20260723)
    parser.add_argument("--live-dashboard", action="store_true")
    args = parser.parse_args(argv)
    profile = Phase3RunProfile.full() if args.profile == "full" else Phase3RunProfile.smoke()
    result = run_contact_push_phase3(
        output_root=args.output_dir,
        source_checkout=_source_checkout(),
        profile=profile,
        root_seed=args.root_seed,
    )
    summary = result.summary_dict()
    showcase = None
    if args.live_dashboard:
        showcase = export_evolution_showcase(
            run_report=result.output_root / "phase3-run.json",
            output_dir=result.output_root / "showcase",
            source_checkout=_source_checkout(),
        )
    print(
        json.dumps(
            {
                "decision": result.champion_promotion.decision.value,
                "profile": profile.name,
                "full_acceptance_run": profile.name == "full",
                "candidate_hash": summary["candidate_hash"],
                "dataset_snapshot_hash": summary["dataset_snapshot_hash"],
                "same_seed_retry_passed": summary["same_seed_retry_passed"],
                "stress_worlds": result.champion_stress.worlds,
                "active_candidate_hash": (result.activation.final_active_candidate_hash),
                "canary_rollback": result.activation.canary.frozen,
                "proof_bundle_hash": result.final_proofs.bundle_hash,
                "report": str(result.output_root / "phase3-run.json"),
                "raw_evidence_manifest": str(result.raw_evidence_manifest_path),
                "raw_evidence_manifest_hash": result.raw_evidence_manifest_hash,
                "dashboard_requested": bool(args.live_dashboard),
                "showcase": showcase,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _export_evolution(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw evolution export")
    parser.add_argument("evolution")
    parser.add_argument("export")
    parser.add_argument("evolution_id")
    parser.add_argument("--format", choices=("showcase",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = Path(args.evolution_id).expanduser().resolve()
    if source.is_dir():
        source = source / "phase3-run.json"
    exported = export_evolution_showcase(
        run_report=source,
        output_dir=args.output,
        source_checkout=_source_checkout(),
    )
    print(json.dumps(exported, indent=2, sort_keys=True))
    return 0


def _run_gazebo_chaos(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw chaos run gazebo-guarded-base")
    parser.add_argument("chaos")
    parser.add_argument("run")
    parser.add_argument("name")
    parser.add_argument(
        "--faults",
        default="agent-kill,rosbridge-loss,odom-stale,worker-crash",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image",
        default="rosclaw/ros2-humble-gazebo:latest",
    )
    parser.add_argument("--rosbridge-port", type=int, default=0)
    args = parser.parse_args(argv)
    requested = {item.strip() for item in args.faults.split(",") if item.strip()}
    supported = {"agent-kill", "rosbridge-loss", "odom-stale", "worker-crash"}
    unknown = requested - supported
    if unknown:
        raise SystemExit(f"unsupported Gazebo chaos faults: {', '.join(sorted(unknown))}")
    if requested != supported:
        raise SystemExit(
            "Phase 3 acceptance requires all faults: "
            "agent-kill,rosbridge-loss,odom-stale,worker-crash"
        )
    result = run_gazebo_guarded_base(
        output_dir=args.output_dir,
        source_checkout=_source_checkout(),
        image=args.image,
        rosbridge_port=args.rosbridge_port,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _run_proof_suite(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw proof run module-causal-v1")
    parser.add_argument("proof")
    parser.add_argument("run")
    parser.add_argument("suite")
    parser.add_argument("--modules", default="")
    parser.add_argument("--task", choices=("contact_push",), default="contact_push")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--root-seed", type=int, default=20260723)
    args = parser.parse_args(argv)
    requested = tuple(item.strip() for item in args.modules.split(",") if item.strip())
    result = run_contact_push_phase3(
        output_root=args.output_dir,
        source_checkout=_source_checkout(),
        profile=(Phase3RunProfile.full() if args.profile == "full" else Phase3RunProfile.smoke()),
        root_seed=args.root_seed,
    )
    if requested:
        result.final_proofs.require_levels(
            minimum=_decision_impact_level(),
            modules=requested,
        )
    print(json.dumps(result.final_proofs.to_dict(), indent=2, sort_keys=True))
    return 0


def _show_proof(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw proof show")
    parser.add_argument("proof")
    parser.add_argument("show")
    parser.add_argument("run")
    parser.add_argument("--tree", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.run).expanduser().resolve()
    if path.is_dir():
        candidates = (
            path / "11-final-proofs" / "proof-bundle-final.json",
            path / "proof-bundle-final.json",
            path / "proof-bundle.json",
        )
        path = next((item for item in candidates if item.is_file()), path)
    if not path.is_file():
        raise SystemExit(f"proof artifact not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not args.tree:
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    proofs = value.get("proofs")
    if not isinstance(proofs, list):
        raise SystemExit("proof artifact does not contain a proof list")
    lines = [f"{value.get('run_id', 'unknown')} [{value.get('bundle_hash', 'unhashed')}]"]
    for index, proof in enumerate(
        sorted(proofs, key=lambda item: str(item.get("module"))),
        start=1,
    ):
        branch = "└─" if index == len(proofs) else "├─"
        lines.append(
            f"{branch} {proof.get('module', 'unknown')}: "
            f"{proof.get('level', 'E?')} "
            f"(impact={bool(proof.get('decision_impact', {}).get('changed'))}, "
            f"replay={bool(proof.get('replay_verified'))})"
        )
    print("\n".join(lines))
    return 0


def _decision_impact_level() -> ModuleEvidenceLevel:
    return ModuleEvidenceLevel.DECISION_IMPACT


def _source_checkout() -> Path:
    return Path(__file__).resolve().parents[3]


__all__ = ["dispatch_phase3_argv"]
