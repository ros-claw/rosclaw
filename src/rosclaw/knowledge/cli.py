"""Thin CLI adapter for versioned Know/How services."""

from __future__ import annotations

import argparse
import json
import uuid
from typing import Any

from .context_adapter import build_how_context
from .contracts import HowAdviceRequestV2, ReferenceContextV2, ResearchRequestV2
from .facade import KnowledgeFacade
from .feedback_adapter import build_usage_feedback
from .policy import RESEARCH_BUDGETS
from .service_manager import KnowledgeServiceConfig, KnowledgeServiceManager


def _print(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _facade() -> tuple[KnowledgeServiceManager, KnowledgeFacade]:
    manager = KnowledgeServiceManager(KnowledgeServiceConfig.from_env())
    return manager, KnowledgeFacade(manager)


def _know_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rosclaw know")
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("status")
    subs.add_parser("doctor")
    research = subs.add_parser("research")
    research.add_argument("topic")
    research.add_argument("--goal", default="find reusable projects and implementation references")
    research.add_argument("--depth", choices=["shallow", "standard", "deep"], default="shallow")
    research.add_argument("--max-sources", type=int)
    research.add_argument("--token-budget", type=int)
    discover = subs.add_parser("discover")
    discover.add_argument("query")
    discover.add_argument("--robot")
    discover.add_argument("--simulator")
    discover.add_argument("--ros-distro")
    discover.add_argument("--top-k", type=int, default=10)
    discover.add_argument("--token-budget", type=int, default=8000)
    explain = subs.add_parser("explain")
    explain.add_argument("query")
    explain.add_argument("--robot")
    explain.add_argument("--simulator")
    explain.add_argument("--ros-distro")
    explain.add_argument("--failure")
    explain.add_argument("--top-k", type=int, default=10)
    diff = subs.add_parser("diff")
    diff.add_argument("project_id")
    diff.add_argument("--from", dest="from_snapshot", required=True)
    diff.add_argument("--to", dest="to_snapshot", required=True)
    refresh = subs.add_parser("refresh")
    refresh.add_argument("source_id")
    refresh.add_argument("--apply", action="store_true")
    freeze = subs.add_parser("freeze")
    freeze.add_argument("--label", required=True)
    pack = subs.add_parser("reference-pack")
    pack_subs = pack.add_subparsers(dest="pack_action", required=True)
    build = pack_subs.add_parser("build")
    build.add_argument("--task", required=True)
    build.add_argument("--robot")
    build.add_argument("--simulator")
    build.add_argument("--ros-distro")
    return parser


def _how_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rosclaw how")
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("doctor")
    explain = subs.add_parser("explain")
    explain.add_argument("advice_id")
    for mode in ("discover", "consult", "diagnose", "catalyze"):
        command = subs.add_parser(mode)
        command.add_argument("query")
        command.add_argument("--task")
        command.add_argument("--robot")
        command.add_argument("--simulator")
        command.add_argument("--ros-distro")
        command.add_argument("--stage")
        command.add_argument("--failure")
        command.add_argument("--top-k", type=int, default=8)
        command.add_argument("--token-budget", type=int, default=8000)
    feedback = subs.add_parser("feedback")
    feedback.add_argument("advice_id")
    feedback.add_argument("--reference-pack-id", required=True)
    feedback.add_argument("--knowledge-unit-id", required=True)
    feedback.add_argument("--context-hash", required=True)
    feedback.add_argument(
        "--verdict",
        choices=["useful", "irrelevant", "stale", "incompatible", "misleading", "unknown"],
        required=True,
    )
    feedback.add_argument("--reason")
    return parser


def dispatch_knowledge_argv(argv: list[str]) -> int | None:
    """Handle v2-only subcommands and leave legacy know/how commands untouched."""

    if len(argv) < 2 or argv[0] not in {"know", "how"}:
        return None
    v2_actions = {
        "know": {
            "status",
            "doctor",
            "research",
            "discover",
            "explain",
            "diff",
            "refresh",
            "freeze",
            "reference-pack",
        },
        "how": {"doctor", "explain", "discover", "consult", "diagnose", "catalyze", "feedback"},
    }
    if argv[1] not in v2_actions[argv[0]]:
        return None
    args = (_know_parser() if argv[0] == "know" else _how_parser()).parse_args(argv[1:])
    manager, facade = _facade()
    try:
        if argv[0] == "know":
            if args.action == "status":
                _print(facade.health())
                return 0
            if args.action == "doctor":
                _print(facade.know_doctor())
                return 0
            if args.action == "research":
                budget = RESEARCH_BUDGETS[args.depth]
                request = ResearchRequestV2(
                    request_id=f"research_{uuid.uuid4().hex[:24]}",
                    topic=args.topic,
                    goal=args.goal,
                    depth=args.depth,
                    max_sources=args.max_sources or budget.max_sources,
                    token_budget=args.token_budget or budget.max_tokens,
                )
                _print(facade.research(request))
                return 0
            if args.action == "explain":
                _print(
                    facade.know_explain(
                        query=args.query,
                        context=ReferenceContextV2(
                            robot=args.robot,
                            simulator=args.simulator,
                            ros_distro=args.ros_distro,
                            current_failure=args.failure,
                        ),
                        top_k=args.top_k,
                    )
                )
                return 0
            if args.action == "diff":
                _print(
                    facade.know_diff(
                        project_id=args.project_id,
                        from_snapshot=args.from_snapshot,
                        to_snapshot=args.to_snapshot,
                    )
                )
                return 0
            if args.action == "refresh":
                _print(facade.know_refresh(source_id=args.source_id, apply=args.apply))
                return 0
            if args.action == "freeze":
                _print(facade.know_freeze(label=args.label))
                return 0
            query = args.query if args.action == "discover" else args.task
            context = ReferenceContextV2(
                task=query,
                robot=args.robot,
                simulator=args.simulator,
                ros_distro=args.ros_distro,
            )
            _print(
                facade.reference_pack(
                    query=query,
                    context=context,
                    top_k=getattr(args, "top_k", 10),
                    token_budget=getattr(args, "token_budget", 8000),
                )
            )
            return 0
        if args.action == "doctor":
            _print(facade.how_doctor())
            return 0
        if args.action == "explain":
            _print(facade.how_explain(args.advice_id))
            return 0
        if args.action == "feedback":
            feedback = build_usage_feedback(
                reference_pack_id=args.reference_pack_id,
                advice_id=args.advice_id,
                knowledge_unit_id=args.knowledge_unit_id,
                context_hash=args.context_hash,
                verdict=args.verdict,
                reason=args.reason,
                origin="user",
            )
            _print({"created": facade.feedback(feedback), "feedback_id": feedback.feedback_id})
            return 0
        context = build_how_context(
            task=args.task or args.query,
            robot_model=args.robot,
            simulator=args.simulator,
            ros_distro=args.ros_distro,
            current_stage=args.stage,
            current_failure=args.failure,
            error_log=args.query if args.action == "diagnose" else None,
        )
        request = HowAdviceRequestV2(
            request_id=f"advice_request_{uuid.uuid4().hex[:24]}",
            mode=args.action,
            query=args.query,
            context=context,
            top_k=args.top_k,
            token_budget=args.token_budget,
        )
        _print(facade.advise(request))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _print({"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"})
        return 2
    finally:
        manager.close()
