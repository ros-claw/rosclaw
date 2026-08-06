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
    research = subs.add_parser("research")
    research.add_argument("topic")
    research.add_argument("--goal", default="find reusable projects and implementation references")
    research.add_argument("--depth", choices=["shallow", "standard", "deep"], default="standard")
    research.add_argument("--max-sources", type=int, default=20)
    discover = subs.add_parser("discover")
    discover.add_argument("query")
    discover.add_argument("--robot")
    discover.add_argument("--simulator")
    discover.add_argument("--ros-distro")
    discover.add_argument("--top-k", type=int, default=10)
    discover.add_argument("--token-budget", type=int, default=8000)
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
        "know": {"status", "research", "discover", "reference-pack"},
        "how": {"discover", "consult", "diagnose", "catalyze", "feedback"},
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
            if args.action == "research":
                request = ResearchRequestV2(
                    request_id=f"research_{uuid.uuid4().hex[:24]}",
                    topic=args.topic,
                    goal=args.goal,
                    depth=args.depth,
                    max_sources=args.max_sources,
                )
                _print(facade.research(request))
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
