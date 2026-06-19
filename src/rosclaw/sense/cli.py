"""CLI commands for ``rosclaw sense``."""

from __future__ import annotations

import argparse
import json
import sys
import time

from rosclaw.sense.interface import SenseInterface


def _make_sense(args: argparse.Namespace) -> SenseInterface:
    return SenseInterface(
        robot_id=getattr(args, "robot_id", "g1_lab_01"),
        collector="mock" if getattr(args, "mock", None) else "mock",
        scenario=getattr(args, "mock", "normal") or "normal",
    )


def cmd_sense_now(args: argparse.Namespace) -> int:
    """Get current BodySense snapshot."""
    sense = _make_sense(args)
    sense.initialize()
    try:
        snapshot = sense.get_body_sense()
        if getattr(args, "json", False):
            print(json.dumps(snapshot.to_dict(), indent=2, default=str))
        else:
            print(f"Robot: {snapshot.robot_id}")
            print(f"Overall: {snapshot.overall_status.upper()}")
            print()
            print("Ready:")
            readiness = snapshot.readiness
            for name, item in sorted(readiness.capabilities.items()):
                icon = "✅" if item.status == "ready" else "❌"
                print(f"  {name:<22} {icon} {item.status}")
            print()
            print("Main risks:")
            for reason in snapshot.main_reasons:
                print(f"  - {reason}")
            if not snapshot.main_reasons:
                print("  (none)")
            print()
            print("Recommendation:")
            if snapshot.recommended_actions:
                for action in snapshot.recommended_actions:
                    print(f"  - {action}")
            else:
                print(f"  {snapshot.natural_language_summary}")
    finally:
        sense.stop()
    return 0


def cmd_sense_state(args: argparse.Namespace) -> int:
    """Show detailed raw BodyState."""
    sense = _make_sense(args)
    sense.initialize()
    try:
        state = sense.get_body_state()
        if getattr(args, "json", False):
            print(json.dumps(state.to_dict(), indent=2, default=str))
        else:
            print(f"Robot: {state.robot_id}")
            print(f"Source: {state.source}")
            print(f"Battery: {state.energy.battery_percent}%")
            print("Joint temperatures:")
            for name, joint in sorted(state.joints.items()):
                print(f"  {name}: {joint.temperature_c}C")
            print(f"Support margin: {state.balance.support_margin}")
            print(f"DDS latency: {state.communication.dds_latency_ms}ms")
            print(f"Target confidence: {state.perception.target_detector_confidence}")
    finally:
        sense.stop()
    return 0


def cmd_sense_readiness(args: argparse.Namespace) -> int:
    """Show body readiness for a task."""
    sense = _make_sense(args)
    sense.initialize()
    try:
        task = getattr(args, "task", None)
        if not task:
            print("Error: --task is required", file=sys.stderr)
            return 1
        readiness = sense.get_readiness(task=task)
        if getattr(args, "json", False):
            print(json.dumps(readiness.to_dict(), indent=2, default=str))
        else:
            item = readiness.capabilities.get(task)
            status = item.status if item else readiness.overall_status
            print(f"Task readiness: {task} = {status.upper()}")
            print()
            if item and item.failed_requirements:
                print("Failed requirements:")
                for req in item.failed_requirements:
                    print(
                        f"  - {req.name}: required {req.required}, current {req.current}"
                    )
                print()
            if item and item.allowed_alternatives:
                print("Allowed alternatives:")
                for alt in item.allowed_alternatives:
                    print(f"  - {alt}")
                print()
            print(sense.explain_block(task))
    finally:
        sense.stop()
    return 0


def cmd_sense_watch(args: argparse.Namespace) -> int:
    """Watch body sense stream."""
    sense = _make_sense(args)
    sense.initialize()
    interval = getattr(args, "interval", 1.0)
    limit = getattr(args, "limit", None)
    try:
        count = 0
        while True:
            snapshot = sense.get_body_sense()
            state = sense.get_body_state()
            right_knee = state.joints.get("right_knee")
            knee_temp = right_knee.temperature_c if right_knee is not None else "N/A"
            line = (
                f"battery {state.energy.battery_percent}% | "
                f"right_knee {knee_temp}C | "
                f"status {snapshot.overall_status.upper()} | "
                f"blocked {snapshot.blocked_capabilities}"
            )
            print(line)
            count += 1
            if limit is not None and count >= limit:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        sense.stop()
    return 0


def cmd_sense_events(args: argparse.Namespace) -> int:
    """Show recent BodyEvents."""
    sense = _make_sense(args)
    sense.initialize()
    try:
        limit = getattr(args, "limit", 20)
        events = sense._runtime.get_events(limit=limit)
        if getattr(args, "json", False):
            print(json.dumps([e.to_dict() for e in events], indent=2, default=str))
        else:
            print(f"Recent body events (last {len(events)}):")
            for event in events:
                print(
                    f"  [{event.severity}] {event.type}: {event.affected_parts}"
                )
    finally:
        sense.stop()
    return 0


def cmd_sense_explain(args: argparse.Namespace) -> int:
    """Explain why a task is blocked or degraded."""
    sense = _make_sense(args)
    sense.initialize()
    try:
        task = getattr(args, "task", None)
        if not task:
            print("Error: --task is required", file=sys.stderr)
            return 1
        print(sense.explain_block(task))
    finally:
        sense.stop()
    return 0
