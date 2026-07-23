#!/usr/bin/env python3
"""Convert a rollout practice session (rollout.* events) to a normalized
LeRobot episode (rosclaw.practice.normalized.v2).

The P5 rollout loop records rollout.observation.validated /
rollout.action.mapped events — NOT physical_feedback_event, so
`practice export --format lerobot` cannot consume these sessions
(measured: explicit "No physical_feedback_event" error).  The P2.1
deterministic dataset flow (normalized episode.json -> run_dataset_export)
IS the sanctioned path for them.

Usage:
    rollout_session_to_episode.py <session_dir> --out <episode_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

JOINTS = ["little", "ring", "middle", "index", "thumb", "thumb_rot"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", default="countdown_pose")
    args = parser.parse_args()

    session = Path(args.session_dir)
    events_path = session / "raw" / "events.jsonl"
    events = []
    with events_path.open() as fh:
        for line in fh:
            if line.strip():
                events.append(json.loads(line))

    observations: dict[int, dict] = {}
    actions: dict[int, dict] = {}
    seq = 0
    for ev in events:
        etype = ev.get("event_type")
        if etype == "rollout.observation.validated":
            observations[seq] = ev
            seq += 1
        elif etype == "rollout.action.mapped":
            actions[len(observations) - 1] = ev

    frames = []
    for index in sorted(observations):
        ev = observations[index]
        snap = (ev.get("payload") or {}).get("snapshot") or {}
        features = snap.get("features") or {}

        def values(feature: str, _features: dict = features) -> list[float]:
            block = _features.get(feature) or {}
            return [float(v) for v in (block.get("values") or [])]

        state = values("observation.state") or [0.0] * 6
        force = values("observation.force") or [0.0] * 6
        current = values("observation.current") or [0.0] * 6
        temp = values("observation.temperature") or [0.0] * 6

        action_ev = actions.get(index)
        action_vec = state
        if action_ev:
            payload = action_ev.get("payload") or {}
            mapped = payload.get("action") or payload.get("mapped") or {}
            vec = mapped.get("values") or mapped.get("vector")
            if isinstance(vec, list) and vec:
                action_vec = [float(v) for v in vec]

        frames.append(
            {
                "frame_index": index,
                "timestamp": (ev.get("timestamp_ns") or 0) / 1e9,
                "source_timestamp_ns": ev.get("timestamp_ns"),
                "clock_domain": "monotonic",
                "episode_time_sec": index * 0.2,
                "observation": {
                    "state": state,
                    "force": force,
                    "current": current,
                    "temperature": temp,
                },
                "action": action_vec,
                "done": index == len(observations) - 1,
                "success": True,
                "safety": {"decision": "ALLOW", "modified": False, "risk_score": 0.0, "reason_code": None},
                "failure": {"active": False, "code": "NONE", "severity": 0},
                "intervention": {"active": False, "source": "NONE", "confidence": None},
                "action_context": {"source": "POLICY", "was_clamped": False},
            }
        )

    practice_id = session.name
    episode = {
        "schema_version": "rosclaw.practice.normalized.v2",
        "episode_id": practice_id,
        "robot": {
            "robot_id": "rh56_rps_robot",
            "body_profile": "rh56_right_01",
            "body_yaml_path": "",
            "body_hash": "",
        },
        "task": {"text": args.task, "task_id": args.task},
        "fps": 5.0,
        "environment": "lab_tabletop",
        "operator": "acceptance_v1.0.1",
        "frames": frames,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "episode.json").write_text(json.dumps(episode, ensure_ascii=False, indent=1))
    print(f"wrote {out / 'episode.json'} with {len(frames)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
