"""ROSClaw-native CLI for the RH56 RPS demo."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Importing this module registers the runtime handler.
from rosclaw_rps.rosclaw_integration import RosclawRpsSession


def _load_rosclaw_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Resolve ~ in data_root / db_path now so downstream Path handling is simple.
    for section, key in (("practice", "data_root"), ("practice", "seekdb"), ("memory", "db_path")):
        if section in data and isinstance(data[section], dict):
            if key == "seekdb":
                seekdb = data[section].get("seekdb")
                if isinstance(seekdb, dict) and "fallback_dir" in seekdb:
                    seekdb["fallback_dir"] = str(Path(seekdb["fallback_dir"]).expanduser())
            elif key in data[section]:
                data[section][key] = str(Path(data[section][key]).expanduser())
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RH56 RPS Demo — ROSClaw runtime version")
    parser.add_argument(
        "--mode",
        choices=["mock", "hand-test", "full"],
        default="mock",
        help="Demo mode",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "configs" / "dual",
        help="Directory containing gesture YAML and rps_rosclaw.yaml",
    )
    parser.add_argument(
        "--rosclaw-config",
        type=Path,
        default=None,
        help="Path to rps_rosclaw.yaml (default: CONFIG_DIR/rps_rosclaw.yaml)",
    )
    parser.add_argument("--rounds", type=int, default=None, help="Override number of rounds")
    parser.add_argument("--auto", action="store_true", help="Auto-advance rounds")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV UI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    rosclaw_config_path = args.rosclaw_config or args.config_dir / "rps_rosclaw.yaml"
    rosclaw_config = _load_rosclaw_config(rosclaw_config_path)

    session = RosclawRpsSession(config_dir=args.config_dir, rosclaw_config=rosclaw_config)
    try:
        session.initialize()
        session.start()
        parameters = {
            "mode": args.mode,
            "rounds": args.rounds,
            "auto": args.auto,
            "headless": args.headless,
        }
        result = session.run_skill(parameters)
    finally:
        session.stop()

    practice_id = session.active_practice_id
    # The top-level result comes from SkillExecutor; the actual skill output is
    # nested under handler_result when a runtime handler is invoked.
    skill_result = result.get("handler_result", result)
    print("\n=== ROSClaw RPS run result ===")
    print(f"status: {skill_result.get('status', result.get('status'))}")
    print(f"mode:   {skill_result.get('mode')}")
    if practice_id:
        print(f"practice_id: {practice_id}")
        data_root = Path(rosclaw_config.get("practice", {}).get("data_root", "~/.rosclaw/practice/runs/rh56_rps")).expanduser()
        session_dir = data_root / "sessions" / practice_id
        print(f"session_dir: {session_dir}")
        print(f"catalog:     {data_root / 'indexes' / 'practice_catalog.sqlite'}")
    print(f"summary: {skill_result.get('summary')}")

    return 0 if skill_result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
