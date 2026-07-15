"""Entry point for the persistent LeRobot policy worker subprocess.

Run as ``python3 -m rosclaw.integrations.lerobot.policy_worker_runtime``.
"""

from __future__ import annotations

import argparse
import sys

from rosclaw.integrations.lerobot.policy_worker_service import PolicyWorkerService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LeRobot persistent policy worker")
    parser.add_argument("--protocol-version", default="")
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)

    service = PolicyWorkerService(
        device=args.device,
        dtype=args.dtype,
        allow_network=args.allow_network,
    )
    service.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
