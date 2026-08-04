"""Console entrypoint with a lightweight product-command fast path."""

from __future__ import annotations

import sys


def main() -> int:
    """Dispatch product workflows without importing the full legacy CLI."""

    from rosclaw.operator.cli import dispatch_operator_argv

    result = dispatch_operator_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.daemon.cli import dispatch_daemon_argv

    result = dispatch_daemon_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.adapters.acp.cli import dispatch_acp_argv

    result = dispatch_acp_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.operatord.cli import dispatch_operatord_argv

    result = dispatch_operatord_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.robot_pack.cli import dispatch_robot_pack_argv

    result = dispatch_robot_pack_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.app.cli import dispatch_app_argv

    result = dispatch_app_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.collective.cli import dispatch_collective_argv

    result = dispatch_collective_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.dream.cli import dispatch_dream_argv

    result = dispatch_dream_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.continual.cli import dispatch_continual_argv

    result = dispatch_continual_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.simforge.g1_muscle_memory_cli import dispatch_muscle_memory_argv

    result = dispatch_muscle_memory_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.simforge.g1_hat_trick_cli import dispatch_hat_trick_argv

    result = dispatch_hat_trick_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.simforge.phase4_cli import dispatch_phase4_argv

    result = dispatch_phase4_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.simforge.phase3_cli import dispatch_phase3_argv

    result = dispatch_phase3_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.product.cli import dispatch_product_argv

    result = dispatch_product_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.simforge.cli import dispatch_simforge_argv

    result = dispatch_simforge_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.cli import main as legacy_main

    return legacy_main()


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
