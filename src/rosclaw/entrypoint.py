"""Console entrypoint with a lightweight product-command fast path."""

from __future__ import annotations

import sys


def main() -> int:
    """Dispatch product workflows without importing the full legacy CLI."""

    from rosclaw.daemon.cli import dispatch_daemon_argv

    result = dispatch_daemon_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.robot_pack.cli import dispatch_robot_pack_argv

    result = dispatch_robot_pack_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.product.cli import dispatch_product_argv

    result = dispatch_product_argv(sys.argv[1:])
    if result is not None:
        return result

    from rosclaw.cli import main as legacy_main

    return legacy_main()


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
