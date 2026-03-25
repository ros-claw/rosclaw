"""Built-in robot definitions for ROSClaw V4.

Pre-configured manifests for supported robot platforms.
"""

from __future__ import annotations

from rosclaw_core.builtins.so101 import (
    create_so101_manifest,
    create_so101_leader_manifest,
    create_so101_follower_manifest,
)

__all__ = [
    "create_so101_manifest",
    "create_so101_leader_manifest",
    "create_so101_follower_manifest",
]
