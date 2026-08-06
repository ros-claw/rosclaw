"""
ROSClaw Swarm - Collaboration Grounding Engine

.. deprecated:: experimental_legacy
   Maturity: **experimental_legacy** (see docs/adr/0000 and docs/adr/0004).
   This module is an in-memory prototype: no network transport, no epoch/lease
   semantics, no partition handling, and a Raft-like skeleton without log
   replication. It is frozen — do not add features, and do not build new code
   on top of it. Multi-robot coordination is being rebuilt in ``rosclaw.team``
   (Team Fabric). This module remains only for backward-compatible imports
   until the migration adapter lands.

Multi-robot coordination through DDS Reflex Handshake.
Microsecond-level synchronization for swarm operations.
"""

from rosclaw.swarm.manager import SwarmRuntimeManager

__all__ = ["SwarmRuntimeManager"]

MATURITY = "experimental_legacy"
