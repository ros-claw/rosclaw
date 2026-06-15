"""ConsensusProtocol — Distributed consensus for multi-robot state agreement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Proposal:
    """A proposal in the consensus protocol."""
    agent_id: str
    key: str
    value: Any
    timestamp: float
    term: int = 0


@dataclass
class ConsensusEntry:
    """Consensus state for a single key."""
    key: str
    agreed_value: Any = None
    agreed_at: float = 0.0
    proposals: list[Proposal] = field(default_factory=list)
    quorum_size: int = 1


class RaftLikeConsensus:
    """Simplified Raft-like consensus for robot swarm state agreement.

    - Leaders propose state updates
    - Followers vote
    - Committed on quorum
    """

    def __init__(self, agent_id: str, peers: list[str], quorum: int | None = None):
        self.agent_id = agent_id
        self.peers = peers
        self.quorum = quorum or (len(peers) // 2 + 1)
        self._entries: dict[str, ConsensusEntry] = {}
        self._term = 0
        self._is_leader = False
        self._vote_callback: Callable | None = None

    def set_leader(self, is_leader: bool) -> None:
        """Set leader status."""
        self._is_leader = is_leader
        if is_leader:
            self._term += 1

    def propose(self, key: str, value: Any, timestamp: float) -> bool:
        """Propose a state update (leader only)."""
        if not self._is_leader:
            return False

        proposal = Proposal(
            agent_id=self.agent_id,
            key=key,
            value=value,
            timestamp=timestamp,
            term=self._term,
        )

        entry = self._entries.get(key)
        if entry is None:
            entry = ConsensusEntry(key=key, quorum_size=self.quorum)
            self._entries[key] = entry

        entry.proposals.append(proposal)
        return True

    def vote(self, key: str, proposer_id: str, value: Any, timestamp: float) -> bool:
        """Vote for a proposal (follower). Returns True if vote accepted."""
        entry = self._entries.get(key)
        if entry is None:
            entry = ConsensusEntry(key=key, quorum_size=self.quorum)
            self._entries[key] = entry

        # Accept if newer term or newer timestamp
        if entry.proposals:
            latest = max(entry.proposals, key=lambda p: (p.term, p.timestamp))
            if timestamp < latest.timestamp and self._term <= latest.term:
                return False

        entry.proposals.append(Proposal(
            agent_id=self.agent_id,
            key=key,
            value=value,
            timestamp=timestamp,
            term=self._term,
        ))
        return True

    def check_commit(self, key: str) -> bool:
        """Check if a key has reached consensus (quorum)."""
        entry = self._entries.get(key)
        if entry is None:
            return False

        if len(entry.proposals) >= entry.quorum_size:
            # Use most recent value
            latest = max(entry.proposals, key=lambda p: (p.term, p.timestamp))
            entry.agreed_value = latest.value
            entry.agreed_at = latest.timestamp
            return True
        return False

    def get(self, key: str) -> Any | None:
        """Get agreed value for a key."""
        entry = self._entries.get(key)
        if entry and entry.agreed_value is not None:
            return entry.agreed_value
        return None

    def get_all_committed(self) -> dict[str, Any]:
        """Get all committed consensus values."""
        return {
            key: entry.agreed_value
            for key, entry in self._entries.items()
            if entry.agreed_value is not None
        }
