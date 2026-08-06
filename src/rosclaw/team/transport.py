"""local_sim transport (PR-TF-070): in-process, deterministic, fault-injectable.

Models network reality for tests and SIM: latency, packet loss, and full
partition between members. NOT a production transport — ROS 2/DDS or
Zenoh adapters implement the same interface later.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class SimLink:
    latency_ms: float = 0.0
    loss_rate: float = 0.0  # 0..1
    partitioned: bool = False


@dataclass
class _Message:
    topic: str
    payload: dict
    deliver_after_ms: float


class LocalSimTransport:
    def __init__(self, *, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._links: dict[tuple[str, str], SimLink] = defaultdict(SimLink)
        self._queues: dict[str, deque] = defaultdict(deque)
        self._subscribers: dict[str, list[Callable[[str, dict], None]]] = defaultdict(list)
        self._clock_ms = 0.0
        self.dropped: list[tuple[str, str, str]] = []

    # ------------------------------------------------------------------
    def set_link(
        self, src: str, dst: str, *, latency_ms: float = 0.0, loss_rate: float = 0.0
    ) -> None:
        self._links[(src, dst)] = SimLink(latency_ms=latency_ms, loss_rate=loss_rate)

    def partition(self, members: list[str]) -> None:
        """Cut all links touching any member of the group."""
        for src, dst in list(self._links):
            if src in members or dst in members:
                self._links[(src, dst)].partitioned = True
        # Also cut default links.
        for src in members:
            for dst in self._queues:
                self._links[(src, dst)].partitioned = True
                self._links[(dst, src)].partitioned = True

    def heal(self) -> None:
        for link in self._links.values():
            link.partitioned = False

    def advance_time(self, ms: float) -> None:
        """Deliver all messages whose (latency-simulated) time has come."""
        self._clock_ms += ms
        for member, queue in self._queues.items():
            while queue and queue[0].deliver_after_ms <= self._clock_ms:
                message = queue.popleft()
                for handler in self._subscribers[message.topic]:
                    handler(member, message.payload)

    # ------------------------------------------------------------------
    def send(self, src: str, dst: str, topic: str, payload: dict) -> bool:
        link = self._links[(src, dst)]
        if link.partitioned:
            self.dropped.append((src, dst, topic))
            return False
        if self._rng.random() < link.loss_rate:
            self.dropped.append((src, dst, topic))
            return False
        self._queues[dst].append(
            _Message(
                topic=topic, payload=payload, deliver_after_ms=self._clock_ms + link.latency_ms
            )
        )
        return True

    def subscribe(self, topic: str, handler: Callable[[str, dict], None]) -> None:
        self._subscribers[topic].append(handler)
