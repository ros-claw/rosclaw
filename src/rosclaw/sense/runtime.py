"""SenseRuntime - central runtime for the rosclaw.sense module."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from rosclaw.core.event_bus import EventBus
from rosclaw.core.lifecycle import LifecycleMixin
from rosclaw.sense.collectors import (
    DDSCollector,
    FileReplayCollector,
    MockCollector,
    ROS2Collector,
)
from rosclaw.sense.collectors.base import BodyStateCollector as BodyStateCollectorABC
from rosclaw.sense.config import SenseConfig
from rosclaw.sense.estimators import (
    HealthEstimator,
    ReadinessEvaluator,
    RiskEstimator,
)
from rosclaw.sense.events import (
    body_updated_event,
    capability_blocked_event,
    capability_degraded_event,
    event_detected_event,
    readiness_updated_event,
    state_updated_event,
)
from rosclaw.sense.explain import SenseExplainer
from rosclaw.sense.schemas import BodyReadiness, BodySense, BodyState
from rosclaw.sense.storage import SenseStorage
from rosclaw.sense.thresholds import get_capability_requirements, load_thresholds

logger = logging.getLogger("rosclaw.sense.runtime")


class SenseRuntime(LifecycleMixin):
    """Owns collectors, estimators, storage, and EventBus publishing.

    The runtime can operate in two modes:
      1. Background loop: ``start()`` spawns a thread that calls ``tick()`` at
         ``config.update_hz``.
      2. Manual: tests and one-shot tools call ``tick()`` directly.
    """

    def __init__(
        self,
        config: SenseConfig,
        event_bus: EventBus,
        robot_id: str | None = None,
    ):
        super().__init__()  # type: ignore[no-untyped-call]
        self.config = config
        self.event_bus = event_bus
        self.robot_id = robot_id or config.robot_id

        self.thresholds = load_thresholds(
            robot_family=config.robot_profile,
            override_path=config.thresholds_path,
        )
        self.capability_requirements = get_capability_requirements(
            robot_family=config.robot_profile,
            override_path=config.thresholds_path,
        )

        self._collector = self._create_collector()
        self._health_estimator = HealthEstimator()
        self._risk_estimator = RiskEstimator(self.thresholds)
        self._readiness_evaluator = ReadinessEvaluator(
            self.thresholds,
            self.capability_requirements,
        )
        self._explainer = SenseExplainer()
        self._storage = SenseStorage(
            max_event_history=config.max_event_history,
            jsonl_path=config.extra.get("jsonl_path"),
        )

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

    def _create_collector(self) -> BodyStateCollectorABC:
        collector_type = self.config.collector
        if collector_type == "mock":
            scenario = self.config.extra.get("scenario", "normal")
            return MockCollector(robot_id=self.robot_id, scenario=scenario)
        if collector_type == "file_replay":
            return FileReplayCollector(
                replay_path=self.config.replay_path,
                robot_id=self.robot_id,
            )
        if collector_type == "ros2":
            return ROS2Collector(robot_id=self.robot_id)
        if collector_type == "dds":
            return DDSCollector(robot_id=self.robot_id)
        raise ValueError(f"Unknown collector type: {collector_type}")

    def _do_initialize(self) -> None:
        self._collector.start()
        logger.info(
            "SenseRuntime initialized for %s with %s collector",
            self.robot_id,
            self.config.collector,
        )

    def _do_start(self) -> None:
        if self.config.update_hz <= 0:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="sense-runtime", daemon=True)
        self._thread.start()
        logger.info("SenseRuntime background loop started at %.2f Hz", self.config.update_hz)

    def _do_stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._collector.stop()
        logger.info("SenseRuntime stopped")

    def _loop(self) -> None:
        interval = 1.0 / self.config.update_hz
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001
                logger.exception("SenseRuntime tick failed: %s", e)
            self._stop_event.wait(interval)

    def tick(self) -> BodySense:
        """Collect state, evaluate risk/readiness, summarize, store, publish."""
        try:
            state = self._collector.collect()
        except Exception as e:  # noqa: BLE001
            logger.exception("Collector failed: %s", e)
            state = self._degraded_state()

        with self._lock:
            self._storage.update_state(state)

        if self.config.publish_events:
            self.event_bus.publish(state_updated_event(state))

        risk_summary, events = self._risk_estimator.evaluate(state)
        for event in events:
            self._storage.add_event(event)
            if self.config.publish_events:
                self.event_bus.publish(event_detected_event(event))

        readiness = self._readiness_evaluator.evaluate_all(
            state,
            risk_summary,
            self.capability_requirements,
        )

        body_sense = self._explainer.summarize(state, risk_summary, readiness)

        with self._lock:
            self._storage.update_sense(body_sense)

        if self.config.publish_events:
            self.event_bus.publish(body_updated_event(body_sense))
            self.event_bus.publish(readiness_updated_event(readiness))
            self._publish_capability_changes(body_sense)

        return body_sense

    def _degraded_state(self) -> BodyState:
        return BodyState(
            robot_id=self.robot_id,
            timestamp=time.time(),
            source="sense:degraded",
        )

    def _publish_capability_changes(self, body_sense: BodySense) -> None:
        for capability in body_sense.blocked_capabilities:
            self.event_bus.publish(
                capability_blocked_event(
                    capability=capability,
                    reason="body_sense_not_ready",
                    evidence={"overall_status": body_sense.overall_status},
                )
            )
        for capability in body_sense.degraded_capabilities:
            if capability not in body_sense.blocked_capabilities:
                self.event_bus.publish(
                    capability_degraded_event(
                        capability=capability,
                        reason="body_sense_degraded",
                        evidence={"overall_status": body_sense.overall_status},
                    )
                )

    def get_latest_state(self) -> BodyState | None:
        with self._lock:
            return self._storage.latest_state

    def get_body_state(self) -> BodyState | None:
        """Alias for :meth:`get_latest_state`."""
        return self.get_latest_state()

    def get_latest_sense(self) -> BodySense | None:
        with self._lock:
            return self._storage.latest_sense

    def get_body_sense(self) -> BodySense | None:
        """Alias for :meth:`get_latest_sense`."""
        return self.get_latest_sense()

    def get_readiness(
        self,
        task: str | None = None,
        requirements: dict[str, Any] | None = None,
    ) -> BodyReadiness:
        """Return readiness for ``task`` or all capabilities."""
        state = self.get_latest_state()
        if state is None:
            state = self._collector.collect()
        risk_summary, _ = self._risk_estimator.evaluate(state)
        if task:
            return self._readiness_evaluator.evaluate(
                state,
                risk_summary,
                task=task,
                requirements=requirements,
            )
        return self._readiness_evaluator.evaluate_all(
            state,
            risk_summary,
            self.capability_requirements,
        )

    def explain_block(self, task: str) -> str:
        readiness = self.get_readiness(task=task)
        return self._explainer.explain_block(task, readiness)

    def get_events(self, event_type: str | None = None, limit: int | None = None) -> list[Any]:
        return self._storage.get_events(event_type=event_type, limit=limit)
