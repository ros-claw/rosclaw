"""Lifecycle idempotency regression tests."""

from rosclaw.core.lifecycle import LifecycleMixin, LifecycleState


class _Module(LifecycleMixin):
    def __init__(self) -> None:
        super().__init__()
        self.starts = 0
        self.stops = 0

    def _do_start(self) -> None:
        self.starts += 1

    def _do_stop(self) -> None:
        self.stops += 1


def test_start_and_stop_are_idempotent() -> None:
    module = _Module()
    module.initialize()

    module.start()
    module.start()
    assert module.state is LifecycleState.RUNNING
    assert module.starts == 1

    module.stop()
    module.stop()
    assert module.state is LifecycleState.STOPPED
    assert module.stops == 1
