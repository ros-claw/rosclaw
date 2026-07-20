"""Fixture-replay transport tests (review §1 golden fixtures)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.rh56.fixture_transport import FixtureModbusTransport
from rosclaw.body.rh56.transport import CommandDelivery, TransportIOError
from rosclaw.body.rh56.transport_profile import load_transport_profile

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "rh56_modbus"
CONFIGS = REPO_ROOT / "configs"


def _transport(fixture: str) -> FixtureModbusTransport:
    profile = load_transport_profile(CONFIGS / "rh56_right_rs485_v1.yaml")
    transport = FixtureModbusTransport(profile, FIXTURES / fixture)
    transport.connect()
    return transport


def test_replay_sample_session() -> None:
    transport = _transport("sample_session.jsonl")
    fb1 = transport.read_state()
    assert fb1.position == [1000] * 6
    transport.read_state()
    delivery = transport.write_position(
        [1000, 1000, 1000, 980, 1000, 1000], speed=100, force_limit=100
    )
    assert delivery == CommandDelivery.ACKNOWLEDGED
    fb3 = transport.read_state()
    assert fb3.position[3] == 990
    assert fb3.current_ma[3] == 145.0
    fb4 = transport.read_state()
    assert fb4.position[3] == 980
    # CURRENT decays to ~0 in static state (real RH56 behaviour).
    assert fb4.current_ma[3] == 0.0


def test_replay_uncertain_write() -> None:
    transport = _transport("uncertain_write.jsonl")
    transport.read_state()
    delivery = transport.write_position(
        [1000, 1000, 1000, 980, 1000, 1000], speed=100, force_limit=100
    )
    assert delivery == CommandDelivery.UNCERTAIN


def test_replay_io_error() -> None:
    transport = _transport("io_error_read.jsonl")
    with pytest.raises(TransportIOError, match="io_error"):
        transport.read_state()
    fb = transport.read_state()
    assert fb.position == [1000] * 6


def test_fixture_exhaustion_and_mismatch() -> None:
    transport = _transport("sample_session.jsonl")
    with pytest.raises(TransportIOError, match="fixture_mismatch"):
        # First frame is a read; a write here must fail loudly.
        transport.write_position([1000] * 6, speed=100, force_limit=100)
    transport2 = _transport("uncertain_write.jsonl")
    transport2.read_state()
    transport2.write_position([1000, 1000, 1000, 980, 1000, 1000], speed=100, force_limit=100)
    transport2.read_state()
    with pytest.raises(TransportIOError, match="fixture_exhausted"):
        transport2.read_state()


def test_fixture_missing() -> None:
    with pytest.raises(TransportIOError, match="fixture_missing"):
        _transport("does_not_exist.jsonl")
