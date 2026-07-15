"""Tests for the MySQL-compatible connection pool in SeekDBMySQLClient."""

from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock

import pytest

from rosclaw.memory.seekdb_client import _ConnectionPool, _PooledConnection


class FakeConnection:
    """Fake PyMySQL connection that tracks close calls."""

    def __init__(self, name: str = "conn") -> None:
        self.name = name
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_pool_creates_connection_on_first_acquire():
    created = []

    def creator():
        conn = FakeConnection(f"c{len(created)}")
        created.append(conn)
        return conn

    pool = _ConnectionPool(creator=creator, pool_size=2)
    with pool.acquire() as conn:
        assert conn is created[0]
        assert not conn.closed
    assert len(created) == 1


def test_pool_reuses_released_connection():
    created = []

    def creator():
        conn = FakeConnection(f"c{len(created)}")
        created.append(conn)
        return conn

    pool = _ConnectionPool(creator=creator, pool_size=2)
    with pool.acquire() as conn1:
        pass
    with pool.acquire() as conn2:
        assert conn2 is conn1
    assert len(created) == 1


def test_pool_respects_max_size():
    """The pool should never create more than ``pool_size`` connections."""
    created_count = 0
    gate = threading.Event()

    def creator():
        nonlocal created_count
        created_count += 1
        return FakeConnection(f"c{created_count}")

    pool = _ConnectionPool(creator=creator, pool_size=2)
    # Hold two connections; a third acquire should block until one is released.
    acquired = queue.Queue()

    def hold_connection():
        with pool.acquire() as conn:
            acquired.put(conn)
            gate.wait()

    t1 = threading.Thread(target=hold_connection)
    t2 = threading.Thread(target=hold_connection)
    t1.start()
    t2.start()
    # Wait until both connections are acquired.
    c1 = acquired.get(timeout=1.0)
    c2 = acquired.get(timeout=1.0)
    assert created_count == 2

    blocked_acquired = threading.Event()
    blocked_conn = None

    def blocked_acquire():
        nonlocal blocked_conn
        with pool.acquire() as conn:
            blocked_conn = conn
            blocked_acquired.set()

    t3 = threading.Thread(target=blocked_acquire)
    t3.start()
    # The third thread should not acquire immediately.
    assert not blocked_acquired.wait(0.1)
    assert created_count == 2

    # Release one connection; the blocked thread should acquire it.
    gate.set()
    t1.join(timeout=1.0)
    t2.join(timeout=1.0)
    assert blocked_acquired.wait(1.0)
    assert blocked_conn is c1 or blocked_conn is c2
    t3.join(timeout=1.0)


def test_broken_connection_is_not_returned_to_pool():
    created = []

    def creator():
        conn = FakeConnection(f"c{len(created)}")
        created.append(conn)
        return conn

    pool = _ConnectionPool(creator=creator, pool_size=2)
    with pytest.raises(RuntimeError), pool.acquire():
        assert len(created) == 1
        raise RuntimeError("boom")

    # A subsequent acquire should create a new connection.
    with pool.acquire() as conn2:
        assert conn2 is not created[0]
        assert len(created) == 2


def test_failed_creator_does_not_leak_pool_capacity():
    attempts = 0

    def creator():
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise ConnectionError("server down")
        return FakeConnection("ok")

    pool = _ConnectionPool(creator=creator, pool_size=2, timeout=0.1)
    # First two acquires fail.
    with pytest.raises(ConnectionError), pool.acquire():
        pass  # pragma: no cover
    with pytest.raises(ConnectionError), pool.acquire():
        pass  # pragma: no cover
    # Third acquire should still be allowed to create a connection.
    with pool.acquire() as conn:
        assert conn.name == "ok"
    assert attempts == 3


def test_pool_close_drains_and_closes_connections():
    created = []

    def creator():
        conn = FakeConnection(f"c{len(created)}")
        created.append(conn)
        return conn

    pool = _ConnectionPool(creator=creator, pool_size=2)
    with pool.acquire():
        pass
    pool.close()
    assert all(conn.closed for conn in created)

    # Acquire after close should raise.
    with pytest.raises(RuntimeError, match="closed"), pool.acquire():
        pass  # pragma: no cover


def test_pooled_connection_tracks_broken_state():
    conn = FakeConnection()
    pool = MagicMock()
    pooled = _PooledConnection(pool, conn)
    try:
        with pooled:
            raise ValueError("broken")
    except ValueError:
        pass
    assert pooled._broken is True
    pool._release.assert_called_once_with(conn, True)
