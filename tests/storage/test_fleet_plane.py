"""Real OceanBase fleet plane tests (PR-OB-1 §8.5, §8.6) — no mocks.

Requires the quay.io/oceanbase/seekdb container on :2881 (skip otherwise).
Three logical robots (rh56_left, rh56_right, mobile_base_01), two tenants.
"""
# ruff: noqa: E402 - imports follow pytest.importorskip by design

from __future__ import annotations

import time

import pytest

pymysql = pytest.importorskip("pymysql", reason="pymysql not installed")

from rosclaw.storage.fleet import FleetIngestCommitter, FleetPlane
from rosclaw.storage.outbox import OutboxStore, OutboxWorker

HOST, PORT = "127.0.0.1", 2881
DB = "rosclaw_fleet_test"


def _server_reachable() -> bool:
    import socket

    try:
        with socket.create_connection((HOST, PORT), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _server_reachable(), reason="fleet container not on :2881")

TENANT_A, TENANT_B = "lab_alpha", "lab_beta"


@pytest.fixture(scope="module")
def planes():
    admin = pymysql.connect(host=HOST, port=PORT, user="root", password="")
    with admin.cursor() as cursor:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB}")
    admin.close()
    conn = pymysql.connect(host=HOST, port=PORT, user="root", password="", database=DB)
    plane_a = FleetPlane(conn, tenant_id=TENANT_A)
    plane_b = FleetPlane(conn, tenant_id=TENANT_B)
    plane_a.ensure_schema()
    # Tests assert on exact tenant contents — start from a clean slate.
    with conn.cursor() as cursor:
        for table in (
            "fleet_robots",
            "fleet_episodes",
            "fleet_memories",
            "fleet_failures",
            "fleet_interventions",
            "fleet_skill_evidence",
            "fleet_sync_watermarks",
            "fleet_memory_search",
        ):
            cursor.execute(f"DELETE FROM {table}")
    conn.commit()
    yield plane_a, plane_b, conn
    conn.close()


def _mem(robot: str, body: str, mtype: str, title: str, **kw) -> dict:
    return {
        "memory_id": kw.get("memory_id", f"{robot}_{mtype}_{int(time.time() * 1000) % 100000}"),
        "robot_id": robot,
        "body_id": body,
        "memory_type": mtype,
        "title": title,
        "document": kw.get("document", title),
        "outcome": kw.get("outcome"),
        "confidence": kw.get("confidence", 0.85),
        "importance": kw.get("importance", 0.6),
        "event_time": time.time(),
        **{
            k: v
            for k, v in kw.items()
            if k not in {"memory_id", "document", "outcome", "confidence", "importance"}
        },
    }


def test_fleet_schema_created(planes) -> None:
    plane_a, _, conn = planes
    with conn.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        tables = {row[0] for row in cursor.fetchall()}
    expected = set(__import__("rosclaw.storage.fleet", fromlist=["FLEET_DDL"]).FLEET_DDL) | {
        "fleet_memory_search"
    }
    assert expected <= tables, f"missing: {expected - tables}"
    # tenant/robot/body columns present on fleet_memories
    with conn.cursor() as cursor:
        cursor.execute("SHOW COLUMNS FROM fleet_memories")
        cols = {row[0] for row in cursor.fetchall()}
    for required in ("tenant_id", "project_id", "site_id", "robot_id", "body_id"):
        assert required in cols


def test_three_robots_sync_and_isolation(planes) -> None:
    plane_a, plane_b, _ = planes
    for robot in ("rh56_left", "rh56_right", "mobile_base_01"):
        plane_a.upsert_robot(
            {"robot_id": robot, "robot_type": "dual_rh56" if "rh56" in robot else "base"}
        )
        plane_a.upsert_memory(
            _mem(robot, f"{robot}_body", "failure", f"{robot} scissors overcurrent")
        )
    # Same tenant: cross-robot query sees all three.
    rows = plane_a.query_memories(memory_type="failure")
    assert {row["robot_id"] for row in rows} == {"rh56_left", "rh56_right", "mobile_base_01"}
    # Different tenant: sees nothing.
    assert plane_b.query_memories(memory_type="failure") == []
    # Tenant B's own data doesn't leak back.
    plane_b.upsert_memory(_mem("rh56_left", "rh56_left_body", "failure", "beta-only failure"))
    assert len(plane_b.query_memories(memory_type="failure")) == 1
    beta_titles = {row["title"] for row in plane_b.query_memories(memory_type="failure")}
    assert "beta-only failure" in beta_titles
    alpha_titles = {row["title"] for row in plane_a.query_memories(memory_type="failure")}
    assert "beta-only failure" not in alpha_titles


def test_idempotent_sync_no_duplicates(planes) -> None:
    plane_a, _, _ = planes
    record = _mem(
        "rh56_left", "rh56_left_body", "intervention", "restore middle to base", memory_id="dedup_1"
    )
    plane_a.upsert_memory(record)
    plane_a.upsert_memory(record)  # redelivery from outbox must not duplicate
    rows = [
        row for row in plane_a.query_memories(robot_id="rh56_left") if row["memory_id"] == "dedup_1"
    ]
    assert len(rows) == 1


def test_cross_robot_skill_pattern(planes) -> None:
    plane_a, _, _ = planes
    plane_a.upsert_skill_evidence(
        {
            "evidence_id": "sk1",
            "robot_id": "rh56_left",
            "body_id": "rh56_left_body",
            "skill_id": "scissors",
            "success_count": 96,
            "failure_count": 48,
            "sample_count": 144,
        }
    )
    plane_a.upsert_skill_evidence(
        {
            "evidence_id": "sk2",
            "robot_id": "rh56_right",
            "body_id": "rh56_right_body",
            "skill_id": "scissors",
            "success_count": 140,
            "failure_count": 4,
            "sample_count": 144,
        }
    )
    pattern = plane_a.skill_success_pattern("scissors")
    by_robot = {row["robot_id"]: row for row in pattern}
    assert by_robot["rh56_right"]["success_count"] == 140
    assert by_robot["rh56_left"]["failure_count"] == 48
    # Body-specific: left hand's worse scissors pattern doesn't overwrite right's.
    assert by_robot["rh56_right"]["body_id"] == "rh56_right_body"


def test_sync_watermark_progress(planes) -> None:
    plane_a, _, _ = planes
    plane_a.bump_watermark("rh56_left", "memory", "mem_100", 10)
    plane_a.bump_watermark("rh56_left", "memory", "mem_120", 5)
    wm = plane_a.watermark("rh56_left", "memory")
    assert wm["last_entity_id"] == "mem_120"
    assert wm["records_synced"] >= 15


def test_outbox_to_fleet_delivery(planes, tmp_path) -> None:
    plane_a, _, _ = planes
    outbox = OutboxStore(db_path=str(tmp_path / "outbox.sqlite"))
    outbox.connect()
    committer = FleetIngestCommitter(plane_a, "mobile_base_01")
    worker = OutboxWorker(outbox, committer, interval_sec=0.05)
    outbox.enqueue(
        "fleet",
        _mem("mobile_base_01", "base", "failure", "base motor overcurrent", memory_id="fleet_ob_1"),
    )
    worker.flush(timeout=10.0)
    worker.stop()
    rows = [
        row
        for row in plane_a.query_memories(robot_id="mobile_base_01")
        if row["memory_id"] == "fleet_ob_1"
    ]
    assert len(rows) == 1
    assert outbox.stats()["total"] == 0  # v1 outbox deletes on delivery


def test_native_vector_and_fulltext_and_hybrid(planes) -> None:
    """§8.3/§8.6: native VECTOR column + FULLTEXT index + hybrid query."""
    _, _, conn = planes
    with conn.cursor() as cursor:
        # Insert with a literal 384-dim vector to prove VECTOR column works.
        vec = "[" + ",".join("0.01" for _ in range(384)) + "]"
        cursor.execute("DELETE FROM fleet_memory_search WHERE memory_id = 'vec_probe'")
        cursor.execute(
            "INSERT INTO fleet_memory_search "
            "(memory_id, tenant_id, robot_id, body_id, memory_type, document, metadata, embedding, event_time) "
            "VALUES ('vec_probe', %s, 'rh56_left', 'rh56_left_body', 'failure', "
            "'剪刀手势失败 食指过流 overcurrent', JSON_OBJECT('probe', true), %s, %s)",
            (TENANT_A, vec, time.time()),
        )
        # Native fulltext MATCH
        cursor.execute(
            "SELECT memory_id FROM fleet_memory_search "
            "WHERE tenant_id = %s AND MATCH(document) AGAINST ('过流' IN NATURAL LANGUAGE MODE)",
            (TENANT_A,),
        )
        ft_hits = [row[0] for row in cursor.fetchall()]
        assert "vec_probe" in ft_hits, "native FULLTEXT MATCH failed"
        # Native vector distance query (l2 distance against the same vector)
        cursor.execute(
            "SELECT memory_id, l2_distance(embedding, %s) AS dist FROM fleet_memory_search "
            "WHERE tenant_id = %s ORDER BY dist ASC LIMIT 3",
            (vec, TENANT_A),
        )
        vec_hits = cursor.fetchall()
        assert vec_hits and vec_hits[0][0] == "vec_probe", "native vector search failed"
        # Hybrid: fulltext filter + vector order (HYBRID_SEARCH-style composition)
        cursor.execute(
            "SELECT memory_id FROM fleet_memory_search "
            "WHERE tenant_id = %s AND MATCH(document) AGAINST ('剪刀' IN NATURAL LANGUAGE MODE) "
            "ORDER BY l2_distance(embedding, %s) ASC LIMIT 3",
            (TENANT_A, vec),
        )
        hybrid_hits = [row[0] for row in cursor.fetchall()]
        assert "vec_probe" in hybrid_hits, "hybrid fulltext+vector composition failed"
    conn.commit()


def test_center_down_does_not_affect_local(tmp_path) -> None:
    """Robot local outbox keeps accepting events when the center is unreachable."""
    outbox = OutboxStore(db_path=str(tmp_path / "outbox.sqlite"))
    outbox.connect()
    # Enqueue while "center down" — must not raise.
    record_id = outbox.enqueue(
        "fleet",
        {
            "id": "offline_1",
            "robot_id": "rh56_left",
            "memory_type": "failure",
            "title": "offline event",
        },
    )
    assert record_id
    assert outbox.stats()["total"] == 1
    outbox.close()
