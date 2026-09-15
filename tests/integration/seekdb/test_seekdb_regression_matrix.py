"""PR-SDB-140-1 (outline §十): the formal SeekDB regression matrix.

Promotes `validation/ty1200/scripts/repro_pyseekdb_140.py` (kept as history)
into a pytest suite that runs against the REAL embedded engine whenever
pyseekdb + pylibseekdb are installed (the data-flywheel gate profile
installs them), and skips honestly otherwise.

T0  basic CRUD                    T8  delete
T1  BM25 + where_document + where T9  refresh index (strict — §九/十三)
T2  dual-leg RRF, both filtered   T10 restart persistence
T3  query() + simple where        T11 wrong vector dimension
T4  KNN + metadata where          T12 collection lifecycle
T5  $and compound metadata filter T13 versioned collections
T6  batch upsert                  T14 projection (seekdb_projection)
T7  update (upsert same id)       T15 idempotent ingestion

The known-bad SDK (pyseekdb 1.4.0) fails T1/T2/T4 — that is the point of
running this on every upgrade.  All tests in this module share ONE embedded
target (pylibseekdb allows one path/database per process).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Real-engine integration suite: excluded from the default regression
# selector (addopts: -m 'not integration and not deployment'); runs in its
# own process via data-flywheel Gate G.  The embedded engine hosts ONE live
# target per process — sharing a process with tests/memory/v2/test_cli.py
# kills both.
pytestmark = pytest.mark.integration

pyseekdb = pytest.importorskip("pyseekdb", reason="regression matrix needs the real SDK")
pytest.importorskip("pylibseekdb", reason="regression matrix needs the embedded engine")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rosclaw.storage.seekdb_native import SeekDBEmbeddedRetrievalStore  # noqa: E402

DOCS = [
    "机械臂关节目标越界导致 sandbox 阻断",
    "gripper contact force exceeded safe threshold",
    "LIMO localization drift after lidar packet loss",
]
META = [
    {"robot": "ur5e", "outcome": "failure"},
    {"robot": "rh56", "outcome": "failure"},
    {"robot": "limo", "outcome": "failure"},
]
VEC = [
    [0.1, 0.2, 0.3, 0.4],
    [0.4, 0.3, 0.2, 0.1],
    [0.2, 0.4, 0.1, 0.3],
]


@pytest.fixture(scope="module")
def matrix_path(tmp_path_factory):
    """THE one embedded target for this process (pylibseekdb single-target)."""
    path = tmp_path_factory.mktemp("seekdb_matrix")
    admin = pyseekdb.AdminClient(path=str(path))
    admin.create_database("matrix")
    admin.close() if hasattr(admin, "close") else None
    return str(path)


@pytest.fixture(scope="module")
def coll(matrix_path):
    """One raw pyseekdb collection on the module's single embedded target."""
    client = pyseekdb.Client(path=matrix_path, database="matrix")
    collection = client.get_or_create_collection(
        "docs",
        configuration=pyseekdb.HNSWConfiguration(dimension=4),
        embedding_function=None,
    )
    collection.upsert(ids=["a", "b", "c"], embeddings=VEC, documents=DOCS, metadatas=META)
    yield collection


# --- T0-T9: raw SDK paths (the #251 surface) -------------------------------


def test_t0_basic_get(coll):
    res = coll.get(ids=["a"], include=["metadatas"])
    assert res["ids"] == ["a"]
    assert res["metadatas"][0]["robot"] == "ur5e"


def test_t1_bm25_where_document_and_metadata_where(coll):
    """#251 breaker: BM25 leg with both filters."""
    res = coll.hybrid_search(
        query={"where_document": {"$contains": "sandbox"}, "n_results": 3, "where": {"robot": "ur5e"}},
        n_results=3,
        include=["metadatas"],
    )
    assert res["ids"] == [["a"]]


def test_t2_dual_leg_rrf_both_filtered(coll):
    """#251 breaker: malformed FULL JOIN on 1.4.0."""
    res = coll.hybrid_search(
        query={"where_document": {"$contains": "sandbox"}, "n_results": 3, "where": {"robot": "ur5e"}},
        knn={"query_embeddings": [[0.1, 0.2, 0.3, 0.4]], "n_results": 3, "where": {"robot": "ur5e"}},
        rank={"rrf": {"rank_window_size": 5, "rank_constant": 60}},
        n_results=3,
        include=["metadatas"],
    )
    assert res["ids"] == [["a"]]


def test_t3_query_simple_where(coll):
    res = coll.query(
        query_embeddings=[[0.1, 0.2, 0.3, 0.4]],
        where={"robot": "ur5e"},
        n_results=3,
        include=["metadatas"],
    )
    assert res["ids"] == [["a"]]


def test_t4_knn_leg_metadata_where(coll):
    """#251 breaker: identifier too long on 1.4.0."""
    res = coll.hybrid_search(
        knn={"query_embeddings": [[0.1, 0.2, 0.3, 0.4]], "n_results": 3, "where": {"robot": "ur5e"}},
        n_results=3,
        include=["metadatas"],
    )
    assert res["ids"] == [["a"]]


def test_t5_and_compound_metadata(coll):
    res = coll.query(
        query_embeddings=[[0.1, 0.2, 0.3, 0.4]],
        where={"$and": [{"robot": "limo"}, {"outcome": "failure"}]},
        n_results=3,
        include=["metadatas"],
    )
    assert res["ids"] == [["c"]]


def test_t6_batch_upsert(coll):
    ids = [f"batch_{i}" for i in range(32)]
    coll.upsert(
        ids=ids,
        embeddings=[[0.1, 0.2, 0.3, 0.4]] * 32,
        documents=[f"batch doc {i}" for i in range(32)],
        metadatas=[{"robot": "batch", "outcome": "success"}] * 32,
    )
    res = coll.get(ids=ids[:4], include=["metadatas"])
    assert len(res["ids"]) == 4


def test_t7_update_via_upsert(coll):
    coll.upsert(
        ids=["a"],
        embeddings=[VEC[0]],
        documents=["updated doc"],
        metadatas=[{"robot": "ur5e", "outcome": "fixed"}],
    )
    res = coll.get(ids=["a"], include=["metadatas"])
    assert res["metadatas"][0]["outcome"] == "fixed"
    # restore for other tests
    coll.upsert(ids=["a"], documents=[DOCS[0]], embeddings=[VEC[0]], metadatas=[META[0]])


def test_t8_delete(coll):
    coll.upsert(ids=["gone"], embeddings=[VEC[0]], documents=["to delete"], metadatas=[{"robot": "x"}])
    coll.delete(ids=["gone"])
    res = coll.get(ids=["gone"])
    assert res["ids"] == []


def test_t9_refresh_index(coll):
    """Explicit refresh must work — and (PR-SDB-140-1) failures must raise."""
    coll.refresh_index()


def test_t11_wrong_vector_dimension(coll):
    # engine raises builtin RuntimeError for dimension mismatch (probed)
    with pytest.raises(RuntimeError):
        coll.upsert(ids=["bad"], embeddings=[[0.1, 0.2]], documents=["bad dim"])


def test_t12_collection_lifecycle(coll, tmp_path_factory):
    client = coll._client
    tmp = client.create_collection(
        "ephemeral",
        configuration=pyseekdb.HNSWConfiguration(dimension=4),
        embedding_function=None,
    )
    tmp.upsert(ids=["e"], embeddings=[[0.1, 0.2, 0.3, 0.4]], documents=["e"])
    assert tmp.get(ids=["e"])["ids"] == ["e"]
    client.delete_collection("ephemeral")
    # engine raises builtin ValueError for a missing collection (probed)
    with pytest.raises(ValueError, match="does not exist"):
        client.get_collection("ephemeral")


# --- T10/T13-T15: rosclaw-level paths on the same engine --------------------


@pytest.fixture(scope="module")
def store(matrix_path):
    """The rosclaw store on the SAME single embedded target (same database)."""
    s = SeekDBEmbeddedRetrievalStore(path=matrix_path, database="matrix")
    s.connect()
    yield s
    s.disconnect()


def test_t10_restart_persistence(store, tmp_path):
    store.insert(
        "memory_items",
        {"id": "persist_1", "title": "restart me", "document": "restart persistence check"},
    )
    path = store._path
    store.disconnect()
    again = SeekDBEmbeddedRetrievalStore(path=path, database="matrix")
    again.connect()
    try:
        rows = again.query("memory_items", {"id": "persist_1"})
        assert len(rows) == 1
    finally:
        again.disconnect()
        store.connect()  # re-establish the module fixture for T13-T15


def test_t13_versioned_collections(store):
    """Versioned build path on the real engine, with a deterministic
    hash-vector provider (the provider protocol's designed fake point)."""
    import hashlib

    from rosclaw.embedding.protocol import EmbeddingProfile
    from rosclaw.storage.versioned_collections import VersionedCollectionManager

    class _HashProvider:
        profile = EmbeddingProfile(
            profile_id="hash4_v1",
            model_id="hash4",
            model_revision="1",
            dimension=4,
            normalize=False,
            distance="cosine",
            query_instruction=None,
            document_instruction=None,
            max_tokens=32,
            provider_type="hash",
        )

        def encode_documents(self, texts):
            return [
                [int(b) / 255 for b in hashlib.sha256(t.encode()).digest()[:4]] for t in texts
            ]

        def encode_queries(self, texts):
            return self.encode_documents(texts)

        def health(self):
            return {"ok": True}

    mgr = VersionedCollectionManager(store, provider=_HashProvider())
    result = mgr.build(
        "matrix_versioned",
        [{"id": "v1", "title": "versioned one", "document": "versioned collection one"}],
    )
    assert result["status"] == "READY"
    mgr.activate("matrix_versioned", analyzer="ngram")
    active = mgr.active("matrix_versioned")
    assert active and active["record_count"] == 1


def test_t14_projection(store):
    from rosclaw.storage.seekdb_projection import MemoryRetrievalProjection

    store.insert(
        "memory_items",
        {"id": "proj_1", "title": "projection", "document": "projection target"},
    )
    store.refresh_index("memory_items")  # strict by default — must not raise
    proj = MemoryRetrievalProjection(store)
    status = proj.status()
    assert status is not None and "projection_count" in status


def test_t15_idempotent_ingestion(store):
    record = {"id": "idem_1", "title": "idempotent", "document": "write me twice"}
    store.insert("memory_items", record)
    store.insert("memory_items", record)
    rows = store.query("memory_items", {"id": "idem_1"})
    assert len(rows) == 1
