from __future__ import annotations

from typing import Any

from rosclaw.storage.seekdb_native import SeekDBNativeStore


def _result(rows: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "ids": [[record_id for record_id, _ in rows]],
        "metadatas": [[{"id": record_id, "body_id": body_id} for record_id, body_id in rows]],
    }


class _EmbeddedCollection:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def hybrid_search(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if kwargs.get("query") is not None:
            return _result([("bm25", "left"), ("shared", "left")])
        return _result([("shared", "left"), ("knn", "left")])


def test_embedded_hybrid_filters_both_legs_before_deterministic_rrf() -> None:
    collection = _EmbeddedCollection()
    store = SeekDBNativeStore(path="/tmp/not-opened")
    store._collections["memory_items"] = collection

    rows = store.hybrid_search(
        "memory_items",
        "joint_not_reached",
        filters={"body_id": "left"},
        limit=3,
        candidate_window=5,
        query_embedding=[0.1, 0.2],
    )

    assert [row["id"] for row in rows] == ["shared", "bm25", "knn"]
    assert len(collection.calls) == 2
    assert collection.calls[0]["query"]["where"] == {"body_id": "left"}
    assert collection.calls[1]["knn"]["where"] == {"body_id": "left"}
    assert collection.calls[1]["knn"]["query_embeddings"] == [[0.1, 0.2]]


def test_fulltext_degradation_keeps_engine_side_filter() -> None:
    collection = _EmbeddedCollection()
    store = SeekDBNativeStore(path="/tmp/not-opened")
    store._collections["memory_items"] = collection

    rows = store.fulltext_search(
        "memory_items",
        "joint_not_reached",
        filters={"robot_id": "robot-a"},
        limit=2,
    )

    assert rows
    assert collection.calls[0]["query"]["where"] == {"robot_id": "robot-a"}
