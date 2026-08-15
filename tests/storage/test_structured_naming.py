"""PR-DF-01 / ADR-0010: canonical storage naming + backward-compat aliases.

The canonical names (StructuredStore, StoreFactory, SeekDBRetrievalStore,
MemoryRetrievalProjection, ROSCLAW_STRUCTURED_SCHEMAS) are the real
definitions; every pre-DF-01 name must remain importable and identical
until the compat layer is retired.
"""

import warnings

import pytest


def test_structured_store_aliases_are_identical():
    from rosclaw.memory.seekdb_client import (
        ROSCLAW_STRUCTURED_SCHEMAS,
        SEEKDB_SCHEMAS,
        InMemoryKnowledgeStore,
        InMemoryStructuredStore,
        SeekDBClient,
        SeekDBMySQLClient,
        SeekDBSQLStore,
        SQLiteKnowledgeStore,
        SQLiteStructuredStore,
        StructuredStore,
    )

    assert SeekDBClient is StructuredStore
    assert InMemoryKnowledgeStore is InMemoryStructuredStore
    assert SQLiteKnowledgeStore is SQLiteStructuredStore
    assert SeekDBMySQLClient is SeekDBSQLStore
    assert SEEKDB_SCHEMAS is ROSCLAW_STRUCTURED_SCHEMAS


def test_factory_aliases_are_identical():
    from rosclaw.storage.factory import StorageFactory, StoreFactory

    assert StorageFactory is StoreFactory
    a = StoreFactory.create_structured_store(backend="memory")
    b = StorageFactory.create_knowledge_store(backend="memory")
    assert type(a) is type(b)


def test_retrieval_store_aliases_are_identical():
    from rosclaw.storage.seekdb_native import (
        SeekDBEmbeddedRetrievalStore,
        SeekDBEmbeddedStore,
        SeekDBNativeStore,
        SeekDBRetrievalStore,
        SeekDBServerRetrievalStore,
        SeekDBServerStore,
    )

    assert SeekDBNativeStore is SeekDBRetrievalStore
    assert SeekDBEmbeddedStore is SeekDBEmbeddedRetrievalStore
    assert SeekDBServerStore is SeekDBServerRetrievalStore


def test_projection_aliases_are_identical():
    from rosclaw.storage.seekdb_projection import (
        MemoryRetrievalProjection,
        MemoryRetrievalProjectionCommitter,
        SeekDBProjection,
        SeekDBProjectionCommitter,
    )

    assert SeekDBProjection is MemoryRetrievalProjection
    assert SeekDBProjectionCommitter is MemoryRetrievalProjectionCommitter


def test_phase0_deprecated_subclasses_still_warn():
    from rosclaw.memory.seekdb_client import InMemoryStructuredStore, SeekDBMemoryClient

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client = SeekDBMemoryClient()
    assert isinstance(client, InMemoryStructuredStore)
    assert any(item.category is DeprecationWarning for item in caught)


def test_package_level_exports_keep_old_names():
    import rosclaw.memory as memory_pkg
    import rosclaw.storage as storage_pkg

    for name in (
        "SeekDBClient",
        "InMemoryKnowledgeStore",
        "SQLiteKnowledgeStore",
        "SeekDBMySQLClient",
        "SEEKDB_SCHEMAS",
    ):
        assert hasattr(memory_pkg, name), name
    assert storage_pkg.StorageFactory is storage_pkg.StoreFactory


def test_new_canonical_names_exported():
    import rosclaw.memory as memory_pkg

    for name in (
        "StructuredStore",
        "InMemoryStructuredStore",
        "SQLiteStructuredStore",
        "SeekDBSQLStore",
        "ROSCLAW_STRUCTURED_SCHEMAS",
    ):
        assert hasattr(memory_pkg, name), name
    with pytest.raises(AttributeError):
        _ = memory_pkg.DoesNotExist  # sanity: hasattr isn't vacuous
