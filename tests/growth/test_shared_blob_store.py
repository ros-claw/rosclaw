import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from rosclaw.growth.shared_blob_store import publish_readonly_blob


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def setup(tmp_path):
    source, store, targets = tmp_path / "source", tmp_path / "store", tmp_path / "targets"
    source.write_bytes(bytes(range(256)) * 100)
    store.mkdir()
    targets.mkdir()
    return source, store, targets


def publish(source, store, target, **extra):
    return publish_readonly_blob(
        source,
        store=store,
        target=target,
        expected_hash=extra.get("expected_hash", digest(source.read_bytes())),
        maximum_blob_bytes=extra.get("maximum_blob_bytes", source.stat().st_size),
    )


def test_complete_bytes_shared_readonly_source_unchanged(tmp_path):
    source, store, targets = setup(tmp_path)
    before, mode = source.read_bytes(), source.stat().st_mode
    first = publish(source, store, targets / "first")
    second = publish(source, store, targets / "second")
    assert len(list(store.glob("*.blob"))) == 1
    assert first.blob_hash == second.blob_hash == digest(before)
    assert first.size_bytes == len(before)
    assert first.target_path.read_bytes() == second.target_path.read_bytes() == before
    assert first.target_path.stat().st_ino == second.target_path.stat().st_ino
    assert first.target_path.stat().st_ino != source.stat().st_ino
    assert first.target_path.stat().st_mode & 0o222 == 0
    assert source.read_bytes() == before and source.stat().st_mode == mode
    assert not list(store.glob(".owned-blob-*"))


def test_parallel_publication_retains_one_blob_and_independent_target_names(tmp_path):
    source, store, targets = setup(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda i: publish(source, store, targets / str(i)), range(8)))
    assert len(list(store.iterdir())) == 1
    assert len({r.target_path.stat().st_ino for r in records}) == 1
    assert all(r.target_path.read_bytes() == source.read_bytes() for r in records)


def test_no_existing_target_overwrite_even_when_identical(tmp_path):
    source, store, targets = setup(tmp_path)
    target = targets / "existing"
    target.write_bytes(source.read_bytes())
    before = target.read_bytes()
    with pytest.raises(ValueError, match="never overwritten"):
        publish(source, store, target)
    assert target.read_bytes() == before
    assert list(store.iterdir()) == []


@pytest.mark.parametrize("kind", ["writable", "changed", "symlink"])
def test_existing_blob_corruption_rejected_without_repair(tmp_path, kind):
    source, store, targets = setup(tmp_path)
    artifact = publish(source, store, targets / "first")
    blob = artifact.blob_path
    if kind == "symlink":
        # Replace only this test's private blob, not any project artifact.
        blob.unlink()
        blob.symlink_to(source)
    else:
        blob.chmod(0o644)
        if kind == "changed":
            blob.write_bytes(b"corrupt")
            blob.chmod(0o444)
    with pytest.raises(ValueError):
        publish(source, store, targets / "second")
    assert not (targets / "second").exists()
    assert not list(store.glob(".owned-blob-*"))


@pytest.mark.parametrize("bound", [0, -1, True, 16 * 1024**3 + 1, 1])
def test_invalid_or_insufficient_size_bound_rejected(tmp_path, bound):
    source, store, targets = setup(tmp_path)
    with pytest.raises(ValueError):
        publish(source, store, targets / "bad", maximum_blob_bytes=bound)
    assert list(store.iterdir()) == []


@pytest.mark.parametrize("hash_", ["../../escape", "sha256:" + "0" * 64, False])
def test_digest_tamper_rejected_without_publication(tmp_path, hash_):
    source, store, targets = setup(tmp_path)
    with pytest.raises(ValueError):
        publish(source, store, targets / "bad", expected_hash=hash_)
    assert list(store.iterdir()) == []


def test_symlinked_directory_and_source_rejected(tmp_path):
    source, store, targets = setup(tmp_path)
    indirect = tmp_path / "indirect"
    indirect.symlink_to(source)
    with pytest.raises(ValueError, match="regular"):
        publish(indirect, store, targets / "bad")
    link = tmp_path / "linked-store"
    link.symlink_to(store, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink"):
        publish(source, link, targets / "bad")
    assert list(store.iterdir()) == []
