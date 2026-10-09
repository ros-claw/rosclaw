"""Private offline pool allocation, crash retention and concurrency contracts."""

import dataclasses
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

import rosclaw.growth.offline_pool_allocation as allocation_module
from rosclaw.growth.offline_pool_allocation import (
    OfflinePoolBinding,
    allocate_once,
    inspect_allocation,
)


def digest(index):
    return "sha256:" + f"{index:064x}"


@pytest.fixture
def binding():
    return OfflinePoolBinding(digest(1), digest(2), digest(3), digest(4), (digest(5),), digest(6))


@pytest.fixture
def ledger(tmp_path):
    root = tmp_path / "ledger"
    root.mkdir(mode=0o700)
    return root


def test_round_trip_private_owned_no_authority(ledger, binding):
    record = allocate_once(ledger, binding)
    assert record == inspect_allocation(ledger, binding)
    assert record["partition_after_allocation"] == "CONSUMED_NO_REUSE_AS_FRESH"
    for key in ("fresh_execution_authorized", "promotion_authorized", "hardware_authorized"):
        assert record[key] is False
    record["binding"]["candidate_hashes"].clear()
    assert inspect_allocation(ledger, binding)["binding"]["candidate_hashes"] == [digest(5)]
    directory = ledger / binding.pool_hash[7:]
    assert directory.stat().st_mode & 0o777 == 0o700
    assert (directory / "allocation.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "field",
    [
        None,
        "protocol_hash",
        "parent_hash",
        "candidate_hashes",
        "execution_commitment_hash",
        "train_split_hash",
    ],
)
def test_no_retry_or_changed_identity_can_reopen_pool(ledger, binding, field):
    allocate_once(ledger, binding)
    changed = (
        binding
        if field is None
        else dataclasses.replace(
            binding, **{field: (digest(8),) if field == "candidate_hashes" else digest(8)}
        )
    )
    with pytest.raises(FileExistsError):
        allocate_once(ledger, changed)
    if field is not None:
        with pytest.raises(ValueError, match="identity"):
            inspect_allocation(ledger, changed)


def test_concurrent_callers_exactly_one_allocation(ledger, binding):
    def attempt(_index):
        try:
            allocate_once(ledger, binding)
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(32)))
    assert sum(results) == 1
    assert inspect_allocation(ledger, binding)["binding"] == binding.to_dict()


def test_fsync_failure_poisoned_pool_is_never_deleted_or_reopened(ledger, binding, monkeypatch):
    def failure(_fd):
        raise OSError("injected durable write failure")

    with monkeypatch.context() as context:
        context.setattr(os, "fsync", failure)
        with pytest.raises(OSError, match="durable write"):
            allocate_once(ledger, binding)
    assert (ledger / binding.pool_hash[7:]).is_dir()
    with pytest.raises(FileExistsError):
        allocate_once(ledger, binding)
    with pytest.raises(FileNotFoundError):
        inspect_allocation(ledger, binding)


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"{",
        b"{}",
        b'{"hardware_authorized":true}',
        b'{"record_hash":"fake","record_hash":"fake"}',
    ],
)
def test_partial_and_tampered_records_refused_without_repair(ledger, binding, content):
    allocate_once(ledger, binding)
    path = ledger / binding.pool_hash[7:] / "allocation.json"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="identity"):
        inspect_allocation(ledger, binding)
    assert path.read_bytes() == content
    with pytest.raises(FileExistsError):
        allocate_once(ledger, binding)


def test_symlink_pool_or_record_never_followed(ledger, binding, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o700)
    pool = ledger / binding.pool_hash[7:]
    pool.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        allocate_once(ledger, binding)
    with pytest.raises(OSError):
        inspect_allocation(ledger, binding)
    assert list(target.iterdir()) == []


def test_symlink_intermediate_directory_refused(ledger, binding, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(ledger.parent, target_is_directory=True)
    with pytest.raises(OSError):
        allocate_once(alias / ledger.name, binding)
    assert list(ledger.iterdir()) == []


def test_nonprivate_root_refused_before_allocation(ledger, binding):
    ledger.chmod(0o755)
    with pytest.raises(ValueError, match="mode0700"):
        allocate_once(ledger, binding)
    assert list(ledger.iterdir()) == []


def test_nonregular_fifo_record_refused_without_blocking(ledger, binding):
    directory = ledger / binding.pool_hash[7:]
    directory.mkdir(mode=0o700)
    os.mkfifo(directory / "allocation.json", 0o600)
    with pytest.raises(ValueError, match="ordinary"):
        inspect_allocation(ledger, binding)


def test_symlink_record_refused_without_reading_target(ledger, binding, tmp_path):
    directory = ledger / binding.pool_hash[7:]
    directory.mkdir(mode=0o700)
    target = tmp_path / "external.json"
    target.write_text("do not read or overwrite")
    (directory / "allocation.json").symlink_to(target)
    with pytest.raises(OSError):
        inspect_allocation(ledger, binding)
    assert target.read_text() == "do not read or overwrite"


@pytest.mark.parametrize("stop_at", [1, 2, 3, 4])
def test_real_child_exit_preserves_consumption_without_auto_retry(ledger, binding, stop_at):
    program = """
import json, os, sys
from pathlib import Path
from rosclaw.growth.offline_pool_allocation import OfflinePoolBinding, allocate_once
values = json.loads(sys.argv[2])
values['candidate_hashes'] = tuple(values['candidate_hashes'])
binding = OfflinePoolBinding(**values)
real_fsync = os.fsync
count = 0
def crash_after_fsync(fd):
    global count
    real_fsync(fd)
    count += 1
    if count == int(sys.argv[3]):
        os._exit(73)
os.fsync = crash_after_fsync
allocate_once(Path(sys.argv[1]), binding)
raise AssertionError('child must exit before successful allocation return')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(ledger), json.dumps(binding.to_dict()), str(stop_at)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 73, result.stderr
    assert (ledger / binding.pool_hash[7:]).is_dir()
    with pytest.raises(FileExistsError):
        allocate_once(ledger, binding)
    if stop_at == 1:
        with pytest.raises(FileNotFoundError):
            inspect_allocation(ledger, binding)
    else:
        assert inspect_allocation(ledger, binding)["fresh_execution_authorized"] is False


def test_same_binding_different_pools_are_independent(ledger, binding):
    allocate_once(ledger, binding)
    other = dataclasses.replace(binding, pool_hash=digest(9))
    allocate_once(ledger, other)
    assert len(list(ledger.iterdir())) == 2


def test_frozen_binding_tamper_revalidated_before_writing(ledger, binding):
    object.__setattr__(binding, "pool_hash", "../../outside")
    with pytest.raises(ValueError, match="content identities"):
        allocate_once(ledger, binding)
    assert list(ledger.iterdir()) == []


def test_owned_identity_cannot_redirect_path_after_validation(ledger, binding, monkeypatch):
    original = dataclasses.replace(binding)
    open_root = allocation_module._open_root

    def mutate_after_validation(root):
        object.__setattr__(binding, "pool_hash", digest(999))
        return open_root(root)

    monkeypatch.setattr(allocation_module, "_open_root", mutate_after_validation)
    record = allocate_once(ledger, binding)
    assert record["binding"] == original.to_dict()
    assert (ledger / original.pool_hash[7:]).is_dir()
    assert not (ledger / digest(999)[7:]).exists()
    assert inspect_allocation(ledger, original) == record


@pytest.mark.parametrize(
    "field,value",
    [
        ("pool_hash", "bad"),
        ("candidate_hashes", []),
        ("candidate_hashes", ()),
        ("candidate_hashes", (digest(5), digest(5))),
        ("candidate_hashes", (digest(4),)),
        ("candidate_hashes", tuple(digest(i) for i in range(20, 37))),
    ],
)
def test_invalid_declarations_refused(binding, field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(binding, **{field: value})


def test_json_record_no_private_cases_or_output_directory(ledger, binding):
    allocate_once(ledger, binding)
    value = json.loads((ledger / binding.pool_hash[7:] / "allocation.json").read_bytes())
    assert set(value["binding"]) == set(binding.to_dict())
    assert "cases" not in value and "output_root" not in value
