"""不可变 SimStore 测试（PR-MH1，ADR-0014，规格 §11/§39，红→绿）。

安全语义全部 fail closed：路径逃逸 / symlink 逃逸 / 外部 URI /
静默覆盖 / 超限对象 / 盘外篡改。
"""

from __future__ import annotations

import json

import pytest

from rosclaw.sim.refs import make_ref, parse_ref
from rosclaw.sim.store import PARTITIONS, SimStore


def _sha64(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


# --- 基本往返 ---------------------------------------------------------------


def test_put_get_roundtrip_dict(tmp_path) -> None:
    store = SimStore(tmp_path)
    ref = store.put("models", {"name": "ur5e", "nq": 6})
    assert ref.startswith("simmdl_")
    kind, hex16 = parse_ref(ref)
    assert kind == "simmdl" and len(hex16) == 16
    assert store.get(ref) == {"name": "ur5e", "nq": 6}
    assert store.exists(ref)
    assert store.verify_digest(ref)


def test_put_get_roundtrip_bytes(tmp_path) -> None:
    store = SimStore(tmp_path)
    blob = b"<mujoco/>"
    ref = store.put("models", blob)
    assert store.get(ref) == blob


def test_all_partitions_accept_objects(tmp_path) -> None:
    store = SimStore(tmp_path)
    for partition in PARTITIONS:
        ref = store.put(partition, {"p": partition})
        assert store.exists(ref), partition


def test_put_idempotent_same_content(tmp_path) -> None:
    store = SimStore(tmp_path)
    ref1 = store.put("models", {"a": 1})
    ref2 = store.put("models", {"a": 1})
    assert ref1 == ref2
    assert len(store.list_children("models")) == 1


def test_put_explicit_ref_consistent_ok(tmp_path) -> None:
    store = SimStore(tmp_path)
    payload = {"x": [1.0, 2.0]}
    from rosclaw.contracts.common import canonical_json

    digest = _sha64(canonical_json(payload).encode("utf-8"))
    ref = make_ref("simsta", digest)
    assert store.put("states", payload, ref=ref) == ref


def test_put_explicit_ref_mismatch_fails(tmp_path) -> None:
    store = SimStore(tmp_path)
    wrong = make_ref("simsta", _sha64(b"other"))
    with pytest.raises(ValueError, match="STORE_DIGEST_MISMATCH"):
        store.put("states", {"x": 1}, ref=wrong)


def test_put_prefix_partition_mismatch_fails(tmp_path) -> None:
    store = SimStore(tmp_path)
    payload = {"x": 1}
    from rosclaw.contracts.common import canonical_json

    ref = make_ref("simtrc", _sha64(canonical_json(payload).encode("utf-8")))
    with pytest.raises(ValueError, match="REF_INVALID"):
        store.put("models", payload, ref=ref)


# --- 不可变与完整性 ---------------------------------------------------------


def test_immutable_violation_never_overwrites(tmp_path) -> None:
    store = SimStore(tmp_path)
    payload = {"v": 1}
    ref = store.put("models", payload)
    target = store.resolve(ref)
    target.write_bytes(b'{"v": 2}')  # 盘外篡改
    with pytest.raises(ValueError, match="STORE_IMMUTABLE_VIOLATION"):
        store.put("models", payload)
    assert target.read_bytes() == b'{"v": 2}'  # 不被覆写


def test_get_tampered_content_fails(tmp_path) -> None:
    store = SimStore(tmp_path)
    ref = store.put("models", {"v": 1})
    store.resolve(ref).write_bytes(b'{"v": 999}')
    with pytest.raises(ValueError, match="STORE_DIGEST_MISMATCH"):
        store.get(ref)
    assert not store.verify_digest(ref)


def test_ref_not_found(tmp_path) -> None:
    store = SimStore(tmp_path)
    ref = make_ref("simmdl", _sha64(b"ghost"))
    with pytest.raises(ValueError, match="REF_NOT_FOUND"):
        store.get(ref)
    with pytest.raises(ValueError, match="REF_NOT_FOUND"):
        store.resolve(ref)
    assert not store.exists(ref)


# --- 路径与 symlink 安全 ----------------------------------------------------


def test_resolve_inside_root(tmp_path) -> None:
    store = SimStore(tmp_path)
    ref = store.put("models", {"a": 1})
    path = store.resolve(ref)
    assert path.is_absolute()
    assert str(path).startswith(str((tmp_path / "sim").resolve()))


def test_partition_symlink_escape_fails(tmp_path) -> None:
    store = SimStore(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    link = tmp_path / "sim" / "models"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="STORE_PATH_ESCAPE"):
        store.put("models", {"evil": True})


def test_symlinked_object_fails(tmp_path) -> None:
    store = SimStore(tmp_path)
    ref = make_ref("simmdl", _sha64(b"x"))
    outside = tmp_path.parent / f"{tmp_path.name}_evil.json"
    outside.write_bytes(b"{}")
    target_dir = tmp_path / "sim" / "models"
    target_dir.mkdir(parents=True)
    (target_dir / f"{ref}.json").symlink_to(outside)
    with pytest.raises(ValueError, match="STORE_PATH_ESCAPE"):
        store.get(ref)


def test_oversized_object_fails(tmp_path) -> None:
    store = SimStore(tmp_path, max_bytes=64)
    with pytest.raises(ValueError, match="STORE_OBJECT_TOO_LARGE"):
        store.put("traces", {"payload": "x" * 1024})


def test_invalid_partition_rejected(tmp_path) -> None:
    store = SimStore(tmp_path)
    with pytest.raises(ValueError, match="STORE_PARTITION_UNKNOWN"):
        store.put("../etc", {"x": 1})
    with pytest.raises(ValueError, match="STORE_PARTITION_UNKNOWN"):
        store.put("nope", {"x": 1})


def test_uri_ref_rejected(tmp_path) -> None:
    store = SimStore(tmp_path)
    for bad in ("file:///etc/passwd", "https://evil/x", "../../../etc/passwd"):
        with pytest.raises(ValueError, match="REF_INVALID"):
            store.get(bad)


# --- 列举与 legacy 桥 -------------------------------------------------------


def test_list_children_only_own_partition(tmp_path) -> None:
    store = SimStore(tmp_path)
    a = store.put("models", {"a": 1})
    store.put("states", {"s": 1})
    assert store.list_children("models") == [a]
    assert len(store.list_children("states")) == 1
    assert store.list_children("traces") == []


def test_legacy_ref_readonly_bridge(tmp_path) -> None:
    # legacy api.py 布局：task_root/models/model_<16hex>.json
    legacy_dir = tmp_path / "models"
    legacy_dir.mkdir()
    legacy_ref = "model_" + "0" * 16
    (legacy_dir / f"{legacy_ref}.json").write_text(json.dumps({"legacy": True}), encoding="utf-8")
    store = SimStore(tmp_path)
    assert store.get(legacy_ref) == {"legacy": True}
    assert store.resolve(legacy_ref) == legacy_dir / f"{legacy_ref}.json"
    with pytest.raises(ValueError, match="STORE_LEGACY_READONLY"):
        store.put("models", {"x": 1}, ref=legacy_ref)


def test_legacy_ref_not_found(tmp_path) -> None:
    store = SimStore(tmp_path)
    with pytest.raises(ValueError, match="REF_NOT_FOUND"):
        store.get("model_" + "9" * 16)
