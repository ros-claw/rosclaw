"""仿真对象引用测试（PR-MH1，ADR-0014，红→绿）。

ref 是内容寻址令牌，不是路径：路径逃逸在词法层 fail closed。
"""

from __future__ import annotations

import pytest

from rosclaw.sim.refs import (
    classify,
    is_legacy_ref,
    make_ref,
    parse_ref,
    partition_for,
)

DIGEST64 = "a" * 64


def test_make_and_parse_roundtrip() -> None:
    ref = make_ref("simmdl", DIGEST64)
    assert ref == f"simmdl_{'a' * 16}"
    kind, hex16 = parse_ref(ref)
    assert kind == "simmdl"
    assert hex16 == "a" * 16
    assert partition_for(ref) == "models"
    assert classify(ref) == "sim"


def test_all_kinds_map_to_partitions() -> None:
    expected = {
        "simmdl": "models",
        "simsta": "states",
        "simtrc": "traces",
        "simobs": "states",
        "simadt": "audits",
        "simrnd": "renders",
        "simexp": "experiments",
        "simart": "experiments",
    }
    for kind, partition in expected.items():
        assert partition_for(make_ref(kind, DIGEST64)) == partition


def test_make_ref_requires_64_hex_digest() -> None:
    with pytest.raises(ValueError, match="REF_INVALID"):
        make_ref("simmdl", "a" * 16)  # 太短
    with pytest.raises(ValueError, match="REF_INVALID"):
        make_ref("simmdl", "G" * 64)  # 非 hex
    with pytest.raises(ValueError, match="REF_INVALID"):
        make_ref("nope", DIGEST64)  # 未知 kind


def test_reject_traversal() -> None:
    for bad in ("../../../etc/passwd", "simmdl_../../../../x", "..", "simmdl_/../.."):
        with pytest.raises(ValueError, match="REF_INVALID"):
            parse_ref(bad)
        assert classify(bad) == "invalid"


def test_reject_absolute_path_and_uri() -> None:
    for bad in ("/tmp/external.xml", "file:///etc/passwd", "https://x/y", "C:\\tmp\\x"):
        with pytest.raises(ValueError, match="REF_INVALID"):
            parse_ref(bad)
        assert classify(bad) == "invalid"


def test_reject_bad_hex_and_length() -> None:
    for bad in (
        "simmdl_" + "a" * 15,  # 短
        "simmdl_" + "a" * 17,  # 长
        "simmdl_" + "A" * 16,  # 大写
        "simmdl_" + "g" * 16,  # 非 hex
        "simmdl_aaaaaaaaaaaaaaaa.json",  # 带后缀
        "simmdl-aaaaaaaaaaaaaaaa",  # 错分隔符
        "",
    ):
        with pytest.raises(ValueError, match="REF_INVALID"):
            parse_ref(bad)


def test_legacy_classify() -> None:
    for legacy in ("model_" + "0" * 16, "obs_" + "f" * 16, "op_" + "1" * 16):
        assert is_legacy_ref(legacy)
        assert classify(legacy) == "legacy"
    # legacy 前缀但格式不对 → invalid
    assert classify("model_short") == "invalid"
    # 新前缀不判 legacy
    assert not is_legacy_ref("simmdl_" + "0" * 16)
