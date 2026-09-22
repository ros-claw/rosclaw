"""任务本地不可变对象存储（PR-MH1，ADR-0014，规格 §11/§39）。

Maturity: experimental（ADR-0000 §4）。

布局：``task_root/sim/{models,states,traces,audits,renders,experiments}/<ref>.json|bin``

语义：
- 内容寻址：ref 后缀 = sha256(内容)[:16]，不同内容必然不同 ref；
- 幂等写：同 ref 同内容 → 直接返回（Agent 重试场景唯一安全语义）；
- 绝不覆写：同 ref 异内容（构造上不可达，防御盘外篡改）→
  ``STORE_IMMUTABLE_VIOLATION``；
- fail closed：路径逃逸 / symlink 逃逸 / 外部 URI / 超限对象 /
  盘外篡改全部报错；
- legacy 桥：``model_``/``obs_``/``op_`` ref 只读解析到
  ``task_root/models/``（sim/api.py 现行布局），写入一律拒绝。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rosclaw.contracts.common import canonical_json
from rosclaw.sim.refs import (
    PREFIX_TO_PARTITION,
    classify,
    is_legacy_ref,
    make_ref,
    parse_ref,
)

PARTITIONS: tuple[str, ...] = (
    "models",
    "states",
    "traces",
    "audits",
    "renders",
    "experiments",
)

#: 分区未显式给 ref 时的默认 kind（spec §11 六分区一一对应）。
PARTITION_DEFAULT_KIND: dict[str, str] = {
    "models": "simmdl",
    "states": "simsta",
    "traces": "simtrc",
    "audits": "simadt",
    "renders": "simrnd",
    "experiments": "simexp",
}

DEFAULT_MAX_BYTES = 64 * 1024 * 1024


class SimStore:
    """任务本地不可变仿真对象存储。"""

    def __init__(self, task_root: Path | str, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._task_root = Path(task_root)
        self._root = self._task_root / "sim"
        self._max_bytes = max_bytes

    @property
    def root(self) -> Path:
        return self._root

    # -- 写入 ---------------------------------------------------------------

    def put(
        self,
        partition: str,
        payload: dict[str, Any] | bytes,
        *,
        ref: str | None = None,
        max_bytes: int | None = None,
    ) -> str:
        """写入不可变对象并返回其 ref（同内容幂等）。

        max_bytes：显式单次预算覆盖（默认沿用 store 级策略 64MB）。
        仅允许**调用方在代码中显式声明**的合法大工件路径使用
        （如 .mjz 自包含模型导出，见 backend.export_model_mjz），
        不是静默放宽全局上限的通道。"""
        if partition not in PARTITIONS:
            raise ValueError(f"STORE_PARTITION_UNKNOWN: {partition!r}")

        if isinstance(payload, dict):
            data = canonical_json(payload).encode("utf-8")
            suffix = ".json"
        elif isinstance(payload, (bytes, bytearray)):
            data = bytes(payload)
            suffix = ".bin"
        else:
            raise ValueError(f"STORE_PAYLOAD_UNSUPPORTED: {type(payload).__name__}")

        budget = self._max_bytes if max_bytes is None else max_bytes
        if len(data) > budget:
            raise ValueError(f"STORE_OBJECT_TOO_LARGE: {len(data)} > {budget}")

        digest_hex = hashlib.sha256(data).hexdigest()
        if ref is None:
            ref = make_ref(PARTITION_DEFAULT_KIND[partition], digest_hex)
        else:
            if is_legacy_ref(ref):
                raise ValueError(f"STORE_LEGACY_READONLY: {ref!r}")
            kind, hex16 = parse_ref(ref)  # 词法 fail closed
            if PREFIX_TO_PARTITION[kind] != partition:
                raise ValueError(f"REF_INVALID: {ref!r} does not belong to partition {partition!r}")
            if hex16 != digest_hex[:16]:
                raise ValueError(f"STORE_DIGEST_MISMATCH: explicit ref {ref!r} != content digest")

        target = self._root / partition / f"{ref}{suffix}"
        self._assert_inside(target)

        if target.exists() or target.is_symlink():
            if target.is_symlink():
                raise ValueError(f"STORE_PATH_ESCAPE: {target} is a symlink")
            existing = target.read_bytes()
            if existing == data:
                return ref  # 幂等
            raise ValueError(f"STORE_IMMUTABLE_VIOLATION: {ref!r} exists with different content")

        target.parent.mkdir(parents=True, exist_ok=True)
        # 同目录临时文件 + os.replace 原子落盘。
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp_")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp_name, target)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return ref

    # -- 读取 ---------------------------------------------------------------

    def get(self, ref: str) -> dict[str, Any] | bytes:
        """读取对象；dict 负载返回 dict，bytes 负载返回 bytes。"""
        if classify(ref) == "legacy":
            return self._get_legacy(ref)
        path = self.resolve(ref)
        data = path.read_bytes()
        self._check_digest(ref, data)
        if path.suffix == ".json":
            return json.loads(data.decode("utf-8"))
        return data

    def exists(self, ref: str) -> bool:
        if classify(ref) != "sim":
            if is_legacy_ref(ref):
                return (self._task_root / "models" / f"{ref}.json").is_file()
            return False
        return self._candidate(ref).is_file()

    def verify_digest(self, ref: str) -> bool:
        """重算内容与 ref 后缀比对；对象缺失或篡改返回 False。"""
        if classify(ref) != "sim":
            return False
        path = self._candidate(ref)
        if not path.is_file() or path.is_symlink():
            return False
        _, hex16 = parse_ref(ref)
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16] == hex16

    def resolve(self, ref: str) -> Path:
        """解析 ref 为绝对路径；对象必须存在且未逃逸。"""
        if classify(ref) == "legacy":
            path = self._task_root / "models" / f"{ref}.json"
            if not path.is_file():
                raise ValueError(f"REF_NOT_FOUND: {ref!r}")
            return path
        path = self._candidate(ref)
        if path.is_symlink():
            raise ValueError(f"STORE_PATH_ESCAPE: {path} is a symlink")
        if not path.is_file():
            raise ValueError(f"REF_NOT_FOUND: {ref!r}")
        return path

    def list_children(self, partition: str) -> list[str]:
        """列举分区内全部 ref（排序，确定性）。"""
        if partition not in PARTITIONS:
            raise ValueError(f"STORE_PARTITION_UNKNOWN: {partition!r}")
        folder = self._root / partition
        if not folder.is_dir():
            return []
        refs = []
        for entry in sorted(folder.iterdir()):
            if (
                entry.is_file()
                and not entry.is_symlink()
                and entry.suffix in (".json", ".bin")
                and classify(entry.stem) == "sim"
            ):
                refs.append(entry.stem)
        return refs

    # -- 内部 ---------------------------------------------------------------

    def _candidate(self, ref: str) -> Path:
        kind, _ = parse_ref(ref)  # 词法 fail closed
        partition = PREFIX_TO_PARTITION[kind]
        base = self._root / partition / f"{ref}.json"
        if base.exists() or not (self._root / partition / f"{ref}.bin").exists():
            return base
        return self._root / partition / f"{ref}.bin"

    def _assert_inside(self, path: Path) -> None:
        root_resolved = self._root.resolve()
        resolved = path.resolve(strict=False)
        if resolved != root_resolved and root_resolved not in resolved.parents:
            raise ValueError(f"STORE_PATH_ESCAPE: {path} resolves outside {root_resolved}")

    def _check_digest(self, ref: str, data: bytes) -> None:
        _, hex16 = parse_ref(ref)
        if hashlib.sha256(data).hexdigest()[:16] != hex16:
            raise ValueError(f"STORE_DIGEST_MISMATCH: {ref!r} content tampered")

    def _get_legacy(self, ref: str) -> dict[str, Any]:
        """legacy 只读桥：sim/api.py 的 task_root/models/<ref>.json。"""
        path = self._task_root / "models" / f"{ref}.json"
        if not path.is_file():
            raise ValueError(f"REF_NOT_FOUND: {ref!r}")
        # legacy ref 的 digest 语义与 canonical_json 不同（历史 json.dumps
        # 默认分隔符），不做内容重算——只读兼容，不代表新对象契约。
        return json.loads(path.read_text(encoding="utf-8"))
