"""Durable local store for product-facing execution receipts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from rosclaw.firstboot.workspace import resolve_home

if TYPE_CHECKING:
    from rosclaw.kernel import ExecutionReceipt

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")


class RunStoreError(RuntimeError):
    """A product run cannot be safely persisted or loaded."""


class RunNotFoundError(RunStoreError):
    """No receipt exists for the requested run reference."""


class ProductRunStore:
    """Persist receipts under ``$ROSCLAW_HOME/runs`` with an atomic latest pointer."""

    def __init__(self, home: Path | None = None) -> None:
        self.home = home.expanduser().resolve() if home is not None else resolve_home()
        self.root = self.home / "runs"

    def save(self, receipt: ExecutionReceipt) -> Path:
        """Persist one canonical receipt and mark it as the latest run."""

        self._assert_safe_root()
        run_id = self._validate_run_id(receipt.action_id)
        directory = self.root / run_id
        if directory.is_symlink():
            raise RunStoreError(f"Run directory cannot be a symbolic link: {run_id}")
        receipt_path = directory / "receipt.json"
        metadata_path = directory / "run.json"
        if metadata_path.exists():
            raise RunStoreError(f"Run already exists and cannot be overwritten: {run_id}")
        receipt_uri = receipt_path.resolve().as_uri()
        if receipt_uri not in receipt.artifacts:
            receipt.artifacts.append(receipt_uri)
        payload = receipt.to_dict()
        if receipt_path.exists():
            existing = self._read_json(receipt_path)
            if not self._matches_runtime_receipt(existing, payload):
                raise RunStoreError(
                    f"Existing runtime receipt conflicts with completed run: {run_id}"
                )
        receipt_bytes = self._serialize(payload)
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        self._atomic_write_bytes(receipt_path, receipt_bytes)
        self._atomic_write(
            metadata_path,
            {
                "schema_version": "rosclaw.product_run.v1",
                "run_id": run_id,
                "receipt": f"{run_id}/receipt.json",
                "trace_id": receipt.trace_id,
                "robot": receipt.body_id,
                "capability": receipt.capability_id,
                "mode": receipt.mode.value,
                "final_state": receipt.final_state.value,
                "evidence_level": receipt.evidence_level.value,
                "verified": receipt.verified,
                "finished_at": payload.get("finished_at"),
                "receipt_sha256": receipt_sha256,
            },
        )
        self._atomic_write(
            self.root / "latest.json",
            {
                "schema_version": "rosclaw.product_run_pointer.v1",
                "run_id": run_id,
                "receipt": f"{run_id}/receipt.json",
                "finished_at": payload.get("finished_at"),
                "receipt_sha256": receipt_sha256,
            },
        )
        return receipt_path

    def load(self, reference: str = "latest") -> tuple[dict[str, Any], Path]:
        """Load a receipt by run ID or the ``latest`` pointer."""

        self._assert_safe_root()
        if reference == "latest":
            pointer_path = self.root / "latest.json"
            if not pointer_path.is_file():
                raise RunNotFoundError(
                    "No product runs found. Run `rosclaw demo run ur5e-reach` first."
                )
            pointer = self._read_json(pointer_path)
            run_id = self._validate_run_id(str(pointer.get("run_id", "")))
            expected_pointer = f"{run_id}/receipt.json"
            if pointer.get("receipt") != expected_pointer:
                raise RunStoreError("Latest run pointer contains an invalid receipt path.")
        else:
            run_id = self._validate_run_id(reference)

        directory = self.root / run_id
        if directory.is_symlink():
            raise RunStoreError(f"Run directory cannot be a symbolic link: {run_id}")
        metadata_path = directory / "run.json"
        metadata = self._read_json(metadata_path)
        if str(metadata.get("run_id", "")) != run_id:
            raise RunStoreError(f"Run metadata does not match its directory: {run_id}")
        if metadata.get("receipt") != f"{run_id}/receipt.json":
            raise RunStoreError(f"Run metadata contains an invalid receipt path: {run_id}")

        receipt_path = directory / "receipt.json"
        if not receipt_path.is_file():
            raise RunNotFoundError(f"Run receipt not found: {reference}")
        receipt_bytes = self._read_bytes(receipt_path)
        expected_digest = str(metadata.get("receipt_sha256", ""))
        actual_digest = hashlib.sha256(receipt_bytes).hexdigest()
        if not expected_digest or actual_digest != expected_digest:
            raise RunStoreError(f"Receipt digest does not match run metadata: {run_id}")
        if reference == "latest" and pointer.get("receipt_sha256") != actual_digest:
            raise RunStoreError("Latest run pointer digest does not match the receipt.")
        receipt = self._decode_json(receipt_bytes, receipt_path)
        if str(receipt.get("action_id", "")) != run_id:
            raise RunStoreError(f"Receipt action_id does not match its run directory: {run_id}")
        expected_values = {
            "trace_id": receipt.get("trace_id"),
            "robot": receipt.get("body_id"),
            "capability": receipt.get("capability_id"),
            "mode": receipt.get("execution_mode"),
            "final_state": receipt.get("final_state"),
            "evidence_level": receipt.get("evidence_level"),
            "verified": receipt.get("verified"),
            "finished_at": receipt.get("finished_at"),
        }
        for key, expected in expected_values.items():
            if metadata.get(key) != expected:
                raise RunStoreError(
                    f"Run metadata field {key!r} does not match its receipt: {run_id}"
                )
        return receipt, receipt_path

    def list(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """List newest persisted runs without following symlinks."""

        self._assert_safe_root()
        if limit < 1:
            return []
        if not self.root.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for path in self.root.glob("*/run.json"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                item = self._read_json(path)
                run_id = self._validate_run_id(path.parent.name)
                self.load(run_id)
            except (RunStoreError, ValueError, OSError):
                continue
            items.append(item)
        items.sort(key=lambda item: str(item.get("finished_at", "")), reverse=True)
        return items[:limit]

    @staticmethod
    def _validate_run_id(run_id: str) -> str:
        if not SAFE_RUN_ID.fullmatch(run_id):
            raise RunStoreError(f"Invalid run reference: {run_id!r}")
        return run_id

    def _assert_safe_root(self) -> None:
        if self.root.is_symlink():
            raise RunStoreError(f"Run store cannot be a symbolic link: {self.root}")
        if self.root.exists() and not self.root.is_dir():
            raise RunStoreError(f"Run store must be a directory: {self.root}")

    @staticmethod
    def _matches_runtime_receipt(
        existing: dict[str, Any],
        completed: dict[str, Any],
    ) -> bool:
        existing_copy = dict(existing)
        completed_copy = dict(completed)
        existing_artifacts = existing_copy.pop("artifacts", [])
        completed_artifacts = completed_copy.pop("artifacts", [])
        return (
            existing_copy == completed_copy
            and isinstance(existing_artifacts, list)
            and isinstance(completed_artifacts, list)
            and all(isinstance(item, str) for item in existing_artifacts)
            and all(isinstance(item, str) for item in completed_artifacts)
            and set(existing_artifacts).issubset(set(completed_artifacts))
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return ProductRunStore._decode_json(ProductRunStore._read_bytes(path), path)

    @staticmethod
    def _read_bytes(path: Path) -> bytes:
        if path.is_symlink():
            raise RunStoreError(f"Run metadata cannot be a symbolic link: {path}")
        try:
            return path.read_bytes()
        except OSError as exc:
            raise RunStoreError(f"Cannot read run metadata {path}: {exc}") from exc

    @staticmethod
    def _decode_json(data: bytes, path: Path) -> dict[str, Any]:
        try:
            raw = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunStoreError(f"Cannot decode run metadata {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise RunStoreError(f"Run metadata must be a JSON object: {path}")
        return cast(dict[str, Any], raw)

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        ProductRunStore._atomic_write_bytes(path, ProductRunStore._serialize(payload))

    @staticmethod
    def _serialize(payload: dict[str, Any]) -> bytes:
        return json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.is_symlink():
            raise RunStoreError(f"Run directory cannot be a symbolic link: {path.parent}")
        if path.is_symlink():
            raise RunStoreError(f"Run metadata cannot be a symbolic link: {path}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        if temporary.is_symlink():
            raise RunStoreError(f"Temporary run path cannot be a symbolic link: {temporary}")
        temporary.write_bytes(data)
        temporary.replace(path)


__all__ = [
    "ProductRunStore",
    "RunNotFoundError",
    "RunStoreError",
]
