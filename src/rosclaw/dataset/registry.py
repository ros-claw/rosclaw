"""Runtime-free registry for downstream dataset source classifiers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from rosclaw.dataset.contracts import DatasetFileAnnotation, DatasetSourceDescriptor
from rosclaw.extension_discovery import (
    EntryPointDiscoveryReport,
    discover_entry_point_objects,
)

DATASET_SOURCE_GROUP = "rosclaw.dataset.sources"
_DISABLE_ENV = "ROSCLAW_DISABLE_DATASET_EXTENSIONS"


@runtime_checkable
class DatasetSource(Protocol):
    """Classify relative names only; no root, file handle, runtime, or driver."""

    descriptor: DatasetSourceDescriptor

    def classify_file(self, dataset_id: str, relative_path: str) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class DatasetAnnotationResolution:
    annotations: tuple[DatasetFileAnnotation, ...]
    errors: tuple[str, ...]


class DatasetSourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, DatasetSource] = {}

    def register_source(self, source: Any) -> str:
        if not isinstance(source, DatasetSource):
            raise TypeError("dataset source does not satisfy the DatasetSource protocol")
        if not isinstance(source.descriptor, DatasetSourceDescriptor):
            raise TypeError("dataset source descriptor has the wrong contract type")
        source_id = source.descriptor.source_id
        if source_id in self._sources:
            raise ValueError(f"duplicate dataset source: {source_id}")
        self._sources[source_id] = source
        return source_id

    def source(self, source_id: str) -> DatasetSource:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise KeyError(f"unknown dataset source: {source_id}") from exc

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._sources))

    @property
    def descriptors(self) -> tuple[DatasetSourceDescriptor, ...]:
        return tuple(self._sources[source_id].descriptor for source_id in self.source_ids)

    def discover(self) -> EntryPointDiscoveryReport:
        return discover_entry_point_objects(
            group=DATASET_SOURCE_GROUP,
            disable_env=_DISABLE_ENV,
            register=self.register_source,
        )

    def classify(self, *, dataset_id: str, relative_path: str) -> DatasetAnnotationResolution:
        annotations: list[DatasetFileAnnotation] = []
        errors: list[str] = []
        for source_id in self.source_ids:
            source = self._sources[source_id]
            descriptor = source.descriptor
            if dataset_id not in descriptor.dataset_ids:
                continue
            try:
                labels = source.classify_file(dataset_id, relative_path)
                normalized = _validated_labels(labels, descriptor=descriptor)
            except Exception as exc:
                errors.append(_bounded_error(source_id, exc))
                continue
            if normalized:
                annotations.append(
                    DatasetFileAnnotation(
                        relative_path=relative_path,
                        source_id=source_id,
                        label_ids=normalized,
                    )
                )
        return DatasetAnnotationResolution(
            annotations=tuple(annotations),
            errors=tuple(errors),
        )


def _validated_labels(
    labels: Any,
    *,
    descriptor: DatasetSourceDescriptor,
) -> tuple[str, ...]:
    if not isinstance(labels, tuple):
        raise TypeError("classify_file must return a stable tuple")
    if len(labels) != len(set(labels)):
        raise ValueError("classify_file labels must be unique")
    allowed = set(descriptor.label_ids)
    if any(not isinstance(value, str) or value not in allowed for value in labels):
        raise ValueError("classify_file returned a label outside its descriptor vocabulary")
    return labels


def _bounded_error(source_id: str, exc: Exception) -> str:
    detail = " ".join(str(exc).split())[:768]
    return f"{source_id}: {type(exc).__name__}: {detail}"


def register_sources(
    registry: DatasetSourceRegistry,
    sources: Iterable[DatasetSource],
) -> tuple[str, ...]:
    """Small helper for deterministic in-process composition and tests."""

    return tuple(registry.register_source(source) for source in sources)


__all__ = [
    "DATASET_SOURCE_GROUP",
    "DatasetAnnotationResolution",
    "DatasetSource",
    "DatasetSourceRegistry",
    "register_sources",
]
