"""Governed dataset provenance and quality inspection.

Inspection is a learning-plane concern. It never authorizes a robot action,
and a technically complete snapshot remains ineligible until license and
target-applicability reviews occur elsewhere.
"""

from rosclaw.dataset.contracts import (
    DatasetDoctorReport,
    DatasetFileAnnotation,
    DatasetFileRecord,
    DatasetInventory,
    DatasetSnapshotState,
    DatasetSourceDescriptor,
    FileHashMode,
)
from rosclaw.dataset.doctor import inspect_dataset_root, write_dataset_doctor_artifacts
from rosclaw.dataset.registry import (
    DATASET_SOURCE_GROUP,
    DatasetAnnotationResolution,
    DatasetSource,
    DatasetSourceRegistry,
    register_sources,
)

__all__ = [
    "DATASET_SOURCE_GROUP",
    "DatasetAnnotationResolution",
    "DatasetDoctorReport",
    "DatasetFileAnnotation",
    "DatasetFileRecord",
    "DatasetInventory",
    "DatasetSnapshotState",
    "DatasetSource",
    "DatasetSourceDescriptor",
    "DatasetSourceRegistry",
    "FileHashMode",
    "inspect_dataset_root",
    "register_sources",
    "write_dataset_doctor_artifacts",
]
