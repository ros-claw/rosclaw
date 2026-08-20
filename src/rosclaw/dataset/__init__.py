"""Governed dataset inventory and quality inspection.

Dataset inspection is an asynchronous learning-plane concern.  It never
authorizes a robot action and a technically complete snapshot is still not
training-eligible until its license and applicability are reviewed.
"""

from rosclaw.dataset.contracts import (
    DatasetDoctorReport,
    DatasetFileRecord,
    DatasetInventory,
    DatasetSnapshotState,
    FileHashMode,
)
from rosclaw.dataset.doctor import inspect_dataset_root, write_dataset_doctor_artifacts

__all__ = [
    "DatasetDoctorReport",
    "DatasetFileRecord",
    "DatasetInventory",
    "DatasetSnapshotState",
    "FileHashMode",
    "inspect_dataset_root",
    "write_dataset_doctor_artifacts",
]
