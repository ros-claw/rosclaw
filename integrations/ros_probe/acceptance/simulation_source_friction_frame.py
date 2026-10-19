"""Compatibility export for the existing pure fixed-SDK SIM preparation API."""

from rosclaw.connectors.ros.context.source_friction import (
    SDK_IMAGE,
    prepare_source_friction_frame,
)

__all__ = ["SDK_IMAGE", "prepare_source_friction_frame"]
