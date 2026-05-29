"""Protocol type stubs for powermem embodied interfaces."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class WorldObjectLike(Protocol):
    """Stub WorldObjectLike protocol."""
    obj_id: str


@runtime_checkable
class PoseLike(Protocol):
    """Stub PoseLike protocol."""
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


@runtime_checkable
class Vec3Like(Protocol):
    """Stub Vec3Like protocol."""
    x: float
    y: float
    z: float


@runtime_checkable
class TemporalIntervalLike(Protocol):
    """Stub TemporalIntervalLike protocol."""
    start: float
    end: float


@runtime_checkable
class PermanenceReportLike(Protocol):
    """Stub PermanenceReportLike protocol."""
    report_id: str


@runtime_checkable
class MemoryAtomLike(Protocol):
    """Stub MemoryAtomLike protocol."""
    atom_id: str
