"""e-URDF Loader — Physical DNA Registry.

Re-exports EURDFLoader from runtime.eurdf_loader
so consumers can import from the unified `rosclaw.eurdf` namespace.
"""

from rosclaw.runtime.eurdf_loader import EURDFLoader

__all__ = ["EURDFLoader"]
