"""PR-DF-24.3 (phase-II DF-16.3): Memory physical cleanup.

``rosclaw.memory.v2.*`` moved up to ``rosclaw.memory.*``; the v2 package
stays as a deprecation shim.  The DATA schema name stays ``memory.v2`` —
source layout version ≠ protocol version.
"""

from __future__ import annotations


def test_protocol_version_name_unchanged():
    from rosclaw.memory.models import SCHEMA_VERSION

    assert SCHEMA_VERSION == "memory.v2"


def test_shim_identity():
    import rosclaw.memory.distillation_service as canon_mod
    import rosclaw.memory.v2.distillation_service as shim_mod
    from rosclaw.memory.distillation_service import MemoryDistillationService as Canon
    from rosclaw.memory.v2.distillation_service import MemoryDistillationService as Shim

    assert shim_mod is canon_mod
    assert Shim is Canon


def test_shim_public_api():
    from rosclaw.memory.models import MemoryItem
    from rosclaw.memory.repository import MemoryRepository
    from rosclaw.memory.v2 import MemoryItem as ShimItem
    from rosclaw.memory.v2 import MemoryRepository as ShimRepo

    assert ShimItem is MemoryItem
    assert ShimRepo is MemoryRepository


def test_shim_deep_subpackages():
    from rosclaw.memory.regime.matcher import RegimeMatcher as CanonMatcher
    from rosclaw.memory.runtime_retrieval.facade import (
        MemoryRetrievalFacade as CanonFacade,
    )
    from rosclaw.memory.v2.regime.matcher import RegimeMatcher as ShimMatcher
    from rosclaw.memory.v2.runtime_retrieval.facade import (
        MemoryRetrievalFacade as ShimFacade,
    )

    assert CanonMatcher is ShimMatcher and CanonFacade is ShimFacade


def test_no_shim_imports_inside_src():
    import subprocess

    out = subprocess.run(
        ["grep", "-rnE", "(from|import)\\s+rosclaw\\.memory\\.v2(\\.|\\s|$)", "src/rosclaw", "--include=*.py"],
        capture_output=True,
        text=True,
    ).stdout
    offenders = [
        line
        for line in out.splitlines()
        if "src/rosclaw/memory/v2/" not in line and "rosclaw.memory.v2_PLACEHOLDER" not in line
    ]
    assert not offenders, f"shim imports survived in src: {offenders}"
