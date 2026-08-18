"""PR-DF-24.2 (phase-II DF-16.2): Evolution physical cleanup.

``rosclaw.auto`` physically moved to ``rosclaw.evolution.orchestrator``;
the old package stays as a deprecation shim (CLI ``rosclaw auto``
unchanged).  Shim modules register into ``sys.modules`` so both paths
share ONE module object — class identity holds across paths.
"""

from __future__ import annotations


def test_canonical_engine_and_plugin():
    from rosclaw.evolution.orchestrator.engine.auto_engine import AutoEngine
    from rosclaw.evolution.orchestrator.plugin import AutoPlugin

    assert AutoEngine is not None and AutoPlugin is not None


def test_shim_identity_single_module_object():
    import rosclaw.auto.engine.auto_engine as shim_mod
    import rosclaw.evolution.orchestrator.engine.auto_engine as canon_mod
    from rosclaw.auto.engine.auto_engine import AutoEngine as ShimEngine
    from rosclaw.evolution.orchestrator.engine.auto_engine import AutoEngine as CanonEngine

    assert shim_mod is canon_mod
    assert ShimEngine is CanonEngine


def test_shim_package_attribute_access():
    import rosclaw.auto
    import rosclaw.evolution.orchestrator.plugin

    assert rosclaw.auto.plugin.AutoPlugin is rosclaw.evolution.orchestrator.plugin.AutoPlugin


def test_shim_deep_paths():
    from rosclaw.auto.storage.local_store import LocalStore as ShimStore
    from rosclaw.evolution.orchestrator.storage.local_store import LocalStore as CanonStore

    assert ShimStore is CanonStore
    from rosclaw.auto.promotion.gate import PromotionGate as ShimGate
    from rosclaw.evolution.orchestrator.promotion.gate import PromotionGate as CanonGate

    assert ShimGate is CanonGate


def test_no_shim_imports_inside_src():
    import subprocess

    # Import statements only (DF-24.1 precedent): event-topic strings like
    # "rosclaw.auto.proposal.created" are wire protocol, not source layout,
    # and must survive the physical move untouched.
    out = subprocess.run(
        ["grep", "-rnE", "(from|import)\\s+rosclaw\\.auto(\\.|\\s|$)", "src/rosclaw", "--include=*.py"],
        capture_output=True,
        text=True,
    ).stdout
    offenders = [
        line
        for line in out.splitlines()
        if "src/rosclaw/auto/" not in line
        and "evolution/orchestrator" not in line
        and "getLogger" not in line
    ]
    assert not offenders, f"shim imports survived in src: {offenders}"
