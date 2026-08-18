"""PR-DF-24.1 (phase-II DF-16.1): Knowledge physical cleanup.

``rosclaw.know`` physically moved to ``rosclaw.knowledge.legacy``; the old
package stays as a deprecation shim for at least one full minor release.
"""

from __future__ import annotations


def test_canonical_package_has_runtime():
    from rosclaw.knowledge.legacy import (
        KnowledgeInterface,
        LegacyKnowledgeRuntime,
    )

    assert LegacyKnowledgeRuntime is KnowledgeInterface


def test_shim_package_reexports_same_objects():
    import rosclaw.know
    from rosclaw.knowledge.legacy import KnowledgeInterface as Canonical

    assert rosclaw.know.KnowledgeInterface is Canonical
    assert rosclaw.know.LegacyKnowledgeRuntime is Canonical


def test_shim_submodules_resolve_to_canonical_modules():
    from rosclaw.know import interface as shim_interface
    from rosclaw.knowledge.legacy import interface as canonical_interface

    assert shim_interface is canonical_interface
    assert shim_interface.KnowledgeInterface is canonical_interface.KnowledgeInterface

    from rosclaw.know import task_pack_adapter as shim_adapter
    from rosclaw.knowledge.legacy import task_pack_adapter as canonical_adapter

    assert shim_adapter is canonical_adapter


def test_no_new_dependencies_on_shim_inside_src():
    """In-repo code must IMPORT the canonical path, not the shim.

    Only import statements count — wire-protocol version strings
    (``rosclaw.know.evidence_ref.v2`` etc.) are the DATA vocabulary and
    must NOT change with the source layout (DF-16.3's warning: source
    layout version ≠ protocol version).
    """
    import subprocess

    out = subprocess.run(
        ["grep", "-rnE", "(from|import)\\s+rosclaw\\.know(\\.|\\s)", "src/rosclaw", "--include=*.py"],
        capture_output=True,
        text=True,
    ).stdout
    offenders = [
        line
        for line in out.splitlines()
        if "knowledge/legacy" not in line
        and "src/rosclaw/know/" not in line
        and "rosclaw_know" not in line
    ]
    assert not offenders, f"shim imports survived in src: {offenders}"
