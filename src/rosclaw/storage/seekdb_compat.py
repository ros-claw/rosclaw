"""SeekDB version/compatibility matrix — single source of truth (PR-SDB-140-1).

The upgrade outline's core discipline: Engine version, pyseekdb SDK version,
pylibseekdb binding version and the new seekdb-bindings package are
DIFFERENT version lines.  ``seekdb 1.4.0 != pyseekdb 1.4.0``.  This module
owns:

* the validated / candidate / known-bad matrix (was: comments in
  pyproject.toml + a lock-file note + a logger.error);
* fail-fast on the known-bad SDK (oceanbase/pyseekdb#251) unless the
  operator explicitly overrides with ``ROSCLAW_ALLOW_KNOWN_BAD_SEEKDB=1``;
* explicit capability detection — deployment mode is decided by
  construction (``path`` vs ``host``) and reported, never guessed later.

Versions are free-form strings because upstream uses ``.postN`` / ``.devN``
suffixes that defeat naive tuple comparison; we match exactly and treat
anything else as "untested" (allowed, warned).
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

logger = logging.getLogger("rosclaw.storage.seekdb_compat")

# --- the matrix ------------------------------------------------------------
#: Officially compatible with seekdb Engine 1.3.x AND the 1.4.0 engine
#: (the v1.4.0 release notes name pyseekdb 1.3.0 as the compatible SDK).
#: 1.4.0.post1 earned VALIDATED in PR-SDB-140-3: #251 repro PASS, T0-T15
#: matrix 16/16 on its bundled 1.4 embedded engine, 273/273 storage+seekdb
#: suites, server benchmark no-regression, zh/en golden-corpus recall
#: identical, p95 latencies equal-or-better (BM25 -33%, metadata -49%,
#: W2R -21% vs SDK 1.3.0 on the same 1.4.0 engine).
VALIDATED_SDK_VERSIONS = frozenset({"1.3.0", "1.4.0.post1"})
#: Currently no candidate in flight.
CANDIDATE_SDK_VERSIONS = frozenset()
#: Hard-broken SQL generation on metadata-filtered search legs.
KNOWN_BAD_SDK_VERSIONS = frozenset({"1.4.0"})

ALLOW_KNOWN_BAD_ENV = "ROSCLAW_ALLOW_KNOWN_BAD_SEEKDB"

DeploymentMode = Literal["legacy_embedded", "local_runtime", "server", "unknown"]


@dataclass(frozen=True)
class SeekDBCapabilities:
    """What this deployment can do — drive behavior off this, not off
    ``if self._path`` guesses scattered through the store."""

    engine_version: str | None
    sdk_version: str | None
    binding_version: str | None
    deployment: DeploymentMode

    native_rrf: bool
    metadata_filter: bool
    vector_search: bool
    bm25: bool
    hybrid_search: bool
    multi_instance: bool
    explicit_refresh: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def installed_distribution_version(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 — metadata corruption must not break storage
        return None


def classify_sdk_version(installed: str | None) -> str:
    """One of: missing | validated | candidate | known_bad | untested."""
    if installed is None:
        return "missing"
    if installed in KNOWN_BAD_SDK_VERSIONS:
        return "known_bad"
    if installed in VALIDATED_SDK_VERSIONS:
        return "validated"
    if installed in CANDIDATE_SDK_VERSIONS:
        return "candidate"
    return "untested"


def validate_sdk_version(installed: str | None) -> str:
    """Fail fast on the known-bad SDK; warn on anything outside the matrix.

    Returns the classification.  Raises RuntimeError for the known-bad
    version unless ROSCLAW_ALLOW_KNOWN_BAD_SEEKDB=1 — a lab escape hatch,
    never a production path.
    """
    classification = classify_sdk_version(installed)
    if classification == "known_bad":
        message = (
            f"pyseekdb {installed} is KNOWN-INCOMPATIBLE with the embedded SeekDB "
            "engine (broken SQL generation on metadata-filtered search legs; "
            "oceanbase/pyseekdb#251). Pin pyseekdb==1.3.0 (production) or "
            "pyseekdb==1.4.0.post1 (candidate). Set "
            f"{ALLOW_KNOWN_BAD_ENV}=1 to override in a lab setting."
        )
        if os.environ.get(ALLOW_KNOWN_BAD_ENV) == "1":
            logger.error("%s — overridden by %s=1", message, ALLOW_KNOWN_BAD_ENV)
            return classification
        raise RuntimeError(message)
    if classification in ("untested",):
        logger.warning(
            "pyseekdb %s is outside the validated version matrix (%s); "
            "native SeekDB behaviour may drift silently",
            installed,
            "/".join(sorted(VALIDATED_SDK_VERSIONS | CANDIDATE_SDK_VERSIONS)),
        )
    return classification


def detect_capabilities(
    *,
    path: str | None,
    host: str | None,
    engine_version: str | None = None,
) -> SeekDBCapabilities:
    """Capabilities from construction facts + installed distributions.

    ``local_runtime`` (seekdb 1.4 background-process embedded) is declared
    but not constructible yet — PR-SDB-140-4 builds it.
    """
    sdk = installed_distribution_version("pyseekdb")
    binding = installed_distribution_version("pylibseekdb") or installed_distribution_version(
        "seekdb-lib"
    )
    if path is not None:
        deployment: DeploymentMode = "legacy_embedded"
    elif host is not None:
        deployment = "server"
    else:
        deployment = "unknown"

    # Current truth (1.3-era): native RRF hybrid fusion only on server
    # deployments; embedded stays single-target; explicit refresh exists on
    # both.  Engine 1.4 lanes re-derive these in PR-2/PR-4.
    server = deployment == "server"
    return SeekDBCapabilities(
        engine_version=engine_version,
        sdk_version=sdk,
        binding_version=binding,
        deployment=deployment,
        native_rrf=server,
        metadata_filter=True,
        vector_search=True,
        bm25=True,
        hybrid_search=True,
        multi_instance=server,
        explicit_refresh=True,
    )


def log_capabilities(caps: SeekDBCapabilities) -> None:
    """One structured line at connect time (outline §四)."""
    logger.info(
        "SeekDB capabilities: deployment=%s engine=%s sdk=%s binding=%s "
        "native_rrf=%s multi_instance=%s explicit_refresh=%s",
        caps.deployment,
        caps.engine_version or "unknown",
        caps.sdk_version or "missing",
        caps.binding_version or "missing",
        caps.native_rrf,
        caps.multi_instance,
        caps.explicit_refresh,
    )
