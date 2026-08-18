"""DEPRECATED import shim (PR-DF-24.1): the legacy Knowledge runtime moved
to rosclaw.knowledge.legacy.evidence_ingest.  This shim stays for at least one
full minor release — migrate imports to the canonical path.
"""

import sys as _sys

from rosclaw.knowledge.legacy import evidence_ingest as _evidence_ingest
from rosclaw.knowledge.legacy.evidence_ingest import *  # noqa: F401,F403

_sys.modules[__name__] = _evidence_ingest
