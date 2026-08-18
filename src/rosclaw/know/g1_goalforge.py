"""DEPRECATED import shim (PR-DF-24.1): the legacy Knowledge runtime moved
to rosclaw.knowledge.legacy.g1_goalforge.  This shim stays for at least one
full minor release — migrate imports to the canonical path.
"""

import sys as _sys

from rosclaw.knowledge.legacy import g1_goalforge as _g1_goalforge
from rosclaw.knowledge.legacy.g1_goalforge import *  # noqa: F401,F403

_sys.modules[__name__] = _g1_goalforge
