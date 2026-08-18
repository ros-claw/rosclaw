"""DEPRECATED import shim (PR-DF-24.1): the legacy Knowledge runtime moved
to rosclaw.knowledge.legacy.storage.  This shim stays for at least one
full minor release — migrate imports to the canonical path.
"""

import sys as _sys

from rosclaw.knowledge.legacy import storage as _storage
from rosclaw.knowledge.legacy.storage import *  # noqa: F401,F403

_sys.modules[__name__] = _storage
