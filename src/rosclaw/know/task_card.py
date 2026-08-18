"""DEPRECATED import shim (PR-DF-24.1): the legacy Knowledge runtime moved
to rosclaw.knowledge.legacy.task_card.  This shim stays for at least one
full minor release — migrate imports to the canonical path.
"""

import sys as _sys

from rosclaw.knowledge.legacy import task_card as _task_card
from rosclaw.knowledge.legacy.task_card import *  # noqa: F401,F403

_sys.modules[__name__] = _task_card
