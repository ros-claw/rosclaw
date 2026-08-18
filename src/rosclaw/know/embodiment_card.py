"""DEPRECATED import shim (PR-DF-24.1): the legacy Knowledge runtime moved
to rosclaw.knowledge.legacy.embodiment_card.  This shim stays for at least one
full minor release — migrate imports to the canonical path.
"""

import sys as _sys

from rosclaw.knowledge.legacy import embodiment_card as _embodiment_card
from rosclaw.knowledge.legacy.embodiment_card import *  # noqa: F401,F403

_sys.modules[__name__] = _embodiment_card
