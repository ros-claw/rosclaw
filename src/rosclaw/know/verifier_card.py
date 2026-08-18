"""DEPRECATED import shim (PR-DF-24.1): the legacy Knowledge runtime moved
to rosclaw.knowledge.legacy.verifier_card.  This shim stays for at least one
full minor release — migrate imports to the canonical path.
"""

import sys as _sys

from rosclaw.knowledge.legacy import verifier_card as _verifier_card
from rosclaw.knowledge.legacy.verifier_card import *  # noqa: F401,F403

_sys.modules[__name__] = _verifier_card
