"""DEPRECATED package shim (PR-DF-24.2): ``rosclaw.auto.storage`` moved to
``rosclaw.evolution.orchestrator.storage``.  Kept for at least one full minor release; the CLI
(``rosclaw auto``) is unchanged.  Modules register into ``sys.modules``
so shim and canonical paths share ONE module object.
"""

import importlib as _importlib
import sys as _sys

local_store = _sys.modules[__name__ + ".local_store"] = _importlib.import_module("rosclaw.evolution.orchestrator.storage.local_store")
