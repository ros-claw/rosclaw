"""DEPRECATED package shim (PR-DF-24.2): ``rosclaw.auto.events`` moved to
``rosclaw.evolution.orchestrator.events``.  Kept for at least one full minor release; the CLI
(``rosclaw auto``) is unchanged.  Modules register into ``sys.modules``
so shim and canonical paths share ONE module object.
"""

import importlib as _importlib
import sys as _sys

publishers = _sys.modules[__name__ + ".publishers"] = _importlib.import_module("rosclaw.evolution.orchestrator.events.publishers")
schemas = _sys.modules[__name__ + ".schemas"] = _importlib.import_module("rosclaw.evolution.orchestrator.events.schemas")
subscribers = _sys.modules[__name__ + ".subscribers"] = _importlib.import_module("rosclaw.evolution.orchestrator.events.subscribers")
