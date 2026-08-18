"""DEPRECATED package shim (PR-DF-24.2): ``rosclaw.auto.promotion`` moved to
``rosclaw.evolution.orchestrator.promotion``.  Kept for at least one full minor release; the CLI
(``rosclaw auto``) is unchanged.  Modules register into ``sys.modules``
so shim and canonical paths share ONE module object.
"""

import importlib as _importlib
import sys as _sys

champion_store = _sys.modules[__name__ + ".champion_store"] = _importlib.import_module("rosclaw.evolution.orchestrator.promotion.champion_store")
gate = _sys.modules[__name__ + ".gate"] = _importlib.import_module("rosclaw.evolution.orchestrator.promotion.gate")
lineage = _sys.modules[__name__ + ".lineage"] = _importlib.import_module("rosclaw.evolution.orchestrator.promotion.lineage")
rollback = _sys.modules[__name__ + ".rollback"] = _importlib.import_module("rosclaw.evolution.orchestrator.promotion.rollback")
