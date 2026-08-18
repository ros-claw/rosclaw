"""DEPRECATED package shim (PR-DF-24.2): ``rosclaw.auto.diagnosis`` moved to
``rosclaw.evolution.orchestrator.diagnosis``.  Kept for at least one full minor release; the CLI
(``rosclaw auto``) is unchanged.  Modules register into ``sys.modules``
so shim and canonical paths share ONE module object.
"""

import importlib as _importlib
import sys as _sys

rule_diagnoser = _sys.modules[__name__ + ".rule_diagnoser"] = _importlib.import_module("rosclaw.evolution.orchestrator.diagnosis.rule_diagnoser")
