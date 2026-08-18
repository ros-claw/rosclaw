"""DEPRECATED package shim (PR-DF-24.2): ``rosclaw.auto.core`` moved to
``rosclaw.evolution.orchestrator.core``.  Kept for at least one full minor release; the CLI
(``rosclaw auto``) is unchanged.  Modules register into ``sys.modules``
so shim and canonical paths share ONE module object.
"""

import importlib as _importlib
import sys as _sys

artifact = _sys.modules[__name__ + ".artifact"] = _importlib.import_module("rosclaw.evolution.orchestrator.core.artifact")
champion = _sys.modules[__name__ + ".champion"] = _importlib.import_module("rosclaw.evolution.orchestrator.core.champion")
deadend = _sys.modules[__name__ + ".deadend"] = _importlib.import_module("rosclaw.evolution.orchestrator.core.deadend")
diagnosis = _sys.modules[__name__ + ".diagnosis"] = _importlib.import_module("rosclaw.evolution.orchestrator.core.diagnosis")
evaluation = _sys.modules[__name__ + ".evaluation"] = _importlib.import_module("rosclaw.evolution.orchestrator.core.evaluation")
experiment = _sys.modules[__name__ + ".experiment"] = _importlib.import_module("rosclaw.evolution.orchestrator.core.experiment")
failure = _sys.modules[__name__ + ".failure"] = _importlib.import_module("rosclaw.evolution.orchestrator.core.failure")
hypothesis = _sys.modules[__name__ + ".hypothesis"] = _importlib.import_module("rosclaw.evolution.orchestrator.core.hypothesis")
patch = _sys.modules[__name__ + ".patch"] = _importlib.import_module("rosclaw.evolution.orchestrator.core.patch")
proposal = _sys.modules[__name__ + ".proposal"] = _importlib.import_module("rosclaw.evolution.orchestrator.core.proposal")
task = _sys.modules[__name__ + ".task"] = _importlib.import_module("rosclaw.evolution.orchestrator.core.task")
