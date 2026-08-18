"""DEPRECATED package shim (PR-DF-24.2): ``rosclaw.auto.g1_kick`` moved to
``rosclaw.evolution.orchestrator.g1_kick``.  Kept for at least one full minor release; the CLI
(``rosclaw auto``) is unchanged.  Modules register into ``sys.modules``
so shim and canonical paths share ONE module object.
"""

import importlib as _importlib
import sys as _sys

continual_runner = _sys.modules[__name__ + ".continual_runner"] = _importlib.import_module("rosclaw.evolution.orchestrator.g1_kick.continual_runner")
curriculum = _sys.modules[__name__ + ".curriculum"] = _importlib.import_module("rosclaw.evolution.orchestrator.g1_kick.curriculum")
parameter_search = _sys.modules[__name__ + ".parameter_search"] = _importlib.import_module("rosclaw.evolution.orchestrator.g1_kick.parameter_search")
shot_adapter_train = _sys.modules[__name__ + ".shot_adapter_train"] = _importlib.import_module("rosclaw.evolution.orchestrator.g1_kick.shot_adapter_train")
trajectory_search = _sys.modules[__name__ + ".trajectory_search"] = _importlib.import_module("rosclaw.evolution.orchestrator.g1_kick.trajectory_search")
