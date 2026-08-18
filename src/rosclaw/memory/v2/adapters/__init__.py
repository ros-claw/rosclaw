"""DEPRECATED package shim (PR-DF-24.3): ``rosclaw.memory.v2.adapters`` moved to
``rosclaw.memory.adapters``.  The DATA schema name stays ``memory.v2`` — source
layout version ≠ protocol version (DF-16.3).  Modules register into
``sys.modules`` so both paths share ONE module object.
"""

import importlib as _importlib
import sys as _sys

base = _sys.modules[__name__ + ".base"] = _importlib.import_module("rosclaw.memory.adapters.base")
registry = _sys.modules[__name__ + ".registry"] = _importlib.import_module("rosclaw.memory.adapters.registry")
rh56_rps = _sys.modules[__name__ + ".rh56_rps"] = _importlib.import_module("rosclaw.memory.adapters.rh56_rps")
