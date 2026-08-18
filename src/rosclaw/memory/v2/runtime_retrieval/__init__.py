"""DEPRECATED package shim (PR-DF-24.3): ``rosclaw.memory.v2.runtime_retrieval`` moved to
``rosclaw.memory.runtime_retrieval``.  The DATA schema name stays ``memory.v2`` — source
layout version ≠ protocol version (DF-16.3).  Modules register into
``sys.modules`` so both paths share ONE module object.
"""

import importlib as _importlib
import sys as _sys

active_resolver = _sys.modules[__name__ + ".active_resolver"] = _importlib.import_module("rosclaw.memory.runtime_retrieval.active_resolver")
facade = _sys.modules[__name__ + ".facade"] = _importlib.import_module("rosclaw.memory.runtime_retrieval.facade")
fallback = _sys.modules[__name__ + ".fallback"] = _importlib.import_module("rosclaw.memory.runtime_retrieval.fallback")
health = _sys.modules[__name__ + ".health"] = _importlib.import_module("rosclaw.memory.runtime_retrieval.health")
native_retriever = _sys.modules[__name__ + ".native_retriever"] = _importlib.import_module("rosclaw.memory.runtime_retrieval.native_retriever")
provider_resolver = _sys.modules[__name__ + ".provider_resolver"] = _importlib.import_module("rosclaw.memory.runtime_retrieval.provider_resolver")
result = _sys.modules[__name__ + ".result"] = _importlib.import_module("rosclaw.memory.runtime_retrieval.result")
