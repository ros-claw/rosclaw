"""DEPRECATED package shim (PR-DF-24.3): ``rosclaw.memory.v2.regime`` moved to
``rosclaw.memory.regime``.  The DATA schema name stays ``memory.v2`` — source
layout version ≠ protocol version (DF-16.3).  Modules register into
``sys.modules`` so both paths share ONE module object.
"""

import importlib as _importlib
import sys as _sys

builder = _sys.modules[__name__ + ".builder"] = _importlib.import_module("rosclaw.memory.regime.builder")
cli = _sys.modules[__name__ + ".cli"] = _importlib.import_module("rosclaw.memory.regime.cli")
detector = _sys.modules[__name__ + ".detector"] = _importlib.import_module("rosclaw.memory.regime.detector")
envelope = _sys.modules[__name__ + ".envelope"] = _importlib.import_module("rosclaw.memory.regime.envelope")
explain = _sys.modules[__name__ + ".explain"] = _importlib.import_module("rosclaw.memory.regime.explain")
features = _sys.modules[__name__ + ".features"] = _importlib.import_module("rosclaw.memory.regime.features")
matcher = _sys.modules[__name__ + ".matcher"] = _importlib.import_module("rosclaw.memory.regime.matcher")
models = _sys.modules[__name__ + ".models"] = _importlib.import_module("rosclaw.memory.regime.models")
persistence = _sys.modules[__name__ + ".persistence"] = _importlib.import_module("rosclaw.memory.regime.persistence")
session_envelopes = _sys.modules[__name__ + ".session_envelopes"] = _importlib.import_module("rosclaw.memory.regime.session_envelopes")
session_samples = _sys.modules[__name__ + ".session_samples"] = _importlib.import_module("rosclaw.memory.regime.session_samples")
