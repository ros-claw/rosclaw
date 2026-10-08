"""Lazy fixture policy entry; legacy passive processes need only ROS/stdlib.

Generic fixtures require the full compiled Body source loader and its declared
host dependencies. Missing generic policy never falls back to a known profile.
"""

import json
from pathlib import Path


def load_frozen_sim_runtime_policy(root):
    root = Path(root)
    generic = (root / "sim_runtime_policy.json").exists() or (
        root / "generic_execution_proposal.json"
    ).exists()
    config = root / "execution_config.json"
    if config.exists():
        with config.open("rb") as stream:
            raw = stream.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("bounded fixture execution config required")
        document = json.loads(raw)
        if type(document) is not dict:
            raise ValueError("typed fixture execution config required")
        generic |= "generic_execution_proposal" in document
    if not generic:
        return None
    if not (root / "sim_runtime_policy.json").is_file():
        raise ValueError("generic fixture requires its frozen runtime policy")
    from rosclaw.connectors.ros.context.sim_runtime_policy import (
        load_frozen_sim_runtime_policy as reopen,
    )

    return reopen(root)
