"""Synthetic observation fixtures, generated outside the agent workspace.

Publish measured trajectories, never the producer model or its parameter values.
The A leg gets the same raw observations; only B gets native dataset objects.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


@lru_cache(maxsize=2)
def sysid_observations(static: bool = False) -> tuple[str, dict[str, Any], list]:
    from benchmarks.harnessbench.tasks_v2 import S01_MODEL, S01_TRUE_DAMPING
    from rosclaw.sim.backends.mujoco.backend import MujocoBackend

    with TemporaryDirectory(prefix="harnessbench-observation-producer-") as private:
        backend = MujocoBackend(Path(private))
        truth = backend.load_model_xml(
            S01_MODEL.replace('damping="0.01"', f'damping="{S01_TRUE_DAMPING}"'),
            source={"kind": "synthetic_observation_producer", "fixture": "sysid-v1"},
        )
        dataset_ref = backend.record_dataset(
            truth.model_ref,
            sequences=[
                {"controller": {"hold": True}, "duration_s": 1.0, "qpos0": [q]}
                for q in ([0.0, 0.0, 0.0] if static else [0.6, -0.4, 0.9])
            ],
        )
        dataset = backend.store.get(dataset_ref)
        objects = [("traces", dataset)]
        sequences = []
        for sequence in dataset["sequences"]:
            trace = backend.store.get(sequence["trace_ref"])
            initial = backend.store.get(sequence["initial_state_ref"])
            blob = backend.store.get(initial["state_vector_ref"])
            objects.extend([("traces", trace), ("states", initial), ("states", blob)])
            sequences.append(
                {
                    "qpos0": sequence["qpos0"],
                    "duration_s": sequence["duration_s"],
                    "controller": sequence["controller"],
                    "states": trace["states"],
                    "trace_ref": sequence["trace_ref"],
                }
            )
        observations = {
            "schema_version": "rosclaw.harnessbench.observations.v1",
            "trust_level": "SIMULATED",
            "dataset_ref": dataset_ref,
            "note": "Synthetic measured trajectories; producer model is withheld. "
            "Use these observations, not data generated from the calibration base model.",
            "sequences": sequences,
        }
        return dataset_ref, observations, objects


def stage_sysid_observations(work: Path, task_id: str, *, leg: str) -> None:
    from rosclaw.sim.store import SimStore

    _, observations, objects = sysid_observations(static=task_id == "S02")
    folder = work / "observations"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sysid.json").write_text(json.dumps(observations), encoding="utf-8")
    if leg == "B":
        store = SimStore(work)
        for partition, payload in objects:
            store.put(partition, payload)
