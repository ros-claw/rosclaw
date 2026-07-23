"""Deterministic level-A samplers for constrained scenario distributions."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections.abc import Callable, Mapping
from typing import Any

from rosclaw.simforge.models import (
    DistributionKind,
    Partition,
    SamplingStrategy,
    ScenarioDistributionSpec,
    ScenarioSample,
    ScenarioVariable,
)

ConstraintEvaluator = Callable[[Mapping[str, Any], Mapping[str, Any]], bool]


class ScenarioSampler:
    """Sample without evaluating arbitrary expressions from benchmark files."""

    def __init__(
        self,
        spec: ScenarioDistributionSpec,
        *,
        constraint_registry: Mapping[str, ConstraintEvaluator] | None = None,
    ) -> None:
        self.spec = spec
        self._constraints: dict[str, ConstraintEvaluator] = dict(_builtin_constraints())
        if constraint_registry:
            self._constraints.update(constraint_registry)
        unknown = sorted(
            constraint.name
            for constraint in spec.constraints
            if constraint.name not in self._constraints
        )
        if unknown:
            raise ValueError(f"unregistered scenario constraints: {', '.join(unknown)}")

    def sample(
        self,
        *,
        count: int,
        seed: int,
        partition: Partition,
        strategy: SamplingStrategy = SamplingStrategy.RANDOM,
    ) -> tuple[ScenarioSample, ...]:
        if count < 1 or count > 1_000_000:
            raise ValueError("sample count must be in [1, 1000000]")
        rng = random.Random(seed)
        raw = self._candidate_stream(count=count, rng=rng, strategy=strategy)
        accepted: list[ScenarioSample] = []
        attempts = 0
        for values in raw:
            attempts += 1
            if attempts > self.spec.max_sampling_attempts:
                break
            if not self._satisfies(values):
                continue
            sample_seed = _derive_sample_seed(seed, partition, len(accepted))
            accepted.append(self._make_sample(values, partition, sample_seed, strategy))
            if len(accepted) == count:
                return tuple(accepted)

        while len(accepted) < count and attempts < self.spec.max_sampling_attempts:
            attempts += 1
            values = self._random_values(rng)
            if self._satisfies(values):
                sample_seed = _derive_sample_seed(seed, partition, len(accepted))
                accepted.append(self._make_sample(values, partition, sample_seed, strategy))
        if len(accepted) != count:
            raise RuntimeError(
                f"constraint sampling exhausted after {attempts} attempts: "
                f"accepted {len(accepted)}/{count}"
            )
        return tuple(accepted)

    def _candidate_stream(
        self, *, count: int, rng: random.Random, strategy: SamplingStrategy
    ) -> list[dict[str, Any]]:
        if strategy is SamplingStrategy.LATIN_HYPERCUBE:
            return self._latin_hypercube(count, rng)
        if strategy is SamplingStrategy.BOUNDARY:
            return self._boundary_values(count, rng)
        if strategy is SamplingStrategy.PAIRWISE:
            return self._pairwise_values(count, rng)
        return [self._random_values(rng) for _ in range(count)]

    def _random_values(self, rng: random.Random) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, variable in self.spec.variables:
            if variable.distribution is DistributionKind.UNIFORM:
                assert variable.minimum is not None and variable.maximum is not None
                result[name] = rng.uniform(variable.minimum, variable.maximum)
            else:
                result[name] = rng.choice(variable.values)
        return result

    def _latin_hypercube(self, count: int, rng: random.Random) -> list[dict[str, Any]]:
        rows = [{} for _ in range(count)]
        for name, variable in self.spec.variables:
            if variable.distribution is DistributionKind.UNIFORM:
                assert variable.minimum is not None and variable.maximum is not None
                slots = list(range(count))
                rng.shuffle(slots)
                width = variable.maximum - variable.minimum
                for row, slot in zip(rows, slots, strict=True):
                    row[name] = variable.minimum + width * ((slot + rng.random()) / count)
            else:
                values = list(variable.values)
                rng.shuffle(values)
                for index, row in enumerate(rows):
                    row[name] = values[index % len(values)]
        return rows

    def _boundary_values(self, count: int, rng: random.Random) -> list[dict[str, Any]]:
        levels = {name: _levels(variable) for name, variable in self.spec.variables}
        names = list(levels)
        rows: list[dict[str, Any]] = []
        if math.prod(len(levels[name]) for name in names) <= 4096:
            combinations = itertools.product(*(levels[name] for name in names))
            for combination in combinations:
                rows.append(dict(zip(names, combination, strict=True)))
                if len(rows) == count:
                    return rows
        while len(rows) < count:
            rows.append({name: rng.choice(levels[name]) for name in names})
        return rows

    def _pairwise_values(self, count: int, rng: random.Random) -> list[dict[str, Any]]:
        levels = {name: _levels(variable) for name, variable in self.spec.variables}
        names = list(levels)
        if len(names) < 2:
            return self._boundary_values(count, rng)
        rows: list[dict[str, Any]] = []
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                for left_value, right_value in itertools.product(levels[left], levels[right]):
                    row = {name: rng.choice(levels[name]) for name in names}
                    row[left] = left_value
                    row[right] = right_value
                    rows.append(row)
                    if len(rows) == count:
                        return rows
        while len(rows) < count:
            rows.append({name: rng.choice(levels[name]) for name in names})
        return rows

    def _satisfies(self, values: Mapping[str, Any]) -> bool:
        for constraint in self.spec.constraints:
            parameters = dict(constraint.parameters)
            try:
                accepted = self._constraints[constraint.name](values, parameters)
            except (KeyError, TypeError, ValueError, ArithmeticError):
                return False
            if accepted is not True:
                return False
        return True

    def _make_sample(
        self,
        values: dict[str, Any],
        partition: Partition,
        seed: int,
        strategy: SamplingStrategy,
    ) -> ScenarioSample:
        canonical = json.dumps(
            {
                "distribution": self.spec.digest,
                "partition": partition.value,
                "seed": seed,
                "values": values,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        scenario_id = "scenario_" + hashlib.sha256(canonical.encode()).hexdigest()[:24]
        return ScenarioSample(
            scenario_id=scenario_id,
            partition=partition,
            seed=seed,
            values=tuple(sorted(values.items())),
            distribution_hash=self.spec.digest,
            strategy=strategy,
        )


def _levels(variable: ScenarioVariable) -> tuple[Any, ...]:
    if variable.distribution is DistributionKind.UNIFORM:
        assert variable.minimum is not None and variable.maximum is not None
        midpoint = (variable.minimum + variable.maximum) / 2.0
        return (variable.minimum, midpoint, variable.maximum)
    return variable.values


def _derive_sample_seed(root_seed: int, partition: Partition, index: int) -> int:
    payload = f"rosclaw.simforge.seed.v1\0{root_seed}\0{partition.value}\0{index}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & 0x7FFF_FFFF


def _builtin_constraints() -> dict[str, ConstraintEvaluator]:
    def finite(values: Mapping[str, Any], _parameters: Mapping[str, Any]) -> bool:
        return all(
            not isinstance(value, float) or math.isfinite(value) for value in values.values()
        )

    def target_reachable(values: Mapping[str, Any], parameters: Mapping[str, Any]) -> bool:
        x = float(values[parameters.get("x", "target_x")])
        y = float(values[parameters.get("y", "target_y")])
        z = float(values[parameters.get("z", "target_z")])
        minimum = float(parameters.get("minimum_m", 0.15))
        maximum = float(parameters.get("maximum_m", 1.25))
        radius = math.sqrt(x * x + y * y + z * z)
        return minimum <= radius <= maximum

    def target_not_inside_obstacle(
        values: Mapping[str, Any], parameters: Mapping[str, Any]
    ) -> bool:
        tx = float(values[parameters.get("target_x", "target_x")])
        ty = float(values[parameters.get("target_y", "target_y")])
        tz = float(values[parameters.get("target_z", "target_z")])
        ox = float(values[parameters.get("obstacle_x", "obstacle_x")])
        oy = float(values.get(str(parameters.get("obstacle_y", "obstacle_y")), 0.5))
        height = float(values[parameters.get("obstacle_height", "obstacle_height")])
        half_x = float(parameters.get("half_x", 0.08))
        half_y = float(parameters.get("half_y", 0.08))
        return not (abs(tx - ox) <= half_x and abs(ty - oy) <= half_y and 0 <= tz <= height)

    def start_goal_distance(values: Mapping[str, Any], parameters: Mapping[str, Any]) -> bool:
        threshold = float(parameters.get("minimum_m", 0.15))
        if "start_goal_distance_m" in values:
            return float(values["start_goal_distance_m"]) >= threshold
        dx = float(values[parameters.get("goal_x", "target_x")]) - float(
            values.get(str(parameters.get("start_x", "start_x")), 0.0)
        )
        dy = float(values[parameters.get("goal_y", "target_y")]) - float(
            values.get(str(parameters.get("start_y", "start_y")), 0.0)
        )
        dz = float(values[parameters.get("goal_z", "target_z")]) - float(
            values.get(str(parameters.get("start_z", "start_z")), 0.0)
        )
        return math.sqrt(dx * dx + dy * dy + dz * dz) >= threshold

    return {
        "finite_values": finite,
        "initial_state_collision_free": lambda values, _parameters: (
            values.get("initial_state_collision_free") is True
        ),
        "target_not_inside_obstacle": target_not_inside_obstacle,
        "target_reachable": target_reachable,
        "start_goal_distance": start_goal_distance,
    }


__all__ = ["ConstraintEvaluator", "ScenarioSampler"]
