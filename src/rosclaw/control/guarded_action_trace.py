"""Dimension-independent evidence of proposed, guarded and applied actions.

Pure numeric, read-only with respect to any environment. No action execution,
robot driver, authorization or policy optimization is provided here. Missing
proposals have an explicit mask and must not be treated as zero-action labels.
"""

import re

import numpy as np
from numpy.typing import DTypeLike, NDArray


class GuardedActionTrace:
    def __init__(
        self,
        *,
        dimension: int,
        maximum_steps: int,
        context_hash: str,
        post_intervention_guard: bool = False,
    ) -> None:
        if (
            type(dimension) is not int
            or type(maximum_steps) is not int
            or not 1 <= dimension <= 4096
            or maximum_steps < 1
            or dimension * maximum_steps > 8_000_000
            or not isinstance(context_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", context_hash) is None
            or type(post_intervention_guard) is not bool
        ):
            raise ValueError("bounded dimension, horizon and explicit context commitment required")
        self.dimension = dimension
        self.maximum_steps = maximum_steps
        self.context_hash = context_hash
        self.post_intervention_guard = post_intervention_guard
        self._records: list[tuple[object, ...]] = []

    @property
    def count(self) -> int:
        return len(self._records)

    def _vector(self, value: NDArray[np.float64]) -> NDArray[np.float64]:
        array = np.asarray(value)
        if (
            array.shape != (self.dimension,)
            or array.dtype.kind not in "fi"
            or not np.isfinite(array).all()
        ):
            raise ValueError("aligned finite numeric action vector required")
        return array.astype(np.float64, copy=True)

    def record(
        self,
        *,
        step: int,
        proposal: NDArray[np.float64] | None,
        guarded: NDArray[np.float64],
        applied: NDArray[np.float64],
        parent: NDArray[np.float64],
        used_parent: bool,
        reason: str | None,
        intervention_fraction: float,
        post_guarded: NDArray[np.float64] | None = None,
        post_guard_used_parent: bool | None = None,
        post_guard_reason: str | None = None,
    ) -> None:
        """Append one completed step, retaining rejected proposals as proposals."""
        if (
            type(step) is not int
            or step != self.count
            or step >= self.maximum_steps
            or type(used_parent) is not bool
            or not np.isfinite(intervention_fraction)
            or not 0 <= intervention_fraction <= 1
            or (used_parent and (not isinstance(reason, str) or not reason or len(reason) > 160))
            or (not used_parent and reason is not None)
            or (proposal is None and not used_parent)
        ):
            raise ValueError(
                "consecutive bounded completed decision and explicit fallback required"
            )
        g, a, p = self._vector(guarded), self._vector(applied), self._vector(parent)
        if used_parent and not np.array_equal(g, p):
            raise ValueError("parent fallback must equal the declared bounded parent command")
        raw = self._vector(proposal) if proposal is not None else np.zeros(self.dimension)
        expected = (
            p
            if intervention_fraction == 0
            else g
            if intervention_fraction == 1
            else p + intervention_fraction * (g - p)
        )
        extra: tuple[object, ...] = ()
        if self.post_intervention_guard:
            if (
                not 0 < intervention_fraction < 1
                or post_guarded is None
                or type(post_guard_used_parent) is not bool
                or (
                    post_guard_used_parent
                    and (
                        not isinstance(post_guard_reason, str)
                        or not post_guard_reason
                        or len(post_guard_reason) > 160
                    )
                )
                or (not post_guard_used_parent and post_guard_reason is not None)
            ):
                raise ValueError("explicit completed post-intervention guard decision required")
            final = self._vector(post_guarded)
            if post_guard_used_parent and not np.array_equal(final, p):
                raise ValueError("post-intervention parent fallback must equal actual parent")
            extra = (expected.copy(), final, bool(post_guard_used_parent), post_guard_reason or "")
            expected = final
        elif (
            post_guarded is not None
            or post_guard_used_parent is not None
            or post_guard_reason is not None
        ):
            raise ValueError("post-intervention evidence requires declared trace mode")
        if not np.array_equal(a, expected):
            raise ValueError("actual action differs from the declared guarded intervention")
        self._records.append(
            (
                raw,
                proposal is not None,
                g,
                a,
                p,
                used_parent,
                reason or "",
                float(intervention_fraction),
            )
            + extra
        )

    def arrays(self) -> dict[str, np.ndarray]:
        """Return owned arrays. A placeholder proposal is never valid without its mask."""
        if not self._records:
            raise ValueError("empty guarded action trace")
        names: tuple[str, ...] = (
            "actor_proposals",
            "actor_proposal_valid",
            "guarded_actions",
            "applied_actions",
            "parent_actions",
            "used_parent",
            "fallback_reasons",
            "intervention_fractions",
        )
        dtypes: tuple[DTypeLike, ...] = (
            np.float64,
            np.bool_,
            np.float64,
            np.float64,
            np.float64,
            np.bool_,
            "U160",
            np.float64,
        )
        if self.post_intervention_guard:
            names += (
                "post_intervention_inputs",
                "post_guarded_actions",
                "post_guard_used_parent",
                "post_guard_reasons",
            )
            dtypes += (np.float64, np.float64, np.bool_, "U160")
        return {
            name: np.asarray([row[i] for row in self._records], dtype=dtype)
            for i, (name, dtype) in enumerate(zip(names, dtypes, strict=True))
        }
