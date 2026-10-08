# Frozen anchor kernel guard

`rosclaw.growth.anchor_kernel.AnchorKernelGuard` is task-neutral training
mathematics, not a policy activation or a safety certificate. It complements
`AnchorProtectionPlane`: a plane can lose all plastic dimensions when successful
states span its feature space. The kernel guard retains exact point protection
without removing the entire span.

For a frozen feature vector `x`, frozen protected anchors `a`, and committed
bandwidth `b`, a new finite plastic residual is multiplied by:

```text
g(x) = 1 - exp(-min_a ||x-a||² / (2b²))
g(x) = 0 exactly when the nearest distance is <= 1e-10
```

The parent policy, encoder, metric, bank and bandwidth must remain frozen. No
future outcomes may enter runtime features. The multiplier is continuous and
lies in `[0,1]`; it attenuates updates near old states but does not guarantee
retention on nearby or unseen trajectories. Actual paired physics, safety,
reload and independent fresh evaluation remain mandatory.

Artifacts reject modified authority fields even when resealed. Inputs are
bounded finite matrices, at most 32,768 anchors and 512 dimensions. Exhausting
the bank is an explicit capacity limit, not permission to silently drop old
successful states.

The full reference path uses NumPy. Optional SciPy acceleration uses an exact
Euclidean nearest-neighbor query, then recomputes the distance from coordinate
differences to avoid cancellation. See the
[SciPy query contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html).
Acceleration changes no protection or authorization semantics.

This design borrows the separation of frozen knowledge from new capacity in
[Progressive Neural Networks](https://arxiv.org/abs/1606.04671), but is not an
implementation of that architecture and does not inherit its experimental
claims. No robot morphology, sport, role, actuator or physics backend appears
in this Core utility. Domain-specific feature engineering and learning remain
downstream.
