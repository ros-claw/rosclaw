# Protected latent readout (experimental training mathematics)

`rosclaw.growth.anchor_plane` is task-neutral and optional: importing the growth
package does not eagerly import NumPy or this helper. It contains no robot,
simulator, policy activation, permit, or registry integration.

For a **frozen** encoder and protected latent matrix H, the helper constructs
P = I - VᵀV using the retained right singular vectors of H. A plastic residual
readout consumes Pφ(s). At a protected latent vector the residual is zero.
Numerical projection norms at or below 1e-10 are explicitly zeroed so a large
readout cannot amplify anchor roundoff. Rank uses a 1e-10 relative threshold.

`fit_protected_readout` fits an advantage-weighted or otherwise caller-weighted
residual regression within this plastic subspace. Targets, weights, and the
serialized plane are **training data, not trustworthy execution evidence**.
All finite arrays and dimensions are checked; plane reload recomputes its rank
and contract. A full-rank anchor bank has no plastic dimensions.

This is inspired by the stability/plasticity motivation of
[Orthogonal Gradient Descent](https://arxiv.org/abs/1910.07104), not a reproduction
of its full nonlinear algorithm. It protects only the declared latent anchors,
not unseen states, changed encoders, trajectories, safety, or task performance.
Physical retention tests and the ordinary Growth safety/promotion gates remain
mandatory. `training_only=true`, `hardware_authorized=false`, and
`promotion_authorized=false` are immutable contract fields.

Tests: `pytest tests/growth/test_anchor_plane.py`; use an isolated
`ROSCLAW_HOME` for surrounding Practice tests instead of migrating user state.
