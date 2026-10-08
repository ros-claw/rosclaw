# Explicit extended proposal optimization

`rosclaw.growth.extended_proposal_regression` is a task-neutral, proposal-only
opt-in numeric API. It increases the inner optimization budget over one frozen
batch; it does not collect data, operate a simulator, authorize deployment,
open a fresh exam, or access hardware.

Use `ExtendedProposalRegressionConfig(steps=1600)` with
`fit_extended_proposal_residual`. An integer budget of 1–2,560 is allowed. All
other original configuration and input checks remain in force: residual and
learning-rate caps, original AR(1) behavior likelihood, original reset rows,
device/determinism requirements, fixed baseline, frozen guard, and both KL
bounds. The original config still rejects more than 160 steps.

The original solver runs once. The initial behavior reference and KL budget
are not reset in successive blocks. Rejection or convergence can stop before
the requested budget. `completed_optimizer_steps` counts accepted updates,
not all attempts. Increased optimization is not additional independent
episodes or online RL. The returned algorithm is a separate
`EXPLICIT_EXTENDED_PROPOSAL_REGRESSION_V1`, with explicit base numeric algorithm
and requested budget; never reseal it as a legacy trained artifact.

Callers must authenticate their batch and training/held-out context provenance
separately, keep source bindings and original parents, test all physical
retention and capability gates, and independently evaluate fresh conditions
before promotion. A lower regression loss or a bounded KL is not a physical
safety or football-success guarantee.

Synthetic tests compare original/extended numerics at equal budgets, actually
accept 192 updates with one original KL reference, reject stale likelihoods
and malformed/over-limit settings, preserve inputs, and restore Torch RNG,
thread and deterministic settings on solver failure. Synthetic tests are not
physical experiments.
