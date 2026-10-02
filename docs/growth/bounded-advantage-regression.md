# Bounded advantage-weighted residual regression

Proposal-only numeric backend, inspired by [AWR](https://arxiv.org/abs/1910.00177).
The author's [reference implementation](https://github.com/xbpeng/awr) was
reviewed at `831442fb8d4c24bd200667cbc5e458c7657effc2` (`learning/awr_agent.py`,
README, MIT LICENSE). No external code, TensorFlow runtime, policies or weights
are copied or loaded. This is an adaptation, not a full reproduction of AWR.

`bounded_advantage_regression.fit_advantage_residual` fits a bounded residual
MLP using marginal Gaussian action regression with clipped exponential frozen
advantage weights. The caller provides features, frozen baseline, fixed guard,
observed actions, trajectory reset masks, actual behavior log probabilities,
and independently established advantages. No task, body, course ID, simulator,
checkpoint, training collector, executor or activation is part of this backend.

The actual AR-conditioned behavior density is reconstructed before fitting;
correlated exploration is not labeled IID. Both conditional and marginal KL
are independently checked against the unchanged behavior (final limit .005).
The objective is weighted marginal regression, **not** a PPO importance ratio.
CPU float64, full-batch accepted-step history, deterministic configuration,
backtracking and at most 160 Adam steps are explicit. The base and guard are
never updated; Torch is optional/lazy and host CPU RNG/settings are restored.

Weights follow `min(exp(A/temperature), maximum_weight)`, with a lower numerical
floor `exp(-64)` and normalization to mean one. Therefore the *normalized*
maximum may exceed the pre-normalization clip. Advantages and their provenance
are caller-owned; this engine does not manufacture critic labels, assert
bootstrapped TD-lambda learning, or verify a physical batch.

Every receipt states `physical_batch_verified=false`,
`distributional_retention_guaranteed=false`, `promotion_authorized=false` and
`hardware_authorized=false`. A loss reduction, numeric KL bound or exact-zero
guard at known states is not generalization or physical safety evidence.
Applications must qualify full closed-loop outcomes independently and retain
failed candidates. Existing correlated PPO and its artifact-bound source are
unchanged. Numeric tests are synthetic fixtures, not robot skill evidence.
