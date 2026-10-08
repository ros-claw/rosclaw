# Positive sample weighting (proposal only)

`growth.sample_weighting` balances loss mass across caller-defined integer
partitions. It contains no robot, football, simulator, activation or transport
logic. All input rows remain present with positive weights in `[1/16, 16]`
and mean one. Excessive imbalance is rejected, not clipped or silently sampled.

The proposal regression API accepts optional `sample_weights`. Absent weights
preserve previous numerical behavior exactly; explicit unit weights produce
the same optimizer result plus a weighting receipt. Nonuniform weights multiply
the advantage weights and normalize their total mass. Behavior density checks,
both unweighted KL limits, frozen guards and actor caps are unchanged.

The receipt binds the exact supplied weights and numeric row count. It cannot
verify physical collection, assert retention on new trajectories, or approve
runtime execution. Partition labels and experimental rationale must be bound
by the downstream complete-batch manifest. This is a changed training objective,
not a claim that phase balancing is original AWR or guarantees better control.

Motivation from the Soccer application: a complete 160-rollout diagnostic
measured about 76.7% of advantage-weighted loss mass after contact and 7.3%
during the contact phase. Equal phase mass is a testable alternative, not a
demonstrated football gain. Old frozen sources and baseline optimizers remain
untouched.
