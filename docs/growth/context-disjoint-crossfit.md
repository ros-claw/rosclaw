# Context-disjoint terminal critic

`growth.context_crossfit.context_crossfit_advantages` is an explicit offline
numeric alternative to the existing whole-trajectory Monte Carlo critic.
It is not enabled by default and does not change the historical learner.

A caller supplies one authenticated context ID per complete trajectory. All
replicates of a context stay in one of four deterministic held-out folds.
Every phase must have held-out support in every fold, and each training solve
requires at least 50 frames. Missing support is rejected rather than silently
falling back to an overlapping split. Sorted context identities determine the
fold assignment; callers must preregister the context labels, not tune their
numbering after evaluating outcomes.

Fold regressions use raw terminal-return units. Normalizing with held-out
returns before a ridge solve can influence an intercept; therefore return
normalization is applied only to the final advantages. Changing the returns
of a held-out fold leaves that fold's raw predictions exactly unchanged.
`critic_readout` and `crossfit_predictions` are in raw terminal-return units,
not the standardized units of the existing critic. These units must not be
silently substituted in an old learner/checkpoint contract.

The returned fold IDs allow callers to audit context separation. The function
does not authenticate dataset hashes or physical outcomes, change an actor,
collect an episode, load a checkpoint, grant a permit, or promote a candidate.
Those responsibilities remain with the existing evidence and review layers.
Component tests do not demonstrate learned skill or improved generalization.

This is a terminal MC/ridge critic, not bootstrapped TD, recurrent return
redistribution, online actor-critic, or RUDDER. The first application is a
read-only comparison of existing frozen learning signals; any actual training
requires a new declared objective and independent physical retention tests.
