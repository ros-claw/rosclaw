# Same-context terminal credit, without a fitted value critic

`grouped_terminal_credit` is an offline numeric alternative for comparing
declared repeated episodes of the same task context. Each episode is compared
with the mean terminal return of the **other** trials in its context. Its own
return never enters its baseline. Relative differences are standardized over
episodes, then broadcast back to that episode's numeric training rows.

The caller must authenticate the physical batch, common task conditions,
behavior policy, independent random draws and context identity. A context
label alone does not prove any of these properties. Complete terminal returns
must be constant within each episode; singleton contexts, missing episodes,
nonfinite/unbounded values, malformed labels and oversized batches fail.

All declared successes and failures remain represented. A positive advantage
within an entirely failed group merely means comparatively less bad; it does
not certify success. Group-relative credit is not a substitute for physical
task success or safety gates. It does not fit a critic, update an actor, perform
rollouts, expose future returns/context labels as actor observations, implement
online PPO, or authorize deployment/promotion. Changing a learner to use this
credit requires an explicit new objective receipt and independent physical
evaluation; do not relabel historical neural-critic receipts.
