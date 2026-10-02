# Current-parent consolidation

`rosclaw.growth.anchor_consolidation.consolidate_unique` extends the existing
`AnchorOutputMemory` without replacing its contract or changing older model
source hashes. It is a task-neutral numerical training helper, not an executor,
learning algorithm, qualification gate or authorization.

After independently qualifying a parent, the application should reconstruct
all its successful trajectories, including capabilities gained by that parent.
Freezing its parameters alone does not protect these trajectories from the next
plastic residual. An anchor bank left at an earlier generation can leave newly
learned states unprotected.

The caller supplies the full causal frozen-encoder observations and actual
parent predictions. Only exact repeated observations with equal predictions
reuse a row. Conflicts are rejected, not averaged or rounded. Every supplied
sample retains a `sample_to_memory_row` entry; no distinct inherited row is
evicted. Capacity overflow fails closed. The caller seals the complete mapping
and independently verified trajectory identities into its evidence manifest.

Returned memory binds the current parent, encoder, evidence and predecessor
memory hashes. Even an all-duplicate addition produces a new parent/evidence
binding. A valid hash is not proof that the parent was qualified or that its
input data was complete. The helper explicitly reports
`evidence_independently_verified=false`.

Reloaded memory preserves recorded outputs at known states. This is a local
numeric guarantee, not unseen-state retention, natural motion, trajectory
safety or physical execution evidence. Promotion and hardware authorization
remain false. Existing independently recorded failures must be retained.
