# P3b common original observation access

P3a's generic A task card did not expose enough original facts for an informed
model decision. Starting a model comparison at that point would confound Expert
interpretation with access to otherwise unavailable input. This change gives
all arms the same immutable common evidence catalog and two typed read-only
operations: `read_task_evidence` and `expand_task_evidence`.

Views cover original task requirements, all observed interfaces and the subset
named in those requirements, navigation parameter fields and all parameters,
signals, transforms, lifecycle, original staging, URDF and offline Body geometry.
Navigation parameter filtering is a common projection of field names, never a
constraint verdict or remedy. Offline geometry is explicitly INFERRED and
OFFLINE_URDF in every arm. Unknown/missing observations are not healthy state.
No gold case IDs or Expert repair candidates are added to common data. B/C keep
their separate Expert case card and C keeps host-admitted historical priors.

Every record binds its original source, JSON pointer, value hash and record
content hash. The common view hash is identical across arms given identical
originals. Original case, model, task-file and compiled-view changes invalidate
reads. Records are returned as independent copies. Pages contain at most 16
records and 2,000 serialized record characters; omitted large values are
explicitly marked with an expandable record ref. Expansion uses bounded,
losslessly reconstructable serialized slices with a SHA256 and next offset.
Counts and next cursors distinguish absence, exhaustion and incomplete pages.
The catalog includes original capture time and SEALED_ORIGINAL_NOT_CURRENT_L1.

These operations do not start ROS/DDS or a World, change staging, infer current
readiness, grant capability/authority or modify the original G5 runner. Their
arguments are typed objects in the facade schema. This closes the missing
access path; it does not prove equal model performance, adequate diagnostic
budgets, Memory benefit or provider schema compatibility. The independent pilot
must still preregister actual input/tool policy, timing and deterministic-only
baseline. All original G5 results remain fixed.
