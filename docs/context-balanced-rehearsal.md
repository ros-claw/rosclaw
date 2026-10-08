# Context-balanced offline rehearsal

`context_balanced_rehearsal_weights` adds a declared rehearsal loss budget to
disjoint whole episodes without changing primary weights or dropping rows.
Each observed rehearsal context gets equal total mass; duplicate episodes do not
gain extra context mass. The rehearsal budget is a ratio of the existing primary
loss mass, not an outcome-derived hyperparameter. Outputs are owned arrays.

The caller authenticates task-specific eligibility and teacher targets. The
module knows no robots, football roles, rewards or control dimensions. It only
constructs supervised loss weights; it does not implement online RL, guarantee
retention, verify physical evidence, or authorize promotion/execution.
All learned candidates still require independent whole-task regression exams.
