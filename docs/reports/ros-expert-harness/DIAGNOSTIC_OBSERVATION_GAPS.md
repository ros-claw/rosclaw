# Diagnostic observation gaps

The ROS Expert Harness must distinguish observed health from missing evidence
before proposing navigation, integration, or repairs on an unfamiliar ROS system.
This change is independent of coverage-planner optimization.

On main `9862058c445c88156b9080832b5dd803c75279ae`, the diagnostic engine could
report HEALTHY despite an incomplete graph, an unobserved declared sensor type
or rate, or a lifecycle node explicitly reporting UNKNOWN. Eleven counterexample
checks failed before the fix; the original test log is retained in the local
implementation evidence.

The engine now includes these gaps in `unknown_checks` for the relevant profile.
A known blocking fault retains BLOCKED precedence. A TF-only check does not
require unrelated sensor/lifecycle observations. The check consumes observations
without changing them, dispatching actions, or granting readiness/physical authority.

Fifteen dedicated checks cover missing evidence, observed health, blocking faults,
profile isolation, and diagnosis/recheck after two sensor topic remappings.
These are deterministic fixture tests, not an unseen-body holdout, live ROS
experiment, autonomous repair, or proof of Memory/Harness causal benefit.

## Overall implementation direction

Per the user's updated priority, preserve the existing cleaning experiments and
negative efficiency results; pause further coverage parameter tuning. The next
milestones concern general ROS engineering:

1. Evidence-grounded discovery and diagnosis across topic/frame/namespace changes.
2. An unfamiliar Body's geometry, TF, sensors and capability binding, with UNKNOWN
   preserved where observations are absent; then guarded point navigation.
3. Matched fault diagnosis and repair/recheck tasks with independently verified
   outcomes, diagnosis/repair time, failed attempts and human interventions.
4. Same-model, isolated Harness and Memory interventions on those tasks.
5. Dynamic-environment tasks and transfer, alongside bounded cleaning regression.

Existing full-coverage and safety requirements are not waived. They remain open
where unaccepted. Each independent PR follows its own measured checks; failure
of the optional efficiency target does not prevent independent diagnostic work.
