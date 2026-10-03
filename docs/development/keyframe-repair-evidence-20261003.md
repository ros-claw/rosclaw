# Named keyframe repair evidence

R03's historical oracle checked the repaired `home` keyframe independently,
but accepted any replayed rollout from that model. A rollout initialized from
the default `qpos0` was therefore incorrectly accepted as evidence for the
named keyframe repair. A real generic sphere/plane fixture reproduced that
false verification before this change. Historical benchmark records remain
untouched.

The revised task permits only a patch to `home.key_qpos`. Compiled body,
inertia, contact, gravity, solver, timestep/integrator and all other keyframe
fields/names remain fixed. Serialization's explicit `size.nkey` is normalized
only while comparing the two compiled models with their keyframes removed;
actual key count, names and other initializer arrays are checked separately.

Verification now binds a native saved rollout to its model-bound `simkey`
initializer and FULL_INTEGRATION snapshot. An independent native
`mj_resetDataKeyframe`/`mj_forward`/`mj_getState` checks the complete initial
vector, its digest, the named initializer fields, and trace initialization.
At least 0.5 seconds of actual recorded settling is required, with valid
per-step runtime checks, stable final velocity/drift, and an exact persisted
RAW_EXACT report paired with a successful native-tool transcript response.
Default-state rollouts, forged initializer references, altered state vectors,
missing telemetry/replay and body/solver modifications cannot qualify.

The oracle no longer generates settling rollouts while grading R03. It checks
initial contacts using forward evaluation, then reads the native saved
settling trajectory. `reset_check.settle_evidence=NOT_RUN_BY_ORACLE` and
`physics_steps_by_oracle=0` distinguish the static check from the actual
rollout evidence. Test fixture transcripts are explicitly synthetic and are
not claimed to be live Kimi sessions. A fresh Kimi trial is needed under this
new task contract.
