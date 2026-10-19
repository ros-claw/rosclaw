# Preserve clock configuration in source-only replicas

The generic Nav2 diagnostic owner has two fixed observation roles and no fault
label input. Previously `prepared_source_stack` rejected a valid staged
`counterfactual_sim` clock configuration unless the capture role was exactly
`rsk_node_clock_mismatch`. Consequently the owner could reproduce its original
TF configuration but could not preserve this original clock configuration.

The launch guard now accepts the existing two source-only observation roles as
well as the original counterfactual clock role. It delegates SDK isolation,
source-observer identity, absence of injected runtime faults, exact capture
fields, run binding and phase checks to the same authoritative unadmitted
capture validator used by the fixture. Read-only replicas additionally require
a full original run ID and the guarded sink bootstrap schema. They remain
restricted to `before_repair`; the existing clock-role rollback remains valid.

The staged clock schema is unchanged: at most two named Nav2 processes, boolean
`use_sim_time` values, no arbitrary parameters, executable or command. Its
configuration is still validated before the context guard. An ordinary
positive/driver/other fault role cannot use this clock override path. Default
Nav2 configuration imports no capture module and keeps SIM time enabled.

When a read-only replica declares clock overrides, capture additionally waits
for actual parameter reads of every declared node in the original namespace.
The values must be strict booleans and their original wall-capture timestamps
must be no more than 500 ms old. Missing, future, stale or mismatched reads do
not become observations merely because the launch configuration requested them.
Without an override, the previous capture behavior is unchanged.

An unadmitted source-only case may have no execution configuration or navigation
policy. Capture preserves that absence. The owner omits Body/policy fields when
no policy file is supplied; an explicitly provided policy is preserved. It does
not manufacture limits or permission just to inspect an unresolved system.

This changes disposable source-only startup compatibility. The owner still
permits one World, zero goals and zero model calls, records the original
configuration and checks its exact policy/recipe binding. The existing capture
path and independent sink review forbid Body admission, leases and commands;
the new guard supplies no execution authority. A missing target, incomplete
readback or failed replica remains unverified, with no retry.

The Core `src` tree, physical geometry, velocity limits and canonical action
path remain unchanged. Component tests and actual original-file review are
separate from model diagnosis, unseen-case evaluation and causal Memory benefit.
