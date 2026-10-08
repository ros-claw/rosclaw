# Offline event credit partitions

`growth.event_credit_partitions` labels every ordered training row as approach,
event-leading window (including the event frame), early-after, late-after, or
explicitly no-event. Events are measured labels supplied once per trajectory;
`-1` means no event. Consecutive recorded frames and complete group identities
are validated. There is no row filtering, clipping or fabricated event time.

These labels can feed the existing positive `balanced_partition_weights`
objective. All samples retain positive mass, and excess imbalance is rejected
by that contract rather than repaired by dropping failures. The actor need not
receive any of these labels or future event times. A downstream adapter must
bind the measured times to independently audited training evidence and keep
them out of live policy observations. Frame units must match the recorded
decision clock, not a guessed simulator time base.

This is a temporal loss partition, not RUDDER, a causal identification result,
a reward change, a learned event predictor, or a guarantee of improved control.
Missing events and events outside the recorded interval keep all their rows.
The utility grants no physical verification, runtime execution or promotion
authority. Existing optimizer defaults and actor contracts remain unchanged.
