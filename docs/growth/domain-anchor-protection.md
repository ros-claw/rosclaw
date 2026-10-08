# Multi-domain frozen anchor protection

`growth.domain_anchor_bank` preserves every declared observation row, domain,
context and source-evidence identity in order. Equal context IDs are allowed
across domains, but duplicate domain IDs and duplicate IDs within one domain
are rejected. No rounding, eviction, selection or duplicate-row deletion occurs.
The combined capacity is 32,768 rows, consistent with `AnchorKernelGuard`.

`DomainAnchorGuard` uses one immutable combined bank during training and
execution. It never selects a weaker gate based on the current domain. A zero
gate preserves the frozen base output at a recorded state when multiplied by
a finite new residual. The bank owns its numeric arrays; exported metadata is
also copied. The frozen parent and encoder are bound by their identities.

This numeric contract does not audit the evidence named by its hashes, prove
all task successes were submitted, or guarantee trajectory retention outside
recorded states. A downstream adapter must enforce its complete success
selection protocol and independently replay the physical records. Source code,
domain order and provenance are part of the bank commitment. All authority and
distributional-retention flags remain false. Existing single-domain contracts
are unchanged.
