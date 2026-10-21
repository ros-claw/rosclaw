# FC-01: original-case entry NO-GO

Frozen experiment source/main: `5e865c9edb14fc57a88f95d2d532c2ab11626015` after PR #761 source3,
exact-main4 original first workflows and local checks passed. Protocol hash:
`closeoutprotocol_e994f6b63f06f8611890fb1ad9dcc7794786008671f159e672af82c5b0fa6457`.

The planned host-only matrix is seven scenarios × environment seeds 101/202.
It contains 14 planned cells, **zero admitted original case instances, zero
actual model episodes, and 14 NOT_RUN_NO_GO outcomes**. The same 14 D reference
cells are NOT_RUN; no same-input comparison was performed. No original model
event exists. Planned cells are not a success-rate denominator; Wilson intervals
are null because no trials were run. No model failure or correct-abstention rate
is inferred from this entry failure.

Before entry, the existing original-input inventory did not establish P1: a
supported single TF fault with all other necessary original constraints complete.
The extended bounded search found 344 captures, reviewed 326 applicable rows
(325 usable, one preserved unusable capture), and independently recomputed the
usable cases. All 63 TF-violated records also had lifecycle UNKNOWN. No original
source clock, lifecycle state, geometry, requirement or threshold was changed.
This does not claim every arbitrary file outside the reviewed input interface
has been exhausted.

The original-input gate was added before any new trial and normally merged in
PR-A. Missing originals are explicit UNAVAILABLE cells. REPLAY/SYNTHETIC fixtures
cannot become ORIGINAL_CAPTURE by relabelling a PASS. The frozen entry review
stops dispatch before any model call. The four-sentinel runtime stop condition
was therefore never reached; it is not misreported as four budget failures.

The old G5, Source316 pilot and RK09 remain unchanged. This is an honest
**input-prerequisite NO-GO**, not a new empirical model-negative result.
