# Registered Native SIM response, verification and parent limits

A capability descriptor owns its existing parent ceiling. Trusted agent configuration may register a shorter response wait and verification period for one explicitly named canonical SIM capability:

```yaml
agent:
  nonreal_action_limits:
    coverage.program_finite_v1:
      response_wait_sec: 310
      parent_timeout_sec: 330
      verification_timeout_sec: 310
```

This configuration is not model input and carries no approval. The configured parent must equal the descriptor ceiling;0<response<=verification<=parent<=330, with response strictly shorter than parent. REAL/SHADOW, malformed/nonfinite limits, changed ceilings and attempts to redefine the legacy coverage.execute_slice contract refuse before remote execution. Legacy coverage_response_wait_sec and default actions retain their previous behavior.

An early response expires as ACTION_RESULT_UNKNOWN with the original action/grant IDs. The same canonical waiter, original parent deadline and action-scoped Session remain owned until original terminal settlement. A replacement action is refused while it is outstanding; channel close revokes it. A terminal receipt or cancel ACK alone does not restore permission or prove physical stop. No retries or alternate Runtime were added.

Historical registration qualification: twelve new registered-bound tests plus existing admission/session/cancellation/receipt/consent checks,78passed4.39s; Ruff affected files clean. Whole SDK collection still fails tests/knowledge/test_contracts.py because optional rosclaw_know.contracts is absent (11264/11324collected,60deselected). Broad agentd attempt was interrupted after158.16s:206passed,4renderer failures caused by unavailable ur5e/canonical renderer dependencies; full agentd is not claimed passing. A full installed-image SDK/actual Native composition and physical common-tool310/330/310 qualification remain required. No model/provider/RobotIO call in these new tests.

## Dispatch and result boundaries

Persisting an intent grants no execution permission. Public finite coverage
rechecks the original source identities, observation ages, Body/epochs, energy,
native terminal state and full-footprint path after persistence. Any registered
budget check runs again using the same parent and deadline before dispatch;
its return is followed by another state/path check. Cleanup retains its original
stop/off observation requirements.

Both estimated and exact-transfer budget adaptation may return only a prefix
of the original unissued work. They cannot shift its entry, add turns or extend
it. Shortening cannot remove a remaining continuous-stroke requirement; paths
below the selector's original 0.10m threshold receive no continuous-stroke credit.

Header ages use original integer nanoseconds and unchanged inclusive limits.
Recognized integer-to-seconds projections may differ in floating-point rounding;
these display values do not change the integer age decision. Missing or
inconsistent original facts still refuse.

A successful SIM query is not successful motion. Explicit unsuccessful terminal
statuses refuse, and unknown or query-only results cannot create a completion
receipt. Existing legacy driver-only payloads retain their limited evidence
scope and never establish physical or REAL success.
