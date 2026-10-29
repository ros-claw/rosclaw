# Native idle preconditions and new proposals

A canonical SIM Dock receipt may report BLOCKED with no native goal dispatched, original native terminal known, independently observed stopped/off state, and an invisible beacon. This remains verified=false. Native exposes this precise negative result as ACTION_PRECONDITION_NOT_MET, rather than treating it as an executed goal failure.

The model must re-observe and change the prerequisite before proposing a new intent. The old idempotency key retains its original blocked result. No action is automatically retried; admission, source readiness, Body, grants and original deadlines apply to every new proposal. Actual native failures, unknown stop/terminal proof, changed receipt identities, and REAL evidence retain the existing fail-closed classification and repeated-failure guard.

Freshness diagnostics preserve the original rejection and report original receive age, capture-to-validation delay and signed native header age separately. Diagnostic fields do not extend any freshness window.
