# ROSClaw Interaction SDK

`rosclaw.interaction` is the host-owned interaction layer for guarded physical actions.
Robot MCP servers provide an exact action intent and an `rosclaw.action-display.v2` card;
the SDK detects negotiated MCP capabilities, presents native confirmation, derives the
connection identity, and hands the accepted proposal to `rosclawd` without exposing permits,
sessions, daemon arming, or action-intent hashes to the Agent.

The SDK supports native MCP form elicitation and daemon-owned pending proposals. When form
elicitation is unavailable, it creates an exact Operator Broker proposal and returns structured
`APPROVAL_PENDING`; if the broker or its durable Ledger is unavailable it returns
`APPROVAL_CHANNEL_UNAVAILABLE`. URL rendering remains an adapter extension point until the protected
operator-console callback is configured. The Agent can query or withdraw a pending proposal through
`get_approval_status` and `cancel_approval`; neither grants decision authority. Cancellation is
also propagated to the backend when a confirmed request is cancelled during submission; progress
notifications are advisory.

`request_action` remains the SHADOW compatibility entry point. A REAL call returns
`INTERACTION_REQUIRED` and identifies `request_guarded_action` as its replacement. The guarded
tool's public schema has no `principal_id`, `approval_id`, permit, session, or arm parameter.

The SDK does not weaken the daemon boundary. `rosclawd` still owns session creation, arming,
permit issuance, action dispatch, verification, and the canonical receipt.

See [OPERATOR_BROKER.md](OPERATOR_BROKER.md) for the cross-UID consent flow. Agent-facing results
never contain the one-time decision challenge, authorized action, or Permit ID.
