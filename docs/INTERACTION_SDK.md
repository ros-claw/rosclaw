# ROSClaw Interaction SDK

`rosclaw.interaction` is the host-owned interaction layer for guarded physical actions.
Robot MCP servers provide an exact action intent and an `rosclaw.action-display.v2` card;
the SDK detects negotiated MCP capabilities, presents native confirmation, derives the
connection identity, and hands the accepted proposal to `rosclawd` without exposing permits,
sessions, daemon arming, or action-intent hashes to the Agent.

The first release supports native MCP form elicitation and reports structured
`APPROVAL_PENDING` or `APPROVAL_CHANNEL_UNAVAILABLE` results for negotiated asynchronous/URL
or absent channels. URL completion remains an adapter extension point until an operator-console
callback is configured. Cancellation is propagated to the backend when a confirmed request is
cancelled during submission; progress notifications are advisory.

`request_action` remains the SHADOW compatibility entry point. A REAL call returns
`INTERACTION_REQUIRED` and identifies `request_guarded_action` as its replacement. The guarded
tool's public schema has no `principal_id`, `approval_id`, permit, session, or arm parameter.

The SDK does not weaken the daemon boundary. `rosclawd` still owns session creation, arming,
permit issuance, action dispatch, verification, and the canonical receipt.
