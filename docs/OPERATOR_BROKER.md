# ROSClaw Operator Broker

The Operator Broker is ROSClaw's trusted consent plane for exact REAL actions. It separates an
Agent's proposal from the decision that may authorize physical execution. The Agent may create and
read its own proposal, but only the `rosclawd` service UID may list decision challenges or accept or
decline a proposal.

## Core flow

1. A robot MCP prepares an immutable REAL `ActionEnvelope` and an `ActionDisplay`.
2. `operator.proposal.create` strips caller approval claims, binds the proposal to the authenticated
   Unix peer UID, Body Snapshot, action-intent hash, daemon generation, nonce, and TTL, and writes
   `OPERATOR_PROPOSAL_CREATED` to the daemon Ledger.
3. A trusted broker process lists pending proposals. This is the only RPC view containing the
   one-time challenge nonce.
4. A decision must match the request ID, nonce, action-intent hash, operator principal, channel, and
   live daemon generation.
5. Acceptance creates the exact daemon Permit, injects it internally, submits the action as the
   originating Agent UID, and supervises its Action Lease until a terminal Receipt. Permit material
   and the decision challenge are never returned through the Agent MCP result.
6. The Receipt authorization decision includes bounded provenance: proposal request ID, operator
   principal, decision channel, decision time, and action-intent hash.

The durable lifecycle is:

```text
CREATED -> PRESENTED -> ACCEPTED -> PERMIT_ISSUED -> SUBMITTED -> TERMINAL
                    \-> DECLINED
                    \-> CANCELLED
                    \-> EXPIRED
                    \-> INVALIDATED
```

Pending proposals are invalidated on daemon restart. Argument mutation, challenge mismatch, intent
hash mismatch, expired deadline, lost Agent Session, unavailable REAL executor, or a failed Ledger
write fails closed before physical dispatch.

## Trusted local reference client

The initial broker client is deliberately small and contains no chat or Agent runtime:

```bash
.venv/bin/python -m rosclaw.entrypoint operator pending --json
.venv/bin/python -m rosclaw.entrypoint operator status PROPOSAL_ID --json
.venv/bin/python -m rosclaw.entrypoint operator decide PROPOSAL_ID \
  --accept \
  --principal-id operator-shift-a \
  --reason "Reviewed the exact action and physical workspace" \
  --json
```

`operator decide` defaults to supervising the accepted action for up to 300 seconds and renews its
lease until a terminal Receipt. Exiting the broker or losing its connection does not turn an
unverified action into success; the daemon lease and orphan policy remain fail closed.

This command must run as the `rosclawd` service UID. In a production deployment, the Agent and
daemon UIDs must differ, the socket may be group-readable for proposal creation, and privileged
RPCs remain daemon-UID-only. Same-UID development validates the protocol but does not prove an
unbypassable consent boundary.

## MCP behavior

When native form elicitation is unavailable, the Interaction SDK now creates a real daemon proposal
and returns its public request ID with `APPROVAL_PENDING`. A missing broker/ledger returns
`APPROVAL_CHANNEL_UNAVAILABLE`; it never falls back to a chat phrase, manual Permit copy, or Agent
self-approval.

The Agent can poll that public lifecycle through the read-only `get_approval_status` MCP tool and
can withdraw a still-pending request through `cancel_approval`. Neither tool exposes the decision
challenge, creates a Permit, or grants Operator decision authority. Cancellation fails once a
trusted decision has begun.

Native MCP form confirmation remains a trusted-host compatibility path. Cross-UID installations
must use the Operator Broker until an authenticated host-decision adapter is configured. Planned
follow-ups are a protected local Web console, URL elicitation callback, SSE events, authenticated
remote channels, and bounded Mission Grants.
