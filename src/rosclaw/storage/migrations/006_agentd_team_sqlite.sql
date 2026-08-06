-- backend: sqlite
-- PR-TF-070/071/072: Team Fabric control plane (ADR-0004).
-- team_members: roster with health states and epoch binding.
-- team_epochs: monotonically increasing epochs; awards/leases bind one.
-- role_leases: conflict_key CAS — one ACTIVE lease per conflict_key/epoch.
-- team_tasks: contract-net announcements/awards with idempotency.
CREATE TABLE IF NOT EXISTS team_members (
    team_id TEXT NOT NULL,
    member_id TEXT NOT NULL,
    card_json TEXT NOT NULL,         -- TeamMemberCardV1
    state TEXT NOT NULL,             -- CANDIDATE|JOINING|READY|SUSPECT|LOST|LEFT
    team_epoch INTEGER NOT NULL DEFAULT 0,
    last_seen_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (team_id, member_id)
);

CREATE TABLE IF NOT EXISTS team_epochs (
    team_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    reason TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (team_id, epoch)
);

CREATE TABLE IF NOT EXISTS role_leases (
    lease_id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL,
    team_epoch INTEGER NOT NULL,
    conflict_key TEXT NOT NULL,
    holder TEXT NOT NULL,
    lease_json TEXT NOT NULL,        -- RoleLeaseV1
    state TEXT NOT NULL,             -- ACTIVE|EXPIRED|REVOKED|CONTESTED
    expires_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- At most one ACTIVE lease per (team, epoch, conflict_key).
CREATE UNIQUE INDEX IF NOT EXISTS ux_role_leases_active
    ON role_leases (team_id, team_epoch, conflict_key) WHERE state = 'ACTIVE';

CREATE TABLE IF NOT EXISTS team_tasks (
    team_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    team_epoch INTEGER NOT NULL,
    announcement_json TEXT NOT NULL,
    status TEXT NOT NULL,            -- ANNOUNCED|AWARDED|ACCEPTED|DONE|FAILED|EXPIRED
    awardee TEXT,
    award_lease_id TEXT,
    bids_json TEXT NOT NULL DEFAULT '[]',
    evidence_json TEXT,
    idempotency_key TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (team_id, task_id),
    UNIQUE (team_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS team_events (
    event_id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL,
    event_type TEXT NOT NULL,        -- rosclaw.team.<entity>.<verb>.v1
    team_epoch INTEGER,
    actor_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    idempotency_key TEXT,
    occurred_at TEXT NOT NULL,
    UNIQUE (team_id, idempotency_key)
);
