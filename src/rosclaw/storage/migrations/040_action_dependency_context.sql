-- backend: sqlite
-- Preserve complete audit hash separately from capability-scoped admission evidence.
-- Empty maps keep legacy leases fail-closed under their original full hash.
ALTER TABLE pi_context_leases ADD COLUMN dependency_hashes_json TEXT NOT NULL DEFAULT '{}';
