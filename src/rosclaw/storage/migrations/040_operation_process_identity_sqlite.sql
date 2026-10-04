-- backend: sqlite
-- Missing legacy evidence remains unknown; never reconstructed from a live PID.
ALTER TABLE operations ADD COLUMN process_identity_json TEXT NOT NULL DEFAULT '';
