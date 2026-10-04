-- backend: sqlite
-- Empty paths identify legacy PIPE operations; never fabricate spool provenance.
ALTER TABLE operations ADD COLUMN output_path TEXT NOT NULL DEFAULT '';
ALTER TABLE operations ADD COLUMN output_checkpoint_json TEXT NOT NULL DEFAULT '';
