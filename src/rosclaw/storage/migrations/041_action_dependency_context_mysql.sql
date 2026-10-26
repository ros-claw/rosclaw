-- backend: mysql
-- MySQL/SeekDB equivalent of SQLite migration 040; TEXT literal defaults are not portable.
ALTER TABLE pi_context_leases ADD COLUMN dependency_hashes_json VARCHAR(16384) NOT NULL DEFAULT '{}';
