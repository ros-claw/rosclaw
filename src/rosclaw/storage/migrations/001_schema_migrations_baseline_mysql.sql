-- backend: mysql
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(64) PRIMARY KEY,
    applied_at DOUBLE NOT NULL,
    checksum VARCHAR(64) NULL
) DEFAULT CHARACTER SET utf8mb4;
