CREATE TABLE IF NOT EXISTS tsumego_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    problem TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    created_at TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 9,
    UNIQUE(user_id, problem)
);

-- For an existing deployment, run once against the live D1:
--   ALTER TABLE tsumego_reports ADD COLUMN size INTEGER NOT NULL DEFAULT 9;
-- (Not idempotent in SQLite, so it's not part of the schema script itself.)