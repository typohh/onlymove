-- SQL schema for the tsumego reports database

CREATE TABLE IF NOT EXISTS tsumego_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    problem TEXT NOT NULL,  -- This is now the problem ID (filename without extension)
    success BOOLEAN NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, problem)
);

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_user_id ON tsumego_reports(user_id);
CREATE INDEX IF NOT EXISTS idx_problem ON tsumego_reports(problem);
CREATE INDEX IF NOT EXISTS idx_success ON tsumego_reports(success);
CREATE INDEX IF NOT EXISTS idx_created_at ON tsumego_reports(created_at);