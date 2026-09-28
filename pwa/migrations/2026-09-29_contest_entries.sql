-- Contribution-rewards entries (2026-09-29): the table behind the /contest page.
-- Additive; the table is only read by the new Worker and only written by
-- auxiliary/contest_entries.mjs. Apply BEFORE the deploy (the page reads it):
--   cd pwa && npx wrangler d1 execute tsumego-reports --remote \
--       --file=migrations/2026-09-29_contest_entries.sql
-- See the note beside the CREATE in schema.sql.

CREATE TABLE IF NOT EXISTS contest_entries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    category       TEXT NOT NULL CHECK (category IN ('strength', 'video')),
    user_id        TEXT NOT NULL,
    submitted_at   TEXT NOT NULL,
    video_url      TEXT,
    video_problems INTEGER,
    views          INTEGER,
    quality        REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_contest_entries_account ON contest_entries(category, user_id);
