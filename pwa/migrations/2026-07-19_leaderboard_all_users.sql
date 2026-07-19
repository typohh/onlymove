-- One-time migration: expand `leaderboard` to all users (2026-07-19).
-- egd_pin NOT NULL -> nullable (anonymous rows carry no EGD binding). SQLite
-- cannot relax a column constraint in place, so this rebuilds the table (DROP
-- also drops idx_leaderboard_egd_pin; it is recreated below). Row count is
-- small (registered players only at migration time), so the copy is cheap.
-- There is deliberately NO index on last_active: it would cost an index write
-- on every novel /report (last_active changes each time), while the reads it
-- would serve are full scans of a table the weekly cron prunes to roughly
-- weekly-active size (see the prune note in schema.sql / leaderboard.js).
--
-- Apply to the remote D1 (from pwa/):
--   npx wrangler d1 execute tsumego-reports --remote --file=migrations/2026-07-19_leaderboard_all_users.sql
-- And to the local miniflare D1 for dev (stop `npm run dev` first):
--   sqlite3 .wrangler/state/v3/d1/miniflare-D1DatabaseObject/<hash>.sqlite ".read migrations/2026-07-19_leaderboard_all_users.sql"
-- (A later pipeline.fetch_database run also propagates the remote schema to the
-- local mirror + miniflare automatically.)

CREATE TABLE leaderboard_new (
    user_id            TEXT PRIMARY KEY,
    egd_pin            TEXT,
    egd_name           TEXT,
    country_code       TEXT,
    gor                INTEGER,
    gor_updated_at     TEXT,
    buffer             TEXT,
    pred_egf_rank      REAL,
    pred_egf_se        REAL,
    gor_rank           REAL,
    prev_week_gor_rank REAL,
    week_id            INTEGER,
    this_week_max      REAL,
    prev_week_max      REAL,
    last_active        TEXT
);
INSERT INTO leaderboard_new
    SELECT user_id, egd_pin, egd_name, country_code, gor, gor_updated_at, buffer,
           pred_egf_rank, pred_egf_se, gor_rank, prev_week_gor_rank, week_id,
           this_week_max, prev_week_max, last_active
      FROM leaderboard;
DROP TABLE leaderboard;
ALTER TABLE leaderboard_new RENAME TO leaderboard;
CREATE UNIQUE INDEX IF NOT EXISTS idx_leaderboard_egd_pin ON leaderboard(egd_pin);
-- If a pre-final version of this migration was already applied (dev DBs), drop
-- the since-removed last_active index; a no-op everywhere else.
DROP INDEX IF EXISTS idx_leaderboard_last_active;
