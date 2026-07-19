-- One-time migration: expand `leaderboard` to all users + split registration
-- out into `external_ids` (2026-07-19).
--
-- Before: leaderboard is registered-only and carries the EGD binding inline
-- (egd_pin NOT NULL, egd_name, country_code, gor, gor_updated_at, gor_rank).
-- After:
--   * external_ids(user_id, egd_pin) holds the binding — identity only, general
--     name so future services (aga_pid, ...) add columns here.
--   * The current GoR of each registered player is seeded as a machine 'EGD'
--     row in rank_surveys (rank = raw GoR digits, created_at = the old
--     gor_updated_at so this week's freshness carries over, country = the EGD
--     federation code). Display name/country come from the claimed invitation
--     row from now on.
--   * leaderboard keeps only state rebuildable from tsumego_reports, one row
--     per user, anonymous rows lazily created on /report and idle rows pruned
--     weekly. The week-peak fields (prev_week_gor_rank, week_id, this_week_max,
--     prev_week_max) are DROPPED outright — the this-week/last-week peak views
--     were removed (2026-07-19), so nothing reads them.
-- There is deliberately NO index on last_active (would cost an index write per
-- novel /report; the weekly prune keeps full scans cheap — see schema.sql).
--
-- Deploy the matching Worker in the same session as this migration: the new
-- code queries external_ids, and the old code writes columns this drops.
--
-- Apply to the remote D1 (from pwa/):
--   npx wrangler d1 execute tsumego-reports --remote --file=migrations/2026-07-19_leaderboard_all_users.sql
-- And to the local miniflare D1 for dev (stop `npm run dev` first):
--   sqlite3 .wrangler/state/v3/d1/miniflare-D1DatabaseObject/<hash>.sqlite ".read migrations/2026-07-19_leaderboard_all_users.sql"
-- (A later pipeline.fetch_database run also propagates the remote schema to the
-- local mirror + miniflare automatically.)

CREATE TABLE IF NOT EXISTS external_ids (
    user_id TEXT PRIMARY KEY,
    egd_pin TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_external_ids_egd_pin ON external_ids(egd_pin);
INSERT OR IGNORE INTO external_ids (user_id, egd_pin)
    SELECT user_id, egd_pin FROM leaderboard WHERE egd_pin IS NOT NULL;

-- Seed the GoR heartbeat rows (guarded so a re-run can't duplicate them).
INSERT INTO rank_surveys (user_id, server, rank, created_at, country)
    SELECT l.user_id, 'EGD', CAST(l.gor AS TEXT),
           COALESCE(l.gor_updated_at, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
           l.country_code
      FROM leaderboard l
     WHERE l.egd_pin IS NOT NULL AND l.gor IS NOT NULL
       AND NOT EXISTS (SELECT 1 FROM rank_surveys s
                        WHERE s.user_id = l.user_id AND s.server = 'EGD');

CREATE TABLE leaderboard_new (
    user_id            TEXT PRIMARY KEY,
    buffer             TEXT,
    pred_egf_rank      REAL,
    pred_egf_se        REAL,
    last_active        TEXT
);
INSERT INTO leaderboard_new
    SELECT user_id, buffer, pred_egf_rank, pred_egf_se, last_active
      FROM leaderboard;
DROP TABLE leaderboard;
ALTER TABLE leaderboard_new RENAME TO leaderboard;
