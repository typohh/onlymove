-- Monthly historical leaderboards (2026-09-21).
--
-- Two changes, both additive — no existing row is rewritten and no column is
-- dropped, so the OLD Worker keeps serving unchanged after the migration and the
-- order is simply migration FIRST, deploy SECOND. That direction is required:
-- the new Worker's /report path SELECTs `best_rank` off `leaderboard`, so a
-- deploy before the migration would have every leaderboard update fail on `no
-- such column` (swallowed by the ctx.waitUntil catch, but the rank would go
-- stale until the weekly reconcile). The old Worker never names the new columns,
-- so running the migration under it costs nothing.
--
--   1. `leaderboard` gains the current-month PEAK tracker: `best_rank` (the max
--      UNTIMED internal rank an observed attempt has reached this month),
--      `best_rank_timed` (that same observation's TIME-ADJUSTED internal rank),
--      and `best_rank_month` (the UTC month those belong to). They start NULL and
--      fill in on each user's next novel /report; a rank observed in a new month
--      replaces them, so the tracker resets at the boundary by itself.
--
--   2. `monthly_leaderboards` holds one frozen row per finished UTC month: `data`
--      is a JSON array of raw `[user_id, best_rank, best_rank_timed]`, written by
--      the month-change cron (src/index.js -> recordMonthlyLeaderboards). Identity
--      is deliberately NOT stored — the page resolves it at read time off
--      `external_ids` / `user_aliases` / `hidden_profiles`, so a later unlink, an
--      alias change or a hide applies to history.
--
-- The three columns are appended in schema.sql's order so a fresh database and
-- the live one stay positionally identical (pipeline.fetch_database's regenerated
-- dump is positional). ADD COLUMN appends, which is why this matches.
--
-- Apply to the live D1 with:
--   cd pwa && npx wrangler d1 execute tsumego-reports --remote \
--       --file=migrations/2026-09-21_monthly_leaderboard.sql
--
-- Local miniflare D1 for dev (stop `npm run dev` first):
--   sqlite3 .wrangler/state/v3/d1/miniflare-D1DatabaseObject/<hash>.sqlite \
--       ".read migrations/2026-09-21_monthly_leaderboard.sql"

ALTER TABLE leaderboard ADD COLUMN best_rank REAL;
ALTER TABLE leaderboard ADD COLUMN best_rank_timed REAL;
ALTER TABLE leaderboard ADD COLUMN best_rank_month TEXT;

CREATE TABLE IF NOT EXISTS monthly_leaderboards (
    month      TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
