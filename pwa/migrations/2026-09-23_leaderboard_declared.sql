-- Declared-rank cache on the leaderboard (2026-09-23), step 1 of 3.
--
-- `leaderboard.declared` caches each user's latest rank_surveys row per server so
-- the /leaderboard page stops grouping the whole rank_surveys log on every render,
-- and `idx_rank_surveys_user` makes the per-user reads of that log (the cache
-- refresh, GET /profile, the heartbeats' weekly guard) index reads. See the
-- declared-rank cache note in src/leaderboard.js.
--
-- Three steps, in this order:
--   1. this file — additive; the old Worker never names `declared`, so it keeps
--      serving unchanged. Required BEFORE the deploy: the new Worker names the
--      column in its leaderboard SELECT and in every rank_surveys write batch.
--   2. `npm run deploy`.
--   3. 2026-09-23_leaderboard_declared_backfill.sql — fills the cache for every
--      existing row. AFTER the deploy, so nothing the old Worker wrote in between
--      is missed. Until it runs, the page's Declared column is empty.
--
-- Apply to the live D1 with:
--   cd pwa && npx wrangler d1 execute tsumego-reports --remote \
--       --file=migrations/2026-09-23_leaderboard_declared.sql
--
-- Local miniflare D1 for dev (stop `npm run dev` first), both files:
--   sqlite3 .wrangler/state/v3/d1/miniflare-D1DatabaseObject/<hash>.sqlite \
--       ".read migrations/2026-09-23_leaderboard_declared.sql" \
--       ".read migrations/2026-09-23_leaderboard_declared_backfill.sql"

ALTER TABLE leaderboard ADD COLUMN declared TEXT;
CREATE INDEX IF NOT EXISTS idx_rank_surveys_user ON rank_surveys(user_id);
