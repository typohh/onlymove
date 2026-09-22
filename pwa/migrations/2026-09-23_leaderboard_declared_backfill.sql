-- Declared-rank cache on the leaderboard (2026-09-23), step 3 of 3 — run AFTER
-- the deploy; see 2026-09-23_leaderboard_declared.sql for the order.
--
-- Recomputes every row's cache from the rank_surveys log. Idempotent, so it is
-- also the repair if the cache is ever suspected of drifting. The expression is
-- declaredJsonSql (src/leaderboard.js) inlined — keep the two in step.
--
--   cd pwa && npx wrangler d1 execute tsumego-reports --remote \
--       --file=migrations/2026-09-23_leaderboard_declared_backfill.sql

UPDATE leaderboard SET declared = (SELECT json_group_array(json_object(
        'server', s.server, 'rank', s.rank, 'country', s.country,
        'rank_label', s.rank_label, 'last_game_at', s.last_game_at))
   FROM rank_surveys s
  WHERE s.id IN (SELECT MAX(id) FROM rank_surveys
                  WHERE user_id = leaderboard.user_id GROUP BY server));
