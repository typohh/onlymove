-- Store the leaderboard rank in the estimator's OWN frame (2026-07-27).
--
-- Before: `pred_egf_rank` held the rank AFTER the personal time dock and the EGF
-- coupling had been applied, and `pred_egf_se` held the fit's standard error — which
-- was never in EGF units at all, but in internal rank (stones). Nothing read that SE
-- column anywhere; it was write-only.
--
-- After: `pred_rank` holds the UNTIMED internal onlymove rank scalar (1k = 0),
-- `pred_rank_se` its standard error in the same units, and `time_delta_rating` the
-- personal time dock in RATING units. computeRank produces all three directly, with
-- no coupling applied. The old columns are DROPPED here, in the same step.
--
-- Why: every reader inverted the stored value straight back out. The Community rank
-- histogram (readActivePlayers -> server-stats.js) wants the untimed internal scalar
-- exclusively; the /leaderboard page wants it for its default column and its sort key,
-- and only its EGF units mode used the stored frame as-is. The dock made it worse — it
-- was computed at write time and discarded, so each read re-derived it from `buffer`
-- (a full pass + weighted median, per row, per render) purely to undo it again.
--
-- A table rebuild rather than ADD + DROP COLUMN, so the surviving columns land in
-- exactly schema.sql's order (ALTER would leave last_active/country ahead of the new
-- rank columns, and pipeline.fetch_database's regenerated dump is positional). The
-- table is roughly weekly-active size — a few hundred rows — so the copy is trivial.
-- There are no indexes to recreate: the PK is the only one, deliberately (an index on
-- last_active would cost an index write per novel /report — see schema.sql).
--
-- THE NEW COLUMNS START NULL AND THAT IS FINE. resolveRowRank recomputes from `buffer`
-- whenever the stored rank is NULL, so every row reads correct from the first request
-- after the deploy; rows then repopulate on their owner's next novel /report, and the
-- update_leaderboard_ranks.mjs step at the end of the chain below fills them all in at
-- once. `buffer`, `last_active` and `country` are carried over untouched.
--
-- RUN AS ONE CHAINED SEQUENCE (this migration switches the schema over in a single
-- step, so the old Worker is briefly live against the new table — see below):
--
--   cd pwa && npx wrangler d1 execute tsumego-reports --remote \
--       --file=migrations/2026-07-27_leaderboard_internal_rank.sql && \
--     npm run deploy && \
--     cd .. && ./.venv/bin/python3 -m pipeline.fetch_database && \
--     node auxiliary/update_leaderboard_ranks.mjs --remote
--
-- Between the migration and the deploy the OLD Worker is still serving and still
-- writing pred_egf_rank/pred_egf_se, which no longer exist. Every path degrades
-- softly and nothing is lost permanently:
--   * POST /report — the leaderboard write is ctx.waitUntil(...).catch(...), so it
--     logs and is swallowed. The attempt still INSERTs into tsumego_reports and the
--     client still gets its 201 + ack. It just misses that row's incremental buffer
--     push — which the final update_leaderboard_ranks.mjs step repairs, since it
--     rebuilds every buffer from tsumego_reports.
--   * GET /leaderboard — readLeaderboardRows is try/caught; the page renders empty
--     (200) for the window instead of erroring.
--   * GET /server-stats — computeLeaderboardStats is try/caught, and the blob is
--     recomputed on the version mismatch the deploy causes anyway.
-- Keep the window short regardless; there is no reason to sit in it.
--
-- The fetch_database step is REQUIRED before update_leaderboard_ranks.mjs, for two
-- separate reasons: it propagates this new schema to the local mirror (the script
-- SELECTs pred_rank off attempts/database.sqlite and fails loudly on `no such column`
-- otherwise), and it is the snapshot the script's buffer rebuild is computed from.
-- CHAIN it into the apply as above — every attempt reported in a gap between snapshot
-- and write is silently truncated out of that user's buffer, permanently (see the
-- script's header).
--
-- Local miniflare D1 for dev (stop `npm run dev` first):
--   sqlite3 .wrangler/state/v3/d1/miniflare-D1DatabaseObject/<hash>.sqlite ".read migrations/2026-07-27_leaderboard_internal_rank.sql"

CREATE TABLE leaderboard_new (
    user_id            TEXT PRIMARY KEY,
    buffer             TEXT,
    pred_rank          REAL,
    pred_rank_se       REAL,
    time_delta_rating  REAL,
    last_active        TEXT,
    country            TEXT
);

-- pred_rank / pred_rank_se / time_delta_rating are left NULL deliberately: the old
-- values are in the wrong frame and there is no honest conversion for the SE (the fit
-- would have to be re-run), so they are recomputed rather than translated.
INSERT INTO leaderboard_new (user_id, buffer, last_active, country)
    SELECT user_id, buffer, last_active, country FROM leaderboard;

DROP TABLE leaderboard;
ALTER TABLE leaderboard_new RENAME TO leaderboard;
