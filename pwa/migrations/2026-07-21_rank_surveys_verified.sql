-- Verified-rating display fields for the machine heartbeat rows ('EGD'/'AGAR')
-- in rank_surveys: the source's own official short rank label (EGD Grade
-- "4d"/"3p", AGA rank_label "5d") and the player's last tournament date (ISO,
-- from EGD Last_Appearance / AGA latest_event_date) — what the /leaderboard
-- "Verified" column displays and activity-gates on (EGF activity criteria:
-- 2y dan, 1y SDK, 6mo DDK). Appended LAST deliberately: the fetch_database
-- dump is positional VALUES() in column order (see the rank_surveys note in
-- schema.sql). Run once against the live D1:
--   npx wrangler d1 execute tsumego-reports --remote --file=migrations/2026-07-21_rank_surveys_verified.sql
ALTER TABLE rank_surveys ADD COLUMN rank_label TEXT;
ALTER TABLE rank_surveys ADD COLUMN last_game_at TEXT;
