-- Self-selected federation identity: the display name moves onto the binding (2026-08-25).
--
-- Before: an account reached the leaderboard by claiming an INVITATION, and the
-- invitation row was where the player's display name and country lived —
-- readLeaderboardRows joined `invitation` on claimed_by_user_id to get them.
-- `external_ids` held nothing but the ids.
--
-- After: a user picks their own EGD or AGA identity from the Account dialog
-- (POST /link), so there is no invitation row to carry the name. The name and
-- country now live on the binding itself, which is also where they belong: they
-- describe the linked identity, and they must survive an unlink/relink. The
-- player's GRADE is deliberately NOT copied here — it is already kept, and kept
-- CURRENT, on their machine heartbeat in rank_surveys, which is what both the
-- leaderboard and the Account dialog read.
--
-- ADD COLUMN (not a table rebuild): schema.sql appends both columns after
-- aga_id, so plain ALTERs already land them in schema order — which matters
-- because pipeline.fetch_database's regenerated dump is positional.
ALTER TABLE external_ids ADD COLUMN display_name TEXT;
ALTER TABLE external_ids ADD COLUMN country_code TEXT;

-- Carry every already-registered player's identity across from the invitation
-- they claimed, so the leaderboard keeps showing their name the moment the
-- deploy that reads external_ids lands. One row per registered account; the
-- subselect takes the LATEST claim (MAX(claimed_at)), matching the join it
-- replaces.
UPDATE external_ids
   SET display_name = (SELECT i.egd_name FROM invitation i
                        WHERE i.claimed_by_user_id = external_ids.user_id
                        ORDER BY i.claimed_at DESC LIMIT 1),
       country_code = (SELECT i.country_code FROM invitation i
                        WHERE i.claimed_by_user_id = external_ids.user_id
                        ORDER BY i.claimed_at DESC LIMIT 1)
 WHERE display_name IS NULL;

-- `invitation` is left in place but is now read and written by NOTHING: the
-- backfill above is the last thing that ever needed it. Drop it once the
-- leaderboard is confirmed showing names off external_ids:
--   DROP TABLE invitation;
