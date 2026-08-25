-- Google sign-in (2026-08-25): link a Google account to an account code, so a
-- device that does not have the code can still reach the account.
--
-- New table only — nothing existing changes shape, and nothing is backfilled:
-- every account starts unlinked and stays exactly as reachable by its code as it
-- was. Idempotent, and identical to the block appended to schema.sql, so a fresh
-- database and the live one agree.
--
-- Apply to the live D1 with:
--   cd pwa && npx wrangler d1 execute tsumego-reports --remote \
--       --file=migrations/2026-08-25_google_accounts.sql
CREATE TABLE IF NOT EXISTS google_accounts (
    google_sub TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_google_accounts_user ON google_accounts(user_id);
