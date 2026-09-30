-- Contribution-rewards entry form (2026-09-30): contest_entries also holds what a
-- player sends from the /contest page (POST /contest/submit), pending until the
-- organiser lists it. listed_at is the public date (NULL = pending); submitted_at,
-- which held that date until now, becomes the time the entry was sent, so the
-- existing rows copy it across. NOT idempotent (ALTER TABLE ADD COLUMN) — apply
-- ONCE, BEFORE the deploy (the page filters on listed_at):
--   cd pwa && npx wrangler d1 execute tsumego-reports --remote \
--       --file=migrations/2026-09-30_contest_submissions.sql
-- See the note beside the CREATE in schema.sql.

ALTER TABLE contest_entries ADD COLUMN listed_at TEXT;
ALTER TABLE contest_entries ADD COLUMN contact_method TEXT CHECK (contact_method IN ('whatsapp', 'discord', 'email'));
ALTER TABLE contest_entries ADD COLUMN contact TEXT;
ALTER TABLE contest_entries ADD COLUMN video_language TEXT;
ALTER TABLE contest_entries ADD COLUMN app_link INTEGER NOT NULL DEFAULT 0;
ALTER TABLE contest_entries ADD COLUMN message TEXT;
UPDATE contest_entries SET listed_at = submitted_at WHERE listed_at IS NULL;
