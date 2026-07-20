-- One-time migration: generalize registration identity to hold an AGA member id
-- alongside the EGD PIN, and let an invitation target either database (2026-07-21).
--
-- Adds AGA (American Go Association) rating linking as a parallel to EGD, through
-- the SAME invite/claim system:
--   * external_ids gains a nullable `aga_id` (uniquely indexed, like egd_pin), and
--     `egd_pin` is relaxed from NOT NULL so an AGA-only registration has egd_pin
--     NULL. The SCHEMA now supports BOTH ids on one row; the "one identity per
--     account" rule is enforced only as policy in handleRegister (register.js), NOT
--     as a DB constraint — so relaxing to "allow both" later needs no migration.
--   * invitation gains `service` ('EGD' default/NULL, or 'AGA'), `aga_id`, and
--     `aga_rating` (the AGA rating snapshotted at invite time). egd_name/country_code
--     are reused as the generic display name/country for both services; `gor` stays
--     EGD-only.
--   * The machine AGA rating heartbeat lands in rank_surveys as server = 'AGAR'
--     (raw AGA rating digits e.g. '3.2' / '-4.5'), distinct from the human-declared
--     'AGA' (rank labels) — exactly as 'EGD' (machine GoR) is distinct from 'EGF'
--     (human). 'AGAR' is NOT added to the calibration SERVERS (utils/rank_surveys.py),
--     so the offline rank calibration ignores it, same as 'EGD'.
--
-- Deploy the matching Worker in the same session as this migration: the new code
-- reads external_ids.aga_id and invitation.service/aga_id/aga_rating.
--
-- Apply to the remote D1 (from pwa/):
--   npx wrangler d1 execute tsumego-reports --remote --file=migrations/2026-07-21_external_ids_aga.sql
-- And to the local miniflare D1 for dev (stop `npm run dev` first):
--   npx wrangler d1 execute tsumego-reports --local --file=migrations/2026-07-21_external_ids_aga.sql
-- (A later pipeline.fetch_database run also propagates the remote schema to the
-- local mirror + miniflare automatically.)

-- external_ids: relax egd_pin NOT NULL (SQLite needs a table rebuild) and add the
-- nullable aga_id column. Both ids are uniquely indexed; NULLs are distinct in a
-- UNIQUE index, so any number of egd_pin-NULL (AGA) or aga_id-NULL (EGD) rows are fine.
CREATE TABLE external_ids_new (
    user_id TEXT PRIMARY KEY,
    egd_pin TEXT,
    aga_id  TEXT
);
INSERT INTO external_ids_new (user_id, egd_pin)
    SELECT user_id, egd_pin FROM external_ids;
DROP TABLE external_ids;
ALTER TABLE external_ids_new RENAME TO external_ids;
CREATE UNIQUE INDEX IF NOT EXISTS idx_external_ids_egd_pin ON external_ids(egd_pin);
CREATE UNIQUE INDEX IF NOT EXISTS idx_external_ids_aga_id  ON external_ids(aga_id);

-- invitation: add the AGA-targeted columns AND relax egd_pin NOT NULL (an AGA invite
-- has egd_pin NULL). Both need a table rebuild — a plain ALTER ADD COLUMN would leave
-- egd_pin NOT NULL and reject every AGA invite. Column order matches schema.sql.
CREATE TABLE invitation_new (
    invite_token       TEXT PRIMARY KEY,
    egd_pin            TEXT,
    egd_name           TEXT,
    country_code       TEXT,
    gor                INTEGER,
    created_at         TEXT,
    created_by_user_id TEXT,
    claimed_by_user_id TEXT,
    claimed_at         TEXT,
    service            TEXT,
    aga_id             TEXT,
    aga_rating         REAL
);
INSERT INTO invitation_new
    (invite_token, egd_pin, egd_name, country_code, gor, created_at,
     created_by_user_id, claimed_by_user_id, claimed_at)
    SELECT invite_token, egd_pin, egd_name, country_code, gor, created_at,
           created_by_user_id, claimed_by_user_id, claimed_at
      FROM invitation;
DROP TABLE invitation;
ALTER TABLE invitation_new RENAME TO invitation;
