CREATE TABLE IF NOT EXISTS tsumego_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    problem TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    created_at TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 9,
    move TEXT,
    time_used INTEGER,
    review_time INTEGER,
    is_repeat INTEGER NOT NULL DEFAULT 0,
    -- Per-profile sync digest (docs/account_sync_redesign.md). user_seq is the
    -- 1-based count of the user's attempts; running_sum is the EXCLUSIVE prefix
    -- sum of shared/sync-digest.js's rowHash(id) over the user's earlier rows
    -- (a row's own id isn't known until insert, so the live digest adds
    -- rowHash(latest.id) at read time). Both are computed in the INSERT (see
    -- src/handlers/report.js) and lazy-seed on a user's first post-migration
    -- attempt — NULL until then. They replace the prevCreatedAt piggyback as
    -- the change-detection signal and let /sync page by id instead of
    -- created_at (so a row inserted with a historical created_at, e.g. an
    -- offline flush, is never missed).
    user_seq INTEGER,
    running_sum INTEGER
);

-- Range index for the server-stats 7-day window. Without it, every
-- `WHERE created_at >= ?` is a full-table scan of all history (D1 bills rows
-- read), which is what the community-stats aggregation hits each recompute.
CREATE INDEX IF NOT EXISTS idx_tsumego_reports_created ON tsumego_reports(created_at);

-- Per-user id-cursor scan for POST /sync (account sync across devices) and the
-- O(1) latest-row lookup the digest INSERT/read use (`... WHERE user_id = ?
-- ORDER BY id DESC LIMIT 1`). /sync pages by `WHERE user_id = ? AND id > ?
-- ORDER BY id`; id (arrival order) rather than created_at (event time) is the
-- gap-free cursor — see the redesign doc.
CREATE INDEX IF NOT EXISTS idx_tsumego_reports_user_id ON tsumego_reports(user_id, id);

-- (The legacy per-user created_at index idx_tsumego_reports_user_created was
-- dropped with the created_at /sync path + the prevCreatedAt piggyback + the
-- /report-enrich handler. On the live D1, drop it AFTER deploying the new
-- Worker, never before — the still-live old Worker uses it until the deploy:
--   DROP INDEX IF EXISTS idx_tsumego_reports_user_created;)

-- One row per attempt (including repeats). There is no UNIQUE constraint on
-- (user_id, problem) anymore — the client is expected to send a report for
-- every solving attempt, and is_repeat flags whether the user had seen the
-- problem before. created_at is the per-attempt timestamp.

-- For an existing deployment, run once against the live D1:
--   ALTER TABLE tsumego_reports ADD COLUMN size INTEGER NOT NULL DEFAULT 9;
--   ALTER TABLE tsumego_reports ADD COLUMN move TEXT;
--   ALTER TABLE tsumego_reports ADD COLUMN time_used INTEGER;
--   ALTER TABLE tsumego_reports ADD COLUMN review_time INTEGER;
--   ALTER TABLE tsumego_reports ADD COLUMN is_repeat INTEGER NOT NULL DEFAULT 0;
-- Sync-digest columns + id-cursor index (docs/account_sync_redesign.md). No
-- backfill UPDATE: existing rows stay NULL and the INSERT lazy-seeds each user's
-- latest row on their first post-migration attempt (a one-time COUNT/SUM read),
-- so this costs ZERO writes — a full ROW_NUMBER backfill would blow the daily
-- write cap and is unnecessary (only the latest row per user is ever read).
--   ALTER TABLE tsumego_reports ADD COLUMN user_seq INTEGER;
--   ALTER TABLE tsumego_reports ADD COLUMN running_sum INTEGER;
--   CREATE INDEX IF NOT EXISTS idx_tsumego_reports_user_id ON tsumego_reports(user_id, id);
-- Dropping the legacy UNIQUE(user_id, problem) index in SQLite requires
-- rebuilding the table; the simplest path is:
--   DROP INDEX IF EXISTS sqlite_autoindex_tsumego_reports_1;
-- (D1 names auto-indexes this way; verify with `PRAGMA index_list('tsumego_reports')`
--  before dropping.)
-- (Not idempotent in SQLite, so it's not part of the schema script itself.)

-- One row PER DEVICE (endpoint), not per account: a single user_id can hold
-- several push subscriptions (phone + laptop + …) and the recall cron fans each
-- reminder out to all of them. The push endpoint URL is globally unique per push
-- channel, so it is the natural PK; user_id is indexed for the cron's per-user
-- lookups and the /schedule activity-watermark update. A device that adopts a
-- different account (account switch) re-subscribes the SAME endpoint, so the
-- ON CONFLICT(endpoint) upsert in handleSubscribe just re-points user_id — no
-- stale row, and no device ever belongs to two accounts at once.
CREATE TABLE IF NOT EXISTS push_subscriptions (
    endpoint          TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL,
    p256dh            TEXT NOT NULL,
    auth              TEXT NOT NULL,
    last_notified_at  INTEGER,         -- per-device anti-spam floor
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL,
    -- Activity watermark for the recall cron's mid-session suppression: bumped
    -- by every /schedule call (across the account's device rows), even when the
    -- schedule diff is empty. Replaces relying on recall_schedule.updated_at,
    -- which diff-mode uploads no longer refresh across the board.
    last_seen_at      INTEGER,
    -- Resolved UI language of the device at subscribe time (one of en/zh/ja/ko;
    -- the client sends navigator.language and the /push/subscribe handler folds
    -- it to a supported code). The recall cron renders each device's reminder in
    -- its own language. NULL / anything unknown -> English at send time.
    lang              TEXT
);
CREATE INDEX IF NOT EXISTS idx_push_subscriptions_user ON push_subscriptions(user_id);

-- MIGRATIONS for an existing deployment (run once each against the live D1;
-- not idempotent, so kept out of the CREATE above):
--   1. last_seen_at column (predates multi-device):
--        ALTER TABLE push_subscriptions ADD COLUMN last_seen_at INTEGER;
--   2. Single-row (user_id PK) -> per-device (endpoint PK). SQLite can't drop a
--      PK in place, so rebuild, preserving existing subscriptions:
--        ALTER TABLE push_subscriptions RENAME TO push_subscriptions_old;
--        CREATE TABLE push_subscriptions (
--          endpoint TEXT PRIMARY KEY, user_id TEXT NOT NULL,
--          p256dh TEXT NOT NULL, auth TEXT NOT NULL, last_notified_at INTEGER,
--          created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
--          last_seen_at INTEGER);
--        INSERT OR IGNORE INTO push_subscriptions
--          (endpoint, user_id, p256dh, auth, last_notified_at, created_at, updated_at, last_seen_at)
--          SELECT endpoint, user_id, p256dh, auth, last_notified_at,
--                 created_at, updated_at, last_seen_at FROM push_subscriptions_old;
--        DROP TABLE push_subscriptions_old;
--        CREATE INDEX IF NOT EXISTS idx_push_subscriptions_user ON push_subscriptions(user_id);
--   3. lang column (per-device notification language):
--        ALTER TABLE push_subscriptions ADD COLUMN lang TEXT;

-- Per-(user, problem) recall window. enter_at / exit_at are unix ms epochs at
-- which the client-side logistic model predicts P(recall) crosses 0.80 and
-- 0.60 respectively. The hourly cron fires one notification when enter_at is
-- reached (recall window opens), and another in the 1-2h before exit_at.
CREATE TABLE IF NOT EXISTS recall_schedule (
    user_id            TEXT NOT NULL,
    problem_id         TEXT NOT NULL,
    enter_at           INTEGER NOT NULL,
    exit_at            INTEGER NOT NULL,
    enter_notified_at  INTEGER,
    exit_notified_at   INTEGER,
    updated_at         INTEGER NOT NULL,
    PRIMARY KEY (user_id, problem_id)
);

CREATE INDEX IF NOT EXISTS idx_recall_due_enter
    ON recall_schedule(enter_at) WHERE enter_notified_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_recall_due_exit
    ON recall_schedule(exit_at)  WHERE exit_notified_at  IS NULL;

-- One row per (user, server) rank declaration. The client surveys the user
-- every ~2 months once they have a meaningful answer history; each submission
-- inserts one row per non-empty server selection. Older rows stay so we can
-- track rank drift over time. country is Cloudflare's edge geo-IP
-- (request.cf.country), filled server-side (same as share_events); null for
-- older clients / when unavailable. It is the LAST column on purpose: an
-- ALTER ... ADD COLUMN on the live D1 appends it last, and the fetch_database
-- dump is positional VALUES() in column order, so CREATE order must match the
-- migrated order or the .sql parser (utils/rank_surveys.py) misreads columns.
-- ALSO holds the MACHINE-written weekly rating heartbeats for registered
-- players: server = 'EGD' (raw GoR digits e.g. '2145', from the client GoR
-- piggyback, country = the EGD federation code) or 'AGAR' (raw AGA rating e.g.
-- '3.2' / '-4.5', from the Worker-side AGA Ratings Explorer fetch) — both
-- distinct from the human-declared 'EGF' / 'AGA' rank labels. Full precision
-- for the leaderboard display; unparseable by rank_to_scalar, so the
-- calibration pipeline ignores 'EGD'/'AGAR' rows until deliberately added to
-- SERVERS. See the external_ids note. Heartbeat rows also carry `rank_label`
-- (the source's OWN official short rank — EGD Grade "4d"/"3p", AGA rank_label
-- "5d") and `last_game_at` (ISO date of the player's last tournament — EGD
-- Last_Appearance / AGA latest_event_date), which the /leaderboard "Declared"
-- column shows and activity-gates on (EGF activity criteria). Both NULL on
-- human survey rows and on heartbeats from sources that omitted them.
CREATE TABLE IF NOT EXISTS rank_surveys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    server TEXT NOT NULL,
    rank TEXT NOT NULL,
    created_at TEXT NOT NULL,
    country TEXT,
    rank_label TEXT,
    last_game_at TEXT
);
-- For an existing deployment, run once against the live D1 (order matters —
-- the fetch_database dump is positional, so ALTERs must append in this order):
--   ALTER TABLE rank_surveys ADD COLUMN country TEXT;
--   ALTER TABLE rank_surveys ADD COLUMN rank_label TEXT;      (migrations/2026-07-21_rank_surveys_verified.sql)
--   ALTER TABLE rank_surveys ADD COLUMN last_game_at TEXT;    (same file)

-- One row per share-link event. event_type is 'create' (user generated a share
-- link) or 'open' (someone arrived via a shared link). referrer is the client's
-- document.referrer for opens — "where the link came from", derived from the
-- navigation's Referer header — and null for creates. country is Cloudflare's
-- edge geo-IP (request.cf.country), filled server-side. size / problem_id /
-- user_id may be null for older clients.
CREATE TABLE IF NOT EXISTS share_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    user_id TEXT,
    problem_id TEXT,
    size INTEGER,
    country TEXT,
    referrer TEXT,
    created_at TEXT NOT NULL
);

-- Cached server-wide stats, recomputed at most once per ~24h by the cron from
-- the preceding 7 days of tsumego_reports. Single logical row
-- (key='latest'); `data` is the precomputed JSON the PWA renders verbatim.
CREATE TABLE IF NOT EXISTS server_stats (
    key         TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    computed_at TEXT NOT NULL
);

-- Per-user daily counter for full-history pulls (POST /sync with sinceId=0 — the
-- from-the-beginning scan: the stats "restore evicted history" refresh, account
-- adoption, and the outbox reconcile all count). Each is a full
-- per-user table scan, so it is server-capped per UTC day (client enforcement is
-- bypassable by clearing localStorage). One row per (user, day); the hourly cron
-- prunes days older than a small window. The (user_id, day) primary key makes
-- the check a point lookup.
CREATE TABLE IF NOT EXISTS history_refresh_log (
    user_id TEXT NOT NULL,
    day     TEXT NOT NULL,
    count   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);

-- External-service player-id bindings: one row per LINKED user. A general table
-- — each supported service is a nullable column here (egd_pin, aga_id, ...),
-- hence not named after EGD. A user creates their own row by picking themselves
-- out of the EGD/AGA search in the app's Account dialog (POST /link,
-- handlers/link.js) and deletes it again with Unlink, so the row is entirely
-- user-owned and NOT a trust boundary: no verification against either database
-- is possible, and none is claimed.
--
-- `display_name` and `country_code` are the identity's own display fields,
-- snapshotted from the service at link time — the /leaderboard "Player" column.
-- They live HERE rather than on a separate row precisely because they describe
-- the binding and must go away with it. (Before 2026-08-25 registration was
-- invite-only and they lived on the claimed `invitation` row; see
-- migrations/2026-08-25_self_link_identity.sql for the backfill.)
--
-- The player's GRADE is deliberately NOT duplicated here. It already lives on the
-- machine heartbeat in `rank_surveys` — refreshed weekly, and the value the
-- leaderboard's Declared column shows — so the Account dialog reads it from there
-- too (readLinkedIdentity) and the two cannot disagree.
--
-- Everything else about a linked player lives elsewhere: rating history as
-- machine-written heartbeat rows in `rank_surveys` (weekly, one row per linked
-- player per week; 'EGD' = raw GoR digits e.g. '2145' from the client GoR
-- piggyback, 'AGAR' = raw AGA rating e.g. '3.2' from the Worker-side AGAGD
-- fetch; the latest such row doubles as the "checked this week" marker, and
-- rank_to_scalar can't parse raw digits so the rank-calibration pipeline ignores
-- 'EGD'/'AGAR' rows until deliberately enabled), and the rank/buffer state on
-- the (prunable) `leaderboard` row. Unlinking deletes this row AND the account's
-- 'EGD'/'AGAR' heartbeats together — a heartbeat is keyed by user_id, so one
-- left behind would keep showing the old player's verified rating.
--
-- The schema supports an egd_pin AND an aga_id on one row; the "one identity per
-- account" rule is enforced only as POLICY in handleLink (link.js), NOT as a DB
-- constraint — so allowing both later is a code change with no migration.
-- Because binding is restricted to one, the frontend assumes a single identity.
CREATE TABLE IF NOT EXISTS external_ids (
    user_id      TEXT PRIMARY KEY,
    egd_pin      TEXT,
    aga_id       TEXT,
    display_name TEXT,
    country_code TEXT
);
-- MIGRATION for an existing deployment (the CREATE above already carries them for
-- a fresh DB; the fetch_database dump is positional, so ALTERs must append in this
-- order) — migrations/2026-08-25_self_link_identity.sql:
--   ALTER TABLE external_ids ADD COLUMN display_name TEXT;
--   ALTER TABLE external_ids ADD COLUMN country_code TEXT;
-- Enforce a service-id <-> account binding is 1:1 (PK already gives account -> one
-- of each id). NULLs are distinct in a UNIQUE index, so many egd_pin-NULL (AGA) or
-- aga_id-NULL (EGD) rows coexist.
CREATE UNIQUE INDEX IF NOT EXISTS idx_external_ids_egd_pin ON external_ids(egd_pin);
CREATE UNIQUE INDEX IF NOT EXISTS idx_external_ids_aga_id  ON external_ids(aga_id);

-- Leaderboard: one row per user (created lazily on their first novel /report).
-- Holds a materialized rank buffer (the user's last 500 novel attempts) + the
-- derived rank scalars, so /report can recompute the rank in-Worker with
-- +1 read/+1 write. EVERY row is rebuildable from tsumego_reports (registration
-- lives in external_ids, not here), so idle rows (last_active > 7 days old) are
-- DELETED by the weekly cron (pruneIdleLeaderboardRows) and lazily re-created on
-- the user's next novel /report — the table stays at roughly weekly-active
-- size. The /leaderboard page shows unregistered rows as "Anonymous" and limits
-- itself to rows active in the past week.
-- `buffer` is a JSON array of {p: problem_id, s: success 0/1, t: time_used ms},
-- newest first, capped at 500. `pred_rank` stays NULL until the buffer holds
-- >= RANK_START_MIN valid novel attempts (rank-start gate).
--
-- The rank is stored in the window estimator's OWN output frame — `pred_rank` is
-- the UNTIMED internal onlymove rank scalar (1k = 0) and `pred_rank_se` its standard
-- error in the same units (stones) — NOT the EGF frame the columns used to hold
-- (`pred_egf_rank`/`pred_egf_se`, replaced 2026-07-27; see migrations/
-- 2026-07-27_leaderboard_internal_rank.sql). Reason: both readers wanted the internal
-- frame and inverted the EGF coupling straight back out of the stored value — the
-- Community rank histogram exclusively, the /leaderboard page for its default column
-- and its sort key — while only the page's EGF units mode consumed the stored frame
-- as-is. `pred_egf_se` was also misnamed: it always held the fit's stones SE, and
-- nothing read the column at all.
--
-- `time_delta_rating` is the user's personal time dock in RATING units, derived by
-- computeRank alongside the rank and stored so the server-frame columns (EGF/AGA,
-- whose couplings are fitted against TIME-ADJUSTED ratings) can add it forward. It
-- used to be baked into the stored rank and re-derived from `buffer` — a full buffer
-- pass + weighted median per row — on every read.
CREATE TABLE IF NOT EXISTS leaderboard (
    user_id            TEXT PRIMARY KEY,
    buffer             TEXT,
    pred_rank          REAL,
    pred_rank_se       REAL,
    time_delta_rating  REAL,
    last_active        TEXT,
    -- Edge geo-IP country (ISO 3166-1 alpha-2), stamped from request.cf.country into
    -- the row's existing /report write (no extra write). The universal flag fallback
    -- for the /leaderboard page: shows a flag for ANY active player, not just the
    -- registered/surveyed few. EGD-sourced country (GoR heartbeat, invitation) still
    -- wins over it — see readLeaderboardRows.
    country            TEXT
);
-- There is deliberately NO index on last_active. It would be maintained on
-- EVERY novel /report (the hot-path UPDATE sets last_active each time, and D1
-- bills index maintenance as rows written — thousands of writes/week), while
-- the only queries it would serve (the /leaderboard page's active-past-week
-- filter, behind a 60s edge cache, and the weekly cron) full-scan a table the
-- weekly prune keeps at roughly weekly-active size — a few hundred rows read,
-- and D1 reads are ~1000x cheaper than writes. Revisit only if the table is
-- allowed to grow unbounded again.

-- RETIRED (2026-08-25). Registration invitations, from when reaching the
-- leaderboard meant claiming a link an existing member minted for you. A user
-- now picks their own EGD/AGA identity in the Account dialog, so nothing reads
-- or writes this table: the display names it held were copied onto
-- external_ids by migrations/2026-08-25_self_link_identity.sql, which is the
-- last thing that needed it. The CREATE stays so a fresh DB still matches
-- production; `DROP TABLE invitation` once the live leaderboard is confirmed
-- showing names off external_ids.
CREATE TABLE IF NOT EXISTS invitation (
    invite_token       TEXT PRIMARY KEY,
    egd_pin            TEXT,
    egd_name           TEXT,
    country_code       TEXT,
    gor                INTEGER,
    created_at         TEXT,
    created_by_user_id TEXT,
    claimed_by_user_id TEXT,
    claimed_at         TEXT,
    service            TEXT,      -- 'EGD' (or NULL = EGD) / 'AGA'
    aga_id             TEXT,      -- AGA member id, when service = 'AGA'
    aga_rating         REAL       -- AGA rating snapshot, when service = 'AGA'
);
-- MIGRATION for an existing deployment (run once against the live D1; the CREATE
-- above already carries the columns for a fresh DB):
--   ALTER TABLE invitation ADD COLUMN created_by_user_id TEXT;
-- The service/aga_id/aga_rating columns (and relaxing egd_pin's NOT NULL for AGA
-- invites) need a table rebuild, not a plain ADD COLUMN — see
-- migrations/2026-07-21_external_ids_aga.sql.

-- Per-user "hide my profile" opt-out for the public /leaderboard + /tournament
-- pages. Presence of a row = hidden: the user's display name, country, and
-- self-declared / GoR survey ranks are withheld from readLeaderboardRows (a
-- hidden player still appears on /leaderboard as an anonymous row carrying only
-- their onlymove estimate, and is dropped from the registered-centric
-- /tournament listing). Keyed by user_id, applying to ANY account — a
-- never-registered player can still have declared ranks in rank_surveys that
-- this hides — so it is its own table rather than a column on external_ids.
-- Toggled from the bottom of the /leaderboard page AND from a checkbox at the
-- foot of the in-app rank survey (both POST /profile/hide) — one flag, two
-- entry points.
CREATE TABLE IF NOT EXISTS hidden_profiles (
    user_id    TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
-- Google sign-in: a second way back into an account CODE, never a second kind of
-- account. Presence of a row means the Google account whose opaque subject id is
-- `google_sub` can be used to look this account's code up again on a device that
-- does not have it (POST /google/signin -> handlers/google.js).
--
-- What is stored is the SUBJECT ID AND NOTHING ELSE. The ID token also carries
-- the address and the display name and neither is kept, so there is no email in
-- this database to serve, to leak, or to have to describe in the privacy policy.
-- The Account dialog names the address from the token it just received, on the
-- linking device only.
--
-- The two keys are what make the binding EXCLUSIVE IN BOTH DIRECTIONS — one code
-- per Google account (PRIMARY KEY) and one Google account per code (the UNIQUE
-- index) — which is what lets a sign-in resolve to a single code rather than to a
-- list. Deleting a row is Unlink: the account is untouched and keeps its code.
CREATE TABLE IF NOT EXISTS google_accounts (
    google_sub TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_google_accounts_user ON google_accounts(user_id);
