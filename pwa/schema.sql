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
    last_seen_at      INTEGER
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
CREATE TABLE IF NOT EXISTS rank_surveys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    server TEXT NOT NULL,
    rank TEXT NOT NULL,
    created_at TEXT NOT NULL,
    country TEXT
);
-- For an existing deployment, run once against the live D1:
--   ALTER TABLE rank_surveys ADD COLUMN country TEXT;

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

-- EGD leaderboard: one row per registered user. Holds the EGD binding + a
-- materialized rank buffer (the user's last 500 novel attempts) + the derived
-- rank scalars, so /report can recompute the rank in-Worker with +1 read/+1 write.
-- `buffer` is a JSON array of {p: problem_id, s: success 0/1, t: time_used ms},
-- newest first, capped at 500. `pred_egf_rank` stays NULL until the buffer holds
-- >= 100 novel attempts (rank-start gate); the leaderboard read filters on
-- `pred_egf_rank IS NOT NULL`. gor_rank is the current/this-week GoR-rank;
-- prev_week_gor_rank is frozen at the week rollover so past deltas don't move.
-- week_id is the Monday-00:00-UTC week number. See docs/leaderboard_plan.md.
CREATE TABLE IF NOT EXISTS leaderboard (
    user_id            TEXT PRIMARY KEY,
    egd_pin            TEXT NOT NULL,
    egd_name           TEXT,
    country_code       TEXT,
    gor                INTEGER,
    gor_updated_at     TEXT,
    buffer             TEXT,
    pred_egf_rank      REAL,
    pred_egf_se        REAL,
    gor_rank           REAL,
    prev_week_gor_rank REAL,
    week_id            INTEGER,
    this_week_max      REAL,
    prev_week_max      REAL,
    last_active        TEXT
);
-- Enforce the 1:1 EGD-id <-> account binding (PK already gives account -> one PIN).
CREATE UNIQUE INDEX IF NOT EXISTS idx_leaderboard_egd_pin ON leaderboard(egd_pin);

-- Registration invitations, issued by an admin (generate_invite.py / the /invite
-- tool page) OR by any registered leaderboard member (the in-app Invite menu). A
-- random 8-char opaque token recorded alongside the target EGD PIN (no crypto). The
-- recipient opens /register?t=<token> in their PWA; the Worker looks the token up,
-- binds their user_id to the PIN, and marks it claimed (one-time use).
-- egd_name/country_code/gor are fetched from EGD at creation so the confirmation
-- preview + claim need no live EGD call. Never expires; deleted/revoked manually.
-- created_by_user_id is the account that minted the link — both the abuse-attribution
-- key and the "who invited this person" record (NULL for an admin CLI invite, which
-- has no user context). See docs/leaderboard_plan.md.
CREATE TABLE IF NOT EXISTS invitation (
    invite_token       TEXT PRIMARY KEY,
    egd_pin            TEXT NOT NULL,
    egd_name           TEXT,
    country_code       TEXT,
    gor                INTEGER,
    created_at         TEXT,
    created_by_user_id TEXT,
    claimed_by_user_id TEXT,
    claimed_at         TEXT
);
-- MIGRATION for an existing deployment (run once against the live D1; the CREATE
-- above already carries the column for a fresh DB):
--   ALTER TABLE invitation ADD COLUMN created_by_user_id TEXT;