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
    is_repeat INTEGER NOT NULL DEFAULT 0
);

-- Range index for the server-stats 7-day window. Without it, every
-- `WHERE created_at >= ?` is a full-table scan of all history (D1 bills rows
-- read), which is what the community-stats aggregation hits each recompute.
CREATE INDEX IF NOT EXISTS idx_tsumego_reports_created ON tsumego_reports(created_at);

-- Per-user cursor scan for POST /sync (account sync across devices). The
-- endpoint runs `WHERE user_id = ? AND created_at > ?` on every device poll;
-- with this index an up-to-date device reads ~0 rows, without it every poll
-- is a scan (D1 bills rows read) — same failure mode the created_at index
-- above guards against for server-stats.
CREATE INDEX IF NOT EXISTS idx_tsumego_reports_user_created ON tsumego_reports(user_id, created_at);

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
-- Dropping the legacy UNIQUE(user_id, problem) index in SQLite requires
-- rebuilding the table; the simplest path is:
--   DROP INDEX IF EXISTS sqlite_autoindex_tsumego_reports_1;
-- (D1 names auto-indexes this way; verify with `PRAGMA index_list('tsumego_reports')`
--  before dropping.)
-- (Not idempotent in SQLite, so it's not part of the schema script itself.)

-- One push subscription per user (single-device assumption — see CLAUDE.md /
-- pwa/README.md notification design notes). On userId collision the loser
-- silently stops getting notifications; acceptable at current scale.
CREATE TABLE IF NOT EXISTS push_subscriptions (
    user_id           TEXT PRIMARY KEY,
    endpoint          TEXT NOT NULL,
    p256dh            TEXT NOT NULL,
    auth              TEXT NOT NULL,
    last_notified_at  INTEGER,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL
);

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
-- track rank drift over time.
CREATE TABLE IF NOT EXISTS rank_surveys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    server TEXT NOT NULL,
    rank TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rank_surveys_user ON rank_surveys(user_id);

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
CREATE INDEX IF NOT EXISTS idx_share_events_problem ON share_events(problem_id);

-- Cached server-wide stats, recomputed at most once per ~24h by the cron from
-- the preceding 7 days of tsumego_reports + rank_surveys. Single logical row
-- (key='latest'); `data` is the precomputed JSON the PWA renders verbatim.
CREATE TABLE IF NOT EXISTS server_stats (
    key         TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    computed_at TEXT NOT NULL
);