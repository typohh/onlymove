# Tsumego SPA

## 1. Overview

The mobile-first Single Page Application served at [onlymove.net](https://onlymove.net) for practising Go (Baduk / Weiqi) problems — whole-board positions in which exactly one move wins. Both 9×9 and 13×13 corpora are supported; the size is picked up from the URL path (`/`, `/9`, or `/13`) at boot and then persists in state — it is an entry hint, not a persistent address.

The pipeline publishes raw per-problem numbers (a rating, a discrimination slope, a guessing floor, and a motif vector) and the client does the rest: it estimates the user's rank from their own recent history, picks problems against that estimate, runs a spaced-repetition model over previously-failed problems, and fits a per-user weakness model over the motif axes. When a user fails a problem they enter Analysis mode, where variations and best-move/score data are read from the SGF itself; if the SGF lacks that data, the front-end calls a server-side KataGo fallback.

The codebase is vanilla JavaScript (ES modules), bundled with esbuild and shipped as a Cloudflare Worker. `docs/methodology.md` (chapter 2) is the methodology behind the client-side models; `CLAUDE.md` is the invariant reference for the subsystems below.

## 2. Technology stack

| Category       | Technology                                  | Purpose                                                           |
|----------------|---------------------------------------------|-------------------------------------------------------------------|
| Frontend       | HTML, CSS, JavaScript (ES modules)          | Application structure, styling, and logic.                        |
| Libraries      | WGo.js                                      | Board rendering and SGF parsing.                                  |
|                | JSZip                                       | Reading zipped tsumego archives client-side.                      |
|                | math.js                                     | Matrix algebra for the client-side model fits.                    |
|                | Sentry (browser)                            | Error reporting.                                                  |
| Storage        | localStorage + IndexedDB                    | Recent attempt history and models; the older history tail.        |
| Build          | Node.js, npm, esbuild                       | Dependency management and bundling.                               |
| Deployment     | Wrangler (Cloudflare Workers)               | Build artefacts deployed manually via `npm run deploy`.           |
| Backend        | Cloudflare Worker (`src/`)                  | Write endpoints, account sync, leaderboard pages, cron.           |
|                | Cloudflare D1 (`tsumego-reports`)           | Attempt log, push subscriptions, schedules, surveys, leaderboard. |
|                | Web Push (VAPID)                            | Recall reminders, delivered from the cron handler.                |

The three `lib/` bundles are loaded as classic scripts from `index.html` and used as globals; everything under `js/` is an ES module bundled by esbuild.

## 3. Project structure

```
/
├── dist/                    # Build output (auto-generated, gitignored)
├── js/                      # Front-end source modules (ES modules, bundled)
│   ├── main.js              # Entry point, boot sequence, event wiring, mode switching
│   ├── state.js             # Centralized state with setter functions + persistence
│   ├── state-epoch.js       # State-epoch gate (replaces migration ladders)
│   ├── persist.js           # localStorage read/write helpers
│   ├── history-store.js     # IndexedDB cold tail of answerHistory
│   ├── constants.js         # Shared constants (incl. APP_MODES, window lengths)
│   │
│   ├── game-logic.js        # Solving-mode click handling and correctness checks
│   ├── analysis.js          # Analysis-mode logic
│   ├── board.js             # WGo.js abstraction layer
│   ├── sgf.js               # SGF parsing, win-rate/score extraction, reorientation
│   ├── coord-input.js       # Keyboard coordinate entry (accessibility path)
│   ├── problem-loader.js    # Shard loading, candidate selection, recall gate
│   │
│   ├── irt.js               # The shared 4PL response curve
│   ├── rank-decomposition.js # Curve application, career decomposition, convergence test
│   ├── rank-translation.js  # Translation to other rating systems
│   ├── recall-model.js      # Retention model, recall schedule, recall serve gate
│   ├── motif-readings.js    # Per-motif readings as a pure function of attempts + rank
│   ├── motif-stats.js       # ...bound to the live history, rank, axis count and curve
│   ├── weakness.js          # Flagged serve directions from those readings
│   ├── optimizer.js         # Damped Newton step, used by the recall fit
│   ├── math-utils.js        # Numeric helpers
│   │
│   ├── attempt-recorder.js  # Per-attempt record and reporting
│   ├── account-sync.js      # Pull-based account sync + offline outbox
│   ├── backfill.js          # Version-gated re-derivation of stored history fields
│   ├── picked-embedding.js  # Played-move motif vector fetch (Range into the sidecar)
│   ├── data-version.js      # Inlined embedding_meta.json → DATA_VERSION
│   ├── population-constants.js # Inlined population constants
│   ├── net.js, api.js       # Fetch helpers; server fallback for analysis
│   │
│   ├── ui.js, overlays.js   # DOM updates; transient informational overlays
│   ├── stats-ui.js          # Detailed-stats modal: tabbed panes
│   ├── stats-compute.js     # Stats aggregation shared by the panes
│   ├── stats-rank-pane.js, stats-weaknesses.js, stats-community.js
│   ├── stats-time-chart.js, stats-activity-chart.js, chart-svg.js
│   ├── motif-bubble.js      # Review-time motif surfacing
│   ├── onboarding.js, audio.js, i18n.js, share.js
│   ├── notifications.js     # Push opt-in + subscription management
│   ├── account-ui.js, rank-survey-prompt.js, registration.js, invite-access.js
│   ├── gor-refresh.js       # Weekly EGD rating heartbeat (client-side; see CLAUDE.md)
│   └── corrections-notice.js, debug-log.js
│
├── shared/                  # Modules bundled into BOTH the client and the Worker
│   ├── rank-math.js         # The single internal rank scale: interp, labels, bands
│   ├── rank-window.js       # The window rank estimator and per-attempt flatten
│   ├── sync-digest.js       # Row hashing for the sync digest
│   └── flag.js              # Country code → flag
│
├── src/                     # Cloudflare Worker
│   ├── index.js             # Thin router over the write endpoints + scheduled() cron
│   ├── http.js              # CORS + JSON response helpers
│   ├── guards.js            # id/user validation, per-IP rate limiting, maintenance freeze
│   ├── handlers/            # One module per route, NAMED FOR THE ROUTE (report,
│   │                        #   report-backfill, sync, push, schedule, rank-survey,
│   │                        #   share-event, profile, register, leaderboard-page,
│   │                        #   tournament-page, invite-tool)
│   ├── leaderboard.js       # Leaderboard buffer maintenance + row resolution
│   ├── attempt-insert.js    # The shared INSERT (maintains the per-profile digest)
│   ├── server-stats.js      # /server-stats compute + cache
│   ├── recall-notifications.js  # The Web Push recall cron
│   ├── aga.js               # AGA rating refresh
│   └── version.js           # Build-stamped APP_VERSION (written by build.mjs)
│
├── migrations/              # Dated D1 migrations, applied manually
├── lib/                     # Bundled third-party libs (WGo.js, JSZip, math.js).
│                            #   Copied wholesale into dist/ and SHIPPED, so keep
│                            #   non-runtime files out; WGo.js's own API reference
│                            #   is docs/vendor/WGo.js.html.
├── sound/                   # UI sound effects
├── corners/                 # Ink-wash corner art
├── tsumegos/                # Published data (gitignored — written by package_for_pwa)
│   ├── <size>/              #   shard zips, per-cell sidecars, metadata.json
│   ├── problem_params.bin   #   rating/slope/floor by id
│   ├── lookup.embeddings.bin#   correct-move motif vectors, lookup-line-aligned
│   └── embedding_meta.json  #   K, versions, motif labels (inlined at build time)
├── lookup.txt               # Global "id zipHash size" index (gitignored)
├── build.mjs                # esbuild build script
├── deploy.mjs               # Clean-tree check + build + wrangler deploy
├── wrangler.toml            # Worker config, D1 binding, rate limiters, MAINTENANCE_MODE
├── schema.sql               # D1 schema (idempotent)
├── index.html               # Single HTML page
├── about.html, changelog.html, privacy.html
├── style.css                # Styles
├── sw.js                    # Service Worker
└── manifest.json            # PWA manifest
```

## 4. Architecture

### 4.1. State management (`js/state.js`)

A single mutable state object, mutated only through exported setter functions.

`answerHistory` is **one unified cross-size array**, date-ordered, each entry tagged with its board size. Rank, the recall model and the weakness model are likewise single shared values — from the user's perspective 9×9 and 13×13 are one rank and one stats page — while histories, rank curves and problem pools stay per-size.

Storage is **split**: the most recent attempts are held as a synchronous localStorage array (the working set, sized to the largest hot window any synchronous consumer reads), and everything older is evicted to IndexedDB by `js/history-store.js`. Every model reads the working set; only the full-career charts await the tail. IndexedDB being unavailable degrades to working-set-only rather than breaking play.

There are **no migration ladders**. `js/state-epoch.js` runs at import time: a stale or absent epoch wipes the account-derived localStorage (keeping user id, board size, audio and onboarding flags) and routes boot through the ordinary account-adoption path with the device's own account, which re-pulls the history from the server. A breaking state change is shipped by bumping the epoch.

### 4.2. Application modes

`state.appMode` (defined in `js/constants.js` as `APP_MODES`):

1. **`'solving'`** — default. Clicks evaluated by `js/game-logic.js`; failure transitions automatically into analysis.
2. **`'analyzing'`** — entered after a failure or via the Review button. Logic in `js/analysis.js`. Best-move and score come from comments embedded in the SGF (`js/sgf.js`); if missing, `js/analysis-api.js` calls the server fallback.
3. **`'linked'`** — entered when the user opens a URL with a specific problem id. Rank and history are not affected, and the `?id` stays in the address bar so a reload re-enters the same shared problem.

### 4.3. SGF format

Each branch comment in a problem SGF is `"<winrate>%/<score>/<visits>"` (e.g. `"86.4%/3.1/1000"`; the visit count is a later addition, and comments predating it carry only the first two fields). `js/sgf.js` reads this to identify the correct move without needing a "Correct" marker. The starting position is tagged with a `"Q"` comment. The module also reorients recalls randomly so a repeat can't be recognised by its visual pattern; novel and linked problems are served at the identity orientation, which the share-link and picked-move contracts depend on.

### 4.4. Problem selection (`js/problem-loader.js` + `js/recall-model.js`)

The displayed problem lives in three slots on `state`: `challengeProblem` (the always-assigned standing challenge, persisted per-size), `currentProblem` (what's on screen), and `previousProblem` (the last completed problem, kept reviewable). `advanceToNextProblem()` picks what to show next:

1. **Due recall** — `getRecallCandidates()` runs the retention model over recent history. A problem's stability grows with each success since its most recent failure, and a candidate is due when its predicted retention falls inside the recall window; the lowest-retention candidate wins. There is deliberately **no ceiling on stability** — graduation out of the pool is the interval outgrowing the history window.
2. **The recall gate** — a due recall only wins the serve decision with probability `recallServeShare()`, the share that holds recall volume at equilibrium for the user's own practice volume and memory. The roll is deterministic in the candidate and the last attempt time, so a tab-resume re-derives the same decision instead of re-rolling. It fails open: while the share isn't estimable, or on any estimator error, due recalls always win.
3. **Standing challenge** — otherwise the persisted `challengeProblem` is shown. `assignNewChallenge()` selects a fresh one: exclude the recent history window, sort the remaining pool by closeness to the user's **serve rank** (the estimate plus an uncertainty-scaled bonus), take the 50 nearest, and pick the single best under **one drawn criterion** (never a blend, never a proportional sample). The draw has one slot per flagged weakness — `weaknessServeCategories()`, argmaxing that axis tail's loadings — plus one **neutral** slot (no criterion — the pool's nearest-to-rank head) that is always present, so no motif takes more than half the serves. Rank precision decides one thing: while the estimate is still converging an extra **information** slot joins the draw, argmaxing `quality` (the 4PL Fisher information at the 80%-clear anchor, `slope² · anchorInfo(floor)`); once converged it drops out. The pick is hashed from the last attempt, so a resumed tab re-derives it. Solving the challenge moves it to `previousProblem` and reselects immediately, so an already-answered problem can't be re-served.

When the candidate pool is thin, or an unloaded shard sits nearer the serve rank, `backgroundLoadCloserZip()` proactively fetches it so the next selection has fresher options.

### 4.5. Board interaction (`js/board.js`)

Wraps WGo.js for rendering, stone placement, markers, and analysis hints. Routes click events to `game-logic.js` or `analysis.js` based on `state.appMode`. Implements a "ghost stone" double-click confirmation for mobile.

### 4.6. The client-side models

Everything rank-shaped is derived on the device from the published raw numbers; the server ships no ranks.

- **Rank** (`shared/rank-window.js`, `js/rank-decomposition.js`, `js/irt.js`) — the displayed rank is a positionally-weighted 4PL maximum-likelihood fit over the recent first-time history, a **pure function** of that history, so two devices that have seen the same attempts agree regardless of how they were batched. Each attempt enters with its own problem's slope and floor, so a sharp problem pins the estimate harder. `serveRank` adds an uncertainty bonus that only problem selection reads.
- **Rank labels** (`shared/rank-math.js`) — one internal scale, AGA-style: 0 is the 1k/1d boundary, every rank is one unit wide, integers are band edges. Shared verbatim with the Worker so the server-rendered rank histogram cannot drift from the client's display.
- **Retention** (`js/recall-model.js`) — the spaced-repetition model, with population constants fitted offline and per-user terms fitted on device.
- **Weakness** (`js/weakness.js`, `js/stats-weaknesses.js`) — a logistic over the motif axes with a coefficient pair per axis (the position's correct move, and the move the user actually played), under a per-axis prior shipped in the population constants. Selection consumes the contrast between the two; the Weaknesses pane runs two-level detection over the posterior under an expected-false-flag budget.

### 4.7. Account sync (`js/account-sync.js`)

The 8-character user id doubles as the account credential: entering it on another device adopts that account. `tsumego_reports` is append-only, so sync is a pull against an **autoincrement-id cursor** — arrival order, so a row inserted with a historical timestamp (an offline flush) is still pulled. The local history is a **mirror by row id**: a row is "had" iff its id is present, and a complete pull converges in both directions, merging what's missing and deleting what the server no longer has.

There is **no polling**. Every `/report` response carries the per-profile attempt count and a running digest; a count that jumps past what this device expects means another device wrote, and triggers a pull. Divergence in either count or digest escalates to a full mirror reconcile. Attempts recorded while offline sit in a durable outbox and are uploaded later with their original event time.

### 4.8. Data files and versioning

`build.mjs` writes `src/version.js` so the Worker and the client share one `APP_VERSION`, and folds a hash of every inlined data file into it. Every write response stamps the version; a mismatch sets a pending reload that fires between problems, never mid-review.

Served data is cache-busted by a separate content-addressed `DATA_VERSION` rather than by `APP_VERSION`, because `tsumegos/` is gitignored and can be re-shipped from an unchanged commit. `embedding_meta.json` is **inlined into the bundle** at build time (`js/data-version.js`), so the manifest can never be served stale; shard archives and sidecars are content-addressed by filename. `js/backfill.js` re-derives stored history fields when the ratings version, the rank-curve version or the motif version changes, and writes its version stamps only after every size's walk has persisted, so an interrupted run simply repeats.

### 4.9. Cloudflare Worker (`src/`)

`src/index.js` is a thin router over a route table plus a `scheduled()` cron handler; handlers and aggregation live in sibling modules. Beyond the two headline responsibilities below it serves account sync (`/sync`), the offline outbox (`/report-backfill`), Web Push (`/push/*`), the recall schedule (`/schedule`), surveys and analytics (`/rank-survey`, `/share-event`), profile visibility (`/profile/hide`), registration (`/register`), the cached community stats (`/server-stats`), and the server-rendered `/leaderboard` and `/tournament` pages. The cron delivers due recall reminders and refreshes the cached stats and the external rating heartbeats. `CLAUDE.md` carries the full route and D1-schema reference.

- **`POST /report`** — validates and inserts an attempt row into the D1 `tsumego_reports` table (binding: `DB`), returning the row id, the per-profile attempt count and the sync digest. `OPTIONS` returns CORS headers.
- **SPA fallback** — for navigation requests that don't match a static asset, serves `index.html` while preserving the URL so the front-end can read the size hint (`/9`, `/13`) from `window.location.pathname`. Non-navigation misses return a real 404 so a stale Service Worker cache can invalidate cleanly.

Write endpoints are guarded cheapest-first, all upstream of D1: a maintenance freeze (`MAINTENANCE_MODE` in `wrangler.toml`, for database surgery), per-IP rate-limit bindings, and in-handler format validation of the identity fields. An outer Cloudflare dashboard rate-limiting rule sits in front of the Worker and is account configuration, not repository configuration.

## 5. Development workflow

### 5.1. Initial setup

```bash
npm install
```

### 5.2. Local testing

Use Wrangler — it builds the bundle, serves the static assets, and runs the Worker (so the write endpoints and the SPA fallback behave like production):

```bash
npm run dev                # watch build + wrangler dev → http://localhost:8787
npm run watch              # watch build only
```

Opening `index.html` directly will not work; the app must be built and served. The local D1 is reset to the remote state by `pipeline.fetch_database`, which overwrites the miniflare database file — stop `npm run dev` before running it.

### 5.3. Schema changes

`schema.sql` is idempotent (`CREATE … IF NOT EXISTS`) and `migrations/` holds dated one-off migrations. Both are applied **manually**, and a migration that adds columns a new handler reads must be applied to the remote D1 *before* the deploy that reads them:

```bash
npx wrangler d1 execute tsumego-reports --remote --file=schema.sql
npx wrangler d1 execute tsumego-reports --remote --file=migrations/<file>.sql
```

### 5.4. Deployment

Deployment is **manual** — there is no git-triggered auto-deploy. `deploy.mjs` checks for a clean git tree, builds, then runs `wrangler deploy`:

```bash
npm run deploy             # build + wrangler deploy
```

This publishes both the static assets (via the `[assets]` binding in `wrangler.toml`) and the Worker code. A data re-ship additionally requires running `pipeline.package_for_pwa` first, and a re-rate requires refreshing the leaderboard ranks afterwards — see `CLAUDE.md` for that ordering, which is load-bearing.
