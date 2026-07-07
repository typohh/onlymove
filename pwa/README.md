# Tsumego SPA

## 1. Overview

The mobile-first Single Page Application served at [onlymove.net](https://onlymove.net) for practising Go (Baduk / Weiqi) life-and-death problems. Both 9×9 and 13×13 corpora are supported; the size is selected from the URL path (`/`, `/9`, or `/13`).

The app picks problems matched to the user's estimated rank, adjusts that rank in real time based on outcomes, and runs a per-user spaced-repetition model to surface previously-failed problems at the right cadence. When a user fails a problem they enter Analysis mode, where variations and best-move/score data are read from the SGF itself; if the SGF lacks that data, the front-end calls a server-side KataGo fallback.

The codebase is vanilla JavaScript (ES modules), bundled with esbuild and shipped as a Cloudflare Worker.

## 2. Technology stack

| Category       | Technology                                  | Purpose                                                           |
|----------------|---------------------------------------------|-------------------------------------------------------------------|
| Frontend       | HTML, CSS, JavaScript (ES modules)          | Application structure, styling, and logic.                        |
| Libraries      | WGo.js                                      | Board rendering and SGF parsing.                                  |
|                | JSZip                                       | Reading zipped tsumego archives client-side.                      |
|                | Sentry (browser)                            | Error reporting.                                                  |
| Build          | Node.js, npm, esbuild                       | Dependency management and bundling.                               |
| Deployment     | Wrangler (Cloudflare Workers)               | Build artefacts deployed manually via `npm run deploy`.           |
| Backend        | Cloudflare Worker (`src/index.js`)          | `/report` endpoint + SPA fallback for navigation requests.        |
|                | Cloudflare D1 (`tsumego-reports`)           | Stores per-attempt rows submitted by clients.                     |

## 3. Project structure

```
/
├── dist/                    # Build output (auto-generated, gitignored)
├── js/                      # Front-end source modules
│   ├── main.js              # Entry point, event wiring, mode switching
│   ├── migrations.js        # One-shot answerHistory-shape migrations (runMigrations)
│   ├── game-logic.js        # Solving-mode click handling and correctness checks
│   ├── analysis.js          # Analysis-mode logic
│   ├── problem-loader.js    # Zip loading, candidate selection, recall fallback
│   ├── model.js             # Per-user logistic-regression recall model + stats computation
│   ├── stats-ui.js          # Detailed-stats modal: tabbed panes + SVG charts
│   ├── user-stats.js        # Rank tracking and attempt reporting
│   ├── board.js             # WGo.js abstraction layer
│   ├── state.js             # Centralized state with setter functions
│   ├── sgf.js               # SGF parsing, win-rate/score extraction, reorientation
│   ├── api.js               # Server fallback for analysis
│   ├── ui.js                # DOM updates
│   ├── overlays.js          # Transient informational overlays
│   ├── audio.js              # UI sounds
│   ├── i18n.js              # Localization
│   └── constants.js         # Shared constants (incl. APP_MODES)
├── shared/
│   └── rank-math.js         # Rank interp/label helpers shared by worker + client
├── src/                     # Cloudflare Worker: index.js router + scheduled()
│   ├── index.js             # thin router + cron; http/guards + handlers/ + server-stats + recall-notifications alongside
│   ├── http.js              # CORS + JSON response helpers
│   ├── guards.js            # id/user validation + per-IP rate limiting
│   ├── handlers/            # one module per route (report, sync, push, schedule, …)
│   ├── server-stats.js      # /server-stats compute + cache
│   └── recall-notifications.js  # the Web Push recall cron
├── lib/                     # Bundled third-party libs (WGo.js, JSZip)
├── sound/                   # UI sound effects
├── tsumegos/<size>/         # Per-size zip archives + metadata.json
├── lookup.txt               # Global "id zipHash size" index for linked problems
├── build.mjs                # esbuild build script
├── wrangler.toml            # Worker config + D1 binding
├── schema.sql               # D1 schema
├── index.html               # Single HTML page
├── style.css                # Styles
├── sw.js                    # Service Worker
└── manifest.json            # PWA manifest
```

## 4. Architecture

### 4.1. State management (`js/state.js`)

A single mutable state object, mutated only through exported setter functions. Persisted to `localStorage` so sessions survive reloads.

### 4.2. Application modes

`state.appMode` (defined in `js/constants.js` as `APP_MODES`):

1. **`'solving'`** — default. Clicks evaluated by `js/game-logic.js`; failure transitions automatically into analysis.
2. **`'analyzing'`** — entered after a failure or via the Review button. Logic in `js/analysis.js`. Best-move and score come from comments embedded in the SGF (`js/sgf.js`); if missing, `js/api.js` calls the server fallback.
3. **`'linked'`** — entered when the user opens a URL with a specific problem id. Rank and history are not affected.

### 4.3. SGF format

Each branch comment in a problem SGF is `"<winrate>%/<score>"` (e.g. `"86.4%/3.1"`). `js/sgf.js` reads this to identify the correct move without needing a "Correct" marker. The starting position is tagged with a `"Q"` comment. The module also reorients problems randomly to add variety.

### 4.4. Problem selection (`js/problem-loader.js` + `js/model.js`)

The displayed problem lives in three slots on `state` (see the challenge-model migration in `state.js`): `challengeProblem` (the always-assigned standing challenge, persisted per-size), `currentProblem` (what's on screen), and `previousProblem` (the last completed problem, kept reviewable). `advanceToNextProblem()` picks what to show next:

1. **Due recall** — `getRecallCandidates()` runs a per-user logistic regression (Adam-trained, persisted) over recent history. Features: solving time, review time, problem difficulty, novelty, and a recall-formula term `log(1 + t · 0.5^r)` where `t` is days since last attempt and `r` is successes since the most recent failure. Problems whose predicted recall probability falls in `[0.60, 0.80]` are candidates; the lowest-probability one wins. The model is clipped so predicted "due" intervals stay within 6 h – 7 d. If one is due it is served as a recall solve.
2. **Standing challenge** — otherwise the persisted `challengeProblem` is shown. `assignNewChallenge()` selects a fresh one: exclude the recent history window, sort the remaining pool by closeness to the user's rank, take the 50 nearest, and weighted-sample by the problem's `quality` (∝ aₚ²). Solving the challenge moves it to `previousProblem` and reselects a new challenge immediately, so an already-answered problem can't be re-served.

When the candidate pool is thin, `backgroundLoadCloserZip()` proactively fetches the closest unloaded archive so the next selection has fresher options.

### 4.5. Board interaction (`js/board.js`)

Wraps WGo.js for rendering, stone placement, markers, and analysis hints. Routes click events to `game-logic.js` or `analysis.js` based on `state.appMode`. Implements a "ghost stone" double-click confirmation for mobile.

### 4.6. Cloudflare Worker (`src/index.js` + `src/`)

`src/index.js` is a thin router (a route table over the rate-limited write endpoints) plus a `scheduled()` cron handler; the handlers and aggregation live in sibling modules (`src/http.js`, `src/guards.js`, `src/handlers/*`, `src/server-stats.js`, `src/recall-notifications.js`). Beyond the headline two responsibilities below, it also serves account sync (`/sync`), Web Push (`/push/*`), the recall schedule (`/schedule`), surveys/analytics (`/rank-survey`, `/share-event`), and the cached community stats (`/server-stats`) — see CLAUDE.md for the full route + D1-schema reference.

- **`POST /report`** — validates and inserts an attempt row into the D1 `tsumego_reports` table (binding: `DB`). `OPTIONS` returns CORS headers.
- **SPA fallback** — for navigation requests that don't match a static asset, serves `index.html` while preserving the URL so the front-end can read the size hint (`/9`, `/13`) from `window.location.pathname`. Non-navigation misses return a real 404 so a stale Service Worker cache can invalidate cleanly.

## 5. Development workflow

### 5.1. Initial setup

```bash
npm install
```

### 5.2. Local testing

Use Wrangler — it builds the bundle, serves the static assets, and runs the Worker (so `/report` and the SPA fallback behave like production):

```bash
npm run dev                # build + wrangler dev → http://localhost:8787
```

Opening `index.html` directly will not work; the app must be built and served.

### 5.3. Deployment

Deployment is **manual** — there is no git-triggered auto-deploy.

```bash
npm run deploy             # build + wrangler deploy
```

This publishes both the static assets (via the `[assets]` binding in `wrangler.toml`) and the Worker code from `src/index.js`.
