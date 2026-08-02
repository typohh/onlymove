# YoseTsumego / onlymove.net

This project generates Go (Baduk / Weiqi) life-and-death problems with KataGo, rates each by difficulty using a model trained on human attempts (with bot benchmarks and KataGo neural-network features as inputs), and serves the result through a PWA at [onlymove.net](https://onlymove.net). Both 9×9 and 13×13 are supported.

Each problem has exactly one correct first move. From the attempt log the pipeline fits three things per problem — a **rating** (with a discrimination slope and a guessing floor), the **rank** that rating maps to on the kyu/dan ladder, and a **motif** vector describing what kind of skill the problem exercises — and publishes them as raw numbers. Everything the player sees (their live rank, which problem comes next, the spaced-repetition schedule, the weakness readouts) is computed on their own device from those numbers.

## Project structure

- **`pipeline/`** — executable scripts that form the core workflow (generation, benchmarking, rating, feature extraction, packaging).
- **`utils/`** — core logic, shared helpers, and the KataGo controller.
- **`auxiliary/`** — analysis tools, rating-model research, and diagnostic scripts (not part of the main flow).
- **`pwa/`** — the front-end (vanilla JS + esbuild) and the Cloudflare Worker that hosts it.
- **`analysis_server/`** — a small KataGo-backed HTTP service used as the PWA's analysis fallback.
- **`android/`** — the Bubblewrap TWA shell that wraps the PWA for the Play Store.
- **`docs/`** — methodology and operational runbooks.
- **`vast_orchestration.py`** — wrapper for running the heavy pipeline steps on a rented Vast.ai GPU.

Three documents go deeper: **`docs/onlymove.md`** is the self-contained methodology — how ratings, ranks and motifs are constructed and how the client consumes them, including which alternatives were tried and lost; **`CLAUDE.md`** documents the subsystems and the cross-file invariants; **`pwa/README.md`** covers the front-end and the Worker.

## Board size

Both 9×9 and 13×13 ship in production (`ALL_BOARD_SIZES = (9, 13)` in `utils/configurations.py`); there is no single global `BOARD_SIZE` switch. The three single-size generation steps — `tune_engine`, `generate_tsumegos`, `benchmark_bots` — take a **mandatory positional `size` arg** (`9` or `13`). The downstream steps (`extract_activations` onward) iterate both sizes in one run and take no size arg. All pipeline outputs go under size-keyed subdirectories (`tsumegos/<size>/`, `attempts/<size>/`, `ratings/<size>/`, `pwa/tsumegos/<size>/`). To refresh both modes, run each generation step once per size (e.g. `generate_tsumegos 9` then `generate_tsumegos 13`), then run the downstream steps once.

## Pipeline execution

Run each step through the venv's Python so imports resolve.

### 1. Fetch database
Pulls the human attempt log and the rank surveys from Cloudflare D1 into `attempts/database.{sqlite,sql}`, and resets the local wrangler D1 to match. Incremental by default (new `tsumego_reports` rows only, by row id); `--full` forces a complete re-download. Stop `npm run dev` before running it.
```bash
./.venv/bin/python3 -m pipeline.fetch_database
```

### 2. Tune engine
Optimizes KataGo batch sizes for the current hardware (rewrites `bots/analysis.cfg` in place). Takes a mandatory `size` arg (`9` or `13`).
```bash
./.venv/bin/python3 -m pipeline.tune_engine <size>
```

### 3. Generate tsumegos
Continuously generates new tsumego positions into `tsumegos/<size>/`. Takes a mandatory `size` arg (`9` or `13`).
```bash
./.venv/bin/python3 -m pipeline.generate_tsumegos <size>
```

### 4. Benchmark bots
Runs a stable of bots against the generated tsumegos to collect performance data into `attempts/<size>/`. Takes a mandatory `size` arg (`9` or `13`).
```bash
./.venv/bin/python3 -m pipeline.benchmark_bots <size>
```

### 5. Extract activations
Captures KataGo trunk activations at every legal cell into `activations/<size>/<id>.npz` (GPU; resumable). Feeds both the rating features and the motif axes.
```bash
./.venv/bin/python3 -m pipeline.extract_activations
```

### 6. Cache move features
Builds the candidate-move-eval cache the rating trainer reads.
```bash
./.venv/bin/python3 -m pipeline.cache_move_features
```

### 7. Train rating
Fits the cross-board rating model on human first-time attempts (activation, bot-winrate, move-eval and game-phase features) into `ratings/<size>/`. Takes no arguments — the shipped configuration is baked in; the experiment surface lives in `auxiliary/rating_dev/`.
```bash
./.venv/bin/python3 -m pipeline.train_rating
```

### 8. Map rating to rank
Fits the rating→rank curve and the per-server couplings against the rank surveys, into `ratings/rating_to_rank.json` and `ratings/rank_map.json`. Board-independent and needs no GPU, but it sits *inside* the sequence: the axes step reads the curve.
```bash
./.venv/bin/python3 -m pipeline.map_rating_to_rank
```

### 9. Build axes
Fits the supervised residual skill axes (the motifs) into `embeddings/joint/`, selecting the axis count from held-out gain. Consumes the rating outputs and the rank curve, so it follows both preceding steps.
```bash
./.venv/bin/python3 -m pipeline.build_axes
```

### 10. Fit population constants
Fits the retention and time-bias constants and the per-axis weakness prior into `ratings/population_constants.json`. It fits one prior entry per axis, so it must follow `build_axes` — packaging refuses to run on a stale prior vector.
```bash
./.venv/bin/python3 -m pipeline.fit_population_constants
```

### 11. Package for PWA
The single writer of `pwa/tsumegos/`: bundles tsumegos, ratings, motifs and the curve into the archives and sidecars the web app serves. Change-detects per size, so a curve-only or labels-only change publishes without rebuilding shards.
```bash
./.venv/bin/python3 -m pipeline.package_for_pwa
```

Steps 7–11 are ordered by hard data dependencies rather than convention; `docs/onlymove.md` §3.1 spells out each edge and what breaks if it is skipped.

### Remote execution

The heavy generation steps can run on a rented Vast.ai GPU:

```bash
python vast_orchestration.py                    # no args prints subcommands
python vast_orchestration.py run <step> <size>  # tune / generate / benchmark are remoted (size forwarded); the rest stay local
```

## PWA

The web app lives in `pwa/` and is deployed as a Cloudflare Worker (with a D1 database for attempt logging, account sync, and the leaderboard). See `pwa/README.md` for the front-end and Worker architecture.

### Build

```bash
cd pwa
npm install
npm run build       # esbuild → pwa/dist/
```

### Local dev

```bash
npm run dev         # watch build + wrangler dev → http://localhost:8787
```

Local testing must go through `npm run dev`: the Worker runs the write endpoints and the SPA fallback (URL-based size hints), so opening the static bundle alone will misbehave.

### Deploy

Deployment is manual — there is no git-triggered auto-deploy:

```bash
npm run deploy      # build + wrangler deploy
```
