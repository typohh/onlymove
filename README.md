# YoseTsumego / onlymove.net

This project generates Go (Baduk / Weiqi) life-and-death problems with KataGo, rates each by difficulty using a model trained on human attempts (with bot benchmarks and KataGo neural-network features as inputs), and serves the result through a PWA at [onlymove.net](https://onlymove.net). Both 9×9 and 13×13 are supported.

## Project structure

- **`pipeline/`** — executable scripts that form the core workflow (generation, benchmarking, rating, feature extraction, packaging).
- **`utils/`** — core logic, shared helpers, and the KataGo controller.
- **`auxiliary/`** — analysis tools, rating-model research, and diagnostic scripts (not part of the main flow).
- **`pwa/`** — the front-end (vanilla JS + esbuild) and the Cloudflare Worker that hosts it.
- **`analysis_server/`** — a small KataGo-backed HTTP service used as the PWA's analysis fallback.
- **`vast_orchestration.py`** — wrapper for running the heavy pipeline steps on a rented Vast.ai GPU.

For deeper architecture, `CLAUDE.md` documents the subsystems and cross-file invariants, and `pwa/README.md` covers the front-end.

## Board size

Both 9×9 and 13×13 ship in production (`ALL_BOARD_SIZES = (9, 13)` in `utils/configurations.py`); there is no single global `BOARD_SIZE` switch. The three single-size generation steps — `tune_engine`, `generate_tsumegos`, `benchmark_bots` — take a **mandatory positional `size` arg** (`9` or `13`). The downstream steps (`extract_activations` onward) iterate both sizes in one run and take no size arg. All pipeline outputs go under size-keyed subdirectories (`tsumegos/<size>/`, `attempts/<size>/`, `ratings/<size>/`, `pwa/tsumegos/<size>/`). To refresh both modes, run each generation step once per size (e.g. `generate_tsumegos 9` then `generate_tsumegos 13`), then run the downstream steps once.

## Pipeline execution

Run each step through the venv's Python so imports resolve.

### 1. Fetch database
Pulls the latest tsumego attempts and rank surveys from Cloudflare D1 into `attempts/`.
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
Captures KataGo trunk activations per problem into `activations/<size>/` (GPU; resumable). Feeds both the rating model and the embedding recommender.
```bash
./.venv/bin/python3 -m pipeline.extract_activations
```

### 6. Cache move features
Builds the candidate-move-eval cache the rating trainer reads.
```bash
./.venv/bin/python3 -m pipeline.cache_move_features
```

### 7. Train rating
Fits the cross-board rating model on human attempts (using activation, bot-winrate, and move-eval features) into `ratings/<size>/`.
```bash
./.venv/bin/python3 -m pipeline.train_rating
```

### 8. Build embedding
Fits the joint embedding over activations for the per-component weakness recommender into `embeddings/`.
```bash
./.venv/bin/python3 -m pipeline.build_embedding
```

### 9. Package for PWA
The single writer of `pwa/tsumegos/`: bundles tsumegos, ratings, and embeddings into the ZIP archives the web app serves.
```bash
./.venv/bin/python3 -m pipeline.package_for_pwa
```

### Rank calibration (board-independent)

These two steps are board-independent and run outside the in-order pipeline. They need `ratings/` and the `attempts/` database (no GPU), and write only to the backend `ratings/`; `package_for_pwa` then publishes their outputs into `pwa/tsumegos/`.

```bash
./.venv/bin/python3 -m pipeline.map_rating_to_egf          # survey-anchored rating→rank curve
./.venv/bin/python3 -m pipeline.fit_population_constants   # retention / time-bias / weakness-ridge constants
```

### Remote execution

The heavy generation steps can run on a rented Vast.ai GPU:

```bash
python vast_orchestration.py                  # no args prints subcommands
python vast_orchestration.py run <step> <size>  # tune / generate / benchmark are remoted (size forwarded); the rest stay local
```

## PWA

The web app lives in `pwa/` and is deployed as a Cloudflare Worker (with a D1 database for attempt logging). See `pwa/README.md` for the front-end architecture.

### Build

```bash
cd pwa
npm install
npm run build       # esbuild → pwa/dist/
```

### Local dev

```bash
npm run dev         # build + wrangler dev → http://localhost:8787
```

Local testing must go through `npm run dev`: the Worker runs the `/report` endpoint and the SPA fallback (URL-based size hints), so opening the static bundle alone will misbehave.

### Deploy

Deployment is manual — there is no git-triggered auto-deploy:

```bash
npm run deploy      # build + wrangler deploy
```
