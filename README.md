# YoseTsumego / onlymove.net

This project generates Go (Baduk / Weiqi) life-and-death problems with KataGo, rates them against a stable of bots, and serves the result through a PWA at [onlymove.net](https://onlymove.net). Both 9×9 and 13×13 are supported.

## Project structure

- **`pipeline/`** — executable scripts that form the core workflow (generation, benchmarking, ranking, packaging).
- **`utils/`** — core logic, shared helpers, and the KataGo controller.
- **`auxiliary/`** — analysis tools, diagnostic scripts, and legacy code.
- **`pwa/`** — the front-end (vanilla JS + esbuild) and the Cloudflare Worker that hosts it.
- **`analysis_server/`** — a small KataGo-backed HTTP service used as the PWA's analysis fallback.
- **`vast_orchestration.py`** — wrapper for running the heavy pipeline steps on a rented Vast.ai GPU.

## Board size

`BOARD_SIZE` in `utils/configurations.py` is a single global switch (`9` or `13`). All pipeline outputs go under size-keyed subdirectories (`tsumegos/<size>/`, `attempts/<size>/`, `ratings/<size>/`, `pwa/tsumegos/<size>/`). To refresh both modes, run the pipeline once with `BOARD_SIZE = 9`, then again with `BOARD_SIZE = 13`.

## Pipeline execution

Run each step through the venv's Python so imports resolve:

### 1. Fetch database
Downloads the latest tsumego attempt data from Cloudflare D1.
```bash
./.venv/bin/python3 -m pipeline.fetch_database
```

### 2. Tune engine
Optimizes KataGo batch sizes for your hardware (rewrites `bots/analysis.cfg` in place).
```bash
./.venv/bin/python3 -m pipeline.tune_engine
```

### 3. Generate tsumegos
Continuously generates new tsumego positions.
```bash
./.venv/bin/python3 -m pipeline.generate_tsumegos
```

### 4. Benchmark bots
Runs various bots against the generated tsumegos to collect performance data.
```bash
./.venv/bin/python3 -m pipeline.benchmark_bots
```

### 5. Rank tsumegos and bots
Trains the rating model based on the benchmark results.
```bash
./.venv/bin/python3 -m pipeline.rank_tsumegos_and_bots
```

### 6. Package for PWA
Groups tsumegos and ratings into ZIP files for the web app.
```bash
./.venv/bin/python3 -m pipeline.package_for_pwa
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
npm run dev         # build + wrangler dev
```

### Deploy

Deployment is manual:

```bash
npm run deploy      # build + wrangler deploy
```
