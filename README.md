# YoseTsumego / onlymove.net

This project generates Go (Baduk / Weiqi) life-and-death problems with KataGo, rates each by difficulty using a model trained on human attempts (with bot benchmarks and KataGo neural-network features as inputs), and serves the result through a PWA at [onlymove.net](https://onlymove.net). Both 9×9 and 13×13 are supported.

Each problem has exactly one correct first move. From the attempt log the pipeline fits three things per problem — a **rating** (with a discrimination slope and a guessing floor), the **rank** that rating maps to on the kyu/dan ladder, and a **motif** vector describing what kind of skill the problem exercises — and publishes them as raw numbers. Everything the player sees (their live rank, which problem comes next, the spaced-repetition schedule, the weakness readouts) is computed on their own device from those numbers.

## Project structure

- **`pipeline/`** — executable scripts that form the core workflow (generation, benchmarking, rating, feature extraction, packaging).
- **`utils/`** — the shared logic, split by domain: **`utils/katago/`** is the Go and KataGo side (engine controller and queries, the vendored PyTorch copy, coordinates, SGF construction, problem finding, board rendering) and **`utils/rating/`** is the statistical side (feature assembly, the rating artifacts reader, rank-curve fitting, surveys, the residual-axes model). The two families share no imports; `utils/constants.py` holds the handful of values both need.
- **`auxiliary/`** — analysis tools, rating-model research, and diagnostic scripts (not part of the main flow).
- **`pwa/`** — the front-end (vanilla JS + esbuild) and the Cloudflare Worker that hosts it.
- **`analysis_server/`** — a small KataGo-backed HTTP service used as the PWA's analysis fallback.
- **`android/`** — the Bubblewrap TWA shell that wraps the PWA for the Play Store.
- **`docs/`** — methodology and operational runbooks (authored); `docs/vendor/` holds third-party reference material, currently WGo.js's API docs.
- **`data/`** — external rank reference inputs (`reference/`), corpus audit findings (`tsumego_audit/`), and gitignored real-user localStorage dumps (`user_dumps/`, which carry account credentials — see `data/README.md`).
- **`db/`** — the local snapshot of the production Cloudflare D1 (`d1.sqlite`, plus the `d1.sql` text dump beside it), pulled by `pipeline.fetch_database`. Gitignored and regenerable; query `d1.sqlite` for any local analysis. Distinct from `attempts/<size>/`, which holds the *tracked* bot benchmark results that feed the rating model.
- **`vast_orchestration.py`** — wrapper for running the heavy pipeline steps on a rented Vast.ai GPU.

Three documents go deeper: **`docs/methodology.md`** is the self-contained methodology — how ratings, ranks and motifs are constructed and how the client consumes them, including which alternatives were tried and lost; **`CLAUDE.md`** documents the subsystems and the cross-file invariants; **`pwa/README.md`** covers the front-end and the Worker.

## Prerequisites

The repository is complete as source and deliberately incomplete as a deployment: the packaged shards, the network weights, the KataGo binary and the human attempt log are all gitignored because they are generated, large, or private. Read [What a fork can run](#what-a-fork-can-run) below before installing anything — roughly half the pipeline reproduces outside this project and half does not, and it is better to know which half in advance. To understand the system rather than run it, `docs/methodology.md` is self-contained and needs no setup at all.

### Runtimes

- **Python 3.11 or newer** (developed on 3.12). NumPy and SciPy set that floor. Create a virtualenv, install `requirements.txt` into it, and invoke every pipeline step through it — `./.venv/bin/python3 -m pipeline.<step>` — so the repo-rooted absolute imports resolve.
- **Node.js 22** (the version the PWA is developed and deployed against) with npm, for the front end. Wrangler installs with the other dev dependencies and runs through `npx`; no global install.

### KataGo

- **The binary is not in the repository** — `bots/` is gitignored apart from `analysis.cfg`, which is tuned state rather than an artifact. Build or download a KataGo yourself and put it at `bots/katago`, or point the `KATAGO_PATH` environment variable at it. The default assumption is an OpenCL build; `vast_orchestration.py` uploads a CUDA build as `bots/katago_cuda` when running remotely.
- **Network weights fetch themselves.** `utils.katago.models.get_model` downloads the `.bin.gz` that the analysis engine reads on first use, so the generation steps need no manual model setup.
- **Activation extraction is the exception.** Step 5 runs the vendored PyTorch copy of KataGo, which loads a `model.ckpt` — a different artifact from the `.bin.gz`, and not derivable from it, since KataGo's own tooling converts PyTorch → binary and not back. Extract it from the matching checkpoint `.zip` to `models/kata1-b18c384nbt-s9996604416-d4316597426/model.ckpt`, or pass `--ckpt`.

### Hardware

A GPU is a practical requirement for generation, bot benchmarking and activation extraction (steps 3–5). Extraction runs about 8 GPU-minutes per 1000 problems on a Quadro M2200, so a full-corpus pass is a day of compute; `vast_orchestration.py` rents a Vast.ai instance for the three remotable steps. Everything downstream of extraction, and everything in `auxiliary/`, is CPU-only.

### Cloudflare (deployment only)

Deploying the PWA needs a Wrangler login, a D1 database created from `pwa/schema.sql`, and two secrets — `VAPID_PRIVATE_KEY` for Web Push and `ADMIN_USER_IDS` for the invite tool — set with `wrangler secret put`. A gitignored `pwa/.dev.vars` supplies the same values to `wrangler dev`. None of this is needed to read the code or to run the generation half of the pipeline.

## What a fork can run

**Steps 2–6 reproduce anywhere.** Tuning, generation, bot benchmarking, activation extraction and the feature cache need only KataGo, the weights and a GPU, and they manufacture their own inputs. That is where a fork starts.

**Step 1 belongs to this project's Cloudflare account.** `fetch_database` runs `wrangler d1 export` against the live D1 holding the human attempt log. A fork cannot authenticate to it, and the log is not published — it is real users' per-attempt answer history.

**Steps 7–10 read that log**, so the rating, rank and motif fits do not reproduce outside the project. The seam for substituting an attempt log of your own is at least a clean one — `db/d1.sqlite` is an ordinary SQLite file whose shape is fixed by `pwa/schema.sql`, and the loaders open it directly — but nobody has done it, so treat it as an opening rather than a documented path.

**`npm run build` fails on a fresh clone**, and the error is worth recognising rather than debugging. `pwa/js/data-version.js` statically imports `../tsumegos/embedding_meta.json` so the data manifest can never be served stale; `pwa/tsumegos/` is written by step 11. Until packaging has run, esbuild stops on the unresolved import.

## Board size

Both 9×9 and 13×13 ship in production (`ALL_BOARD_SIZES = (9, 13)` in `utils/constants.py`); there is no single global `BOARD_SIZE` switch. The three single-size generation steps — `tune_engine`, `generate_tsumegos`, `benchmark_bots` — take a **mandatory positional `size` arg** (`9` or `13`). The downstream steps (`extract_activations` onward) iterate both sizes in one run and take no size arg. All pipeline outputs go under size-keyed subdirectories (`tsumegos/<size>/`, `attempts/<size>/`, `ratings/<size>/`, `pwa/tsumegos/<size>/`). To refresh both modes, run each generation step once per size (e.g. `generate_tsumegos 9` then `generate_tsumegos 13`), then run the downstream steps once.

## Pipeline execution

Run each step through the venv's Python so imports resolve.

### 1. Fetch database
Pulls the human attempt log and the rank surveys from Cloudflare D1 into `db/d1.{sqlite,sql}`, and resets the local wrangler D1 to match. Incremental by default (new `tsumego_reports` rows only, by row id); `--full` forces a complete re-download. Stop `npm run dev` before running it.
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
./.venv/bin/python3 -m pipeline.cache_rating_features
```

### 7. Train the MIRT
Fits the rating and the K skill axes together, as one compensatory model on human first-time attempts (activation, bot-winrate, move-eval and game-phase features), into `ratings/<size>/` and `axes/`. Takes no arguments — the shipped configuration is baked in; the experiment surface lives in `auxiliary/rating_dev/train_joint_mirt.py`.
```bash
./.venv/bin/python3 -m pipeline.train_mirt
```

### 8. Write per-cell axis vectors
Fans the axis projection out to every legal cell into `axes/per_cell/<size>/<tid>.npz`, which the client needs because it scores the move the user picked and not only the answer. Resumable by file existence.
```bash
./.venv/bin/python3 -m pipeline.write_mirt_per_cell
```

### 9. Map rating to rank
Fits the rating→rank curve and the per-server couplings against the rank surveys, into `ratings/rating_to_rank.json` and `ratings/rank_map.json`. Board-independent and needs no GPU, but it sits *inside* the sequence: it reads the ratings the MIRT wrote.
```bash
./.venv/bin/python3 -m pipeline.map_rating_to_rank
```

### 10. Fit population constants
Fits the retention and time-bias constants and the per-axis weakness prior into `ratings/population_constants.json`. It fits one prior entry per axis, so it must follow `train_mirt` — packaging refuses to run on a stale prior vector.
```bash
./.venv/bin/python3 -m pipeline.fit_population_constants
```

### 11. Package for PWA
The single writer of `pwa/tsumegos/`: bundles tsumegos, ratings, motifs and the curve into the archives and sidecars the web app serves. Change-detects per size, so a curve-only or labels-only change publishes without rebuilding shards.
```bash
./.venv/bin/python3 -m pipeline.package_for_pwa
```

Steps 7–11 are ordered by hard data dependencies rather than convention; `docs/methodology.md` §3.1 spells out each edge and what breaks if it is skipped.

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
