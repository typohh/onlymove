# onlymove.net

This project generates Go (Baduk / Weiqi) problems with KataGo (whole-board positions from play in which exactly one move wins), rates each by difficulty using a model trained on human attempts (with bot benchmarks and KataGo neural-network features as inputs), and serves the result through a PWA at [onlymove.net](https://onlymove.net). Both 9×9 and 13×13 are supported.

The problems themselves live in a separate repository, [onlymove-data](https://github.com/typohh/onlymove-data); this one holds the code that generates, rates and serves them (see [The corpus](#the-corpus)).

Each problem has exactly one correct first move. From the attempt log the pipeline fits three things per problem — a **rating** (with a discrimination slope and a guessing floor), the **rank** that rating maps to on the kyu/dan ladder, and a **motif** vector describing what kind of skill the problem exercises — and publishes them as raw numbers. Everything the player sees (their live rank, which problem comes next, the spaced-repetition schedule, the weakness readouts) is computed on their own device from those numbers.

## Project structure

- **`pipeline/`** — executable scripts that form the core workflow (generation, benchmarking, rating, feature extraction, packaging).
- **`utils/`** — the shared logic, split by domain: **`utils/katago/`** is the Go and KataGo side (engine controller and queries, the vendored PyTorch copy, coordinates, SGF construction, problem finding, board rendering) and **`utils/rating/`** is the statistical side (feature assembly, the model's layer stack, the rating artifacts reader, rank-curve fitting, surveys, the motif rules, the SE-floor geometry). The boundary is one-way and holds at import time: `utils/katago/` imports nothing from `utils/rating/`, and importing the rating side loads nothing from the KataGo side — the few engine helpers it needs are imported inside the functions that use them. `utils/constants.py` holds the handful of values both need and imports from neither.
- **`ablations/`** — repeatable, seeded measurements against what ships (the rating fit, the rank map, the client's estimators), and the ledger of every number the documents quote. See [Ablations and the documents](#ablations-and-the-documents).
- **`auxiliary/`** — analysis tools, rating-model research, and diagnostic scripts (not part of the main flow), plus `vast_orchestration.py`, the wrapper for running the heavy pipeline steps on a rented Vast.ai GPU.
- **`pwa/`** — the front-end (vanilla JS + esbuild) and the Cloudflare Worker that hosts it.
- **`analysis_server/`** — a small KataGo-backed HTTP service used as the PWA's analysis fallback. `worker.py` lets a faster machine dial in and take that service's work over while it is running.
- **`android/`** — the Bubblewrap TWA shell that wraps the PWA for the Play Store.
- **`docs/`** — methodology and operational runbooks (authored); `docs/vendor/` holds third-party reference material, currently WGo.js's API docs.
- **`data/`** — external rank reference inputs (`reference/`), corpus audit findings (`tsumego_audit/`), and gitignored real-user localStorage dumps (`user_dumps/`, which carry account credentials — see `data/README.md`).
- **`db/`** — the local snapshot of the production Cloudflare D1 (`d1.sqlite`, plus the `d1.sql` text dump beside it), pulled by `pipeline.fetch_database`. Gitignored and regenerable; query `d1.sqlite` for any local analysis. Distinct from `attempts/<size>/`, which holds the *tracked* bot benchmark results that feed the rating model.

Where to go deeper: **`docs/overview.md`** describes what each subsystem is for and who owns what state, with no algorithms — start there. **`docs/methodology.md`** is the self-contained methodology: how ratings, ranks and motifs are constructed and how the client consumes them, including which alternatives were tried and lost. **`CLAUDE.md`** is the operating layer — the commands and the cross-file invariants — and its "Where the detail lives" table indexes the rest of `docs/`. **`pwa/README.md`** covers the front-end and the Worker, and **`docs/contract.md`** is the editing contract for all of the above.

## Prerequisites

The repository is complete as source and deliberately incomplete as a deployment: the packaged shards, the network weights, the KataGo binary and the human attempt log are all gitignored because they are generated, large, or private. Read [What a fork can run](#what-a-fork-can-run) below before installing anything — roughly half the pipeline reproduces outside this project and half does not, and it is better to know which half in advance. To understand the system rather than run it, `docs/methodology.md` is self-contained and needs no setup at all.

### The corpus

The generated problems are their own repository, [onlymove-data](https://github.com/typohh/onlymove-data), and every step reads it from a checkout beside this one:

```bash
git clone git@github.com:typohh/onlymove.git onlymove
git clone git@github.com:typohh/onlymove-data.git onlymove-data
```

To keep it elsewhere, set `ONLYMOVE_TSUMEGOS_DIR` to its path. Below, `tsumegos/` means that checkout.

### Runtimes

- **Python 3.11 or newer** (developed on 3.12). NumPy and SciPy set that floor. Create a virtualenv, install `requirements.txt` into it, and invoke every pipeline step through it — `./.venv/bin/python3 -m pipeline.<step>` — so the repo-rooted absolute imports resolve.
- **Node.js 22** (the version the PWA is developed and deployed against) with npm, for the front end. Wrangler installs with the other dev dependencies and runs through `npx`; no global install.

### KataGo

- **The binary is not in the repository** — `bots/` is gitignored apart from `analysis.cfg`, which is tuned state rather than an artifact. Build or download a KataGo yourself and put it in `bots/`: an OpenCL build as `bots/katago`, CUDA builds as `bots/katago_cuda12` / `bots/katago_cuda13` — the CUDA major version is part of the filename because the binary links `libcublas.so.<major>` and is not portable across majors. `utils/katago/binary.py` picks the build that works on the machine (a CUDA build only once it answers a probe query, else OpenCL) and caches the choice per host; the `KATAGO_PATH` environment variable overrides it.
- **Network weights fetch themselves.** `utils.katago.models.get_model` downloads the `.bin.gz` that the analysis engine reads on first use, so the generation steps need no manual model setup. The default network is KataGo's transformer network, which needs engine v1.17 or newer.
- **So does the activation checkpoint.** Step 5 runs the vendored PyTorch copy of KataGo, which loads a `.ckpt` — a different artifact from the `.bin.gz`, and not derivable from it, since KataGo's own tooling converts PyTorch → binary and not back. `utils.katago.models.get_checkpoint` downloads and unpacks it into `models/` on first use.

### Hardware

A GPU is a practical requirement for generation, bot benchmarking and the three extraction passes (steps 3–7), and the rating fit (step 10) takes about 15–20 minutes on one RTX 3070 where it would take far longer on a CPU. The activation store for the full corpus is on the order of 30 GB on disk. `auxiliary/vast_orchestration.py` rents a Vast.ai instance for the three remotable steps. The remaining steps are CPU-only.

### Cloudflare (deployment only)

Deploying the PWA needs a Wrangler login, a D1 database created from `pwa/schema.sql`, and one secret — `VAPID_PRIVATE_KEY` for Web Push — set with `wrangler secret put`. A gitignored `pwa/.dev.vars` supplies the same values to `wrangler dev`. None of this is needed to read the code or to run the generation half of the pipeline.

## What a fork can run

**Steps 2–9 reproduce anywhere.** Tuning, generation, bot benchmarking, the three extraction passes, the feature caches and the motif proxies need only KataGo, the weights and a GPU, and they manufacture their own inputs. That is where a fork starts.

**Step 1 belongs to this project's Cloudflare account.** `fetch_database` runs `wrangler d1 export` against the live D1 holding the human attempt log. A fork cannot authenticate to it, and the log is not published — it is real users' per-attempt answer history.

**Steps 10–13 read that log**, so the rating, rank and motif fits do not reproduce outside the project. The seam for substituting an attempt log of your own is at least a clean one — `db/d1.sqlite` is an ordinary SQLite file whose shape is fixed by `pwa/schema.sql`, and the loaders open it directly — but nobody has done it, so treat it as an opening rather than a documented path.

**`npm run build` fails on a fresh clone**, and the error is worth recognising rather than debugging. `pwa/js/data-version.js` statically imports `../tsumegos/embedding_meta.json` so the data manifest can never be served stale; `pwa/tsumegos/` is written by step 14. Until packaging has run, esbuild stops on the unresolved import.

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

### 6. Extract ownership swing
Measures, per problem, how many stones' worth of ownership the correct first move swings against the best resolved refutation, into `activations/_signal_cache/ownership_swing.csv` (GPU, via the KataGo analysis binary; resumable by the `tid` column). **Mandatory, not diagnostic**: two of the shipped motif rules are written over it, and step 9 aborts without it.
```bash
./.venv/bin/python3 -m pipeline.extract_ownership_swing
```

### 7. Extract ownership maps
Records the ownership map at every intersection after a pass, after the answer and after the next four first moves, into `activations/_signal_cache/ownership_maps/` (GPU; resumable). Also mandatory: the territory, influence and yose motif rules are written over it.
```bash
./.venv/bin/python3 -m pipeline.extract_ownership_maps
```

### 8. Cache rating features
Builds both roll-up caches — the correct-cell activation vectors the rating trainer reads, and the candidate-move evaluations the motif rules and the axis analyses read — in one command, so they cannot drift apart. Incremental, and reconciles in both directions.
```bash
./.venv/bin/python3 -m pipeline.cache_rating_features
```

### 9. Build motif proxies
Labels every problem against the motif proxy rules into `axes/motif_proxies.npz`, one column per rule. These are the directions the next step teaches its axes onto, and it aborts without them. A rule or threshold change is invisible to the per-problem key, so it needs `--force`.
```bash
./.venv/bin/python3 -m pipeline.build_motif_proxies
```

### 10. Train the MIRT
Fits the rating and the K skill axes together, as one compensatory model on human first-time attempts, from the correct move's network activations, the bots' win rates and the board size, into `ratings/` and `axes/`. A third of the attempts are held aside from training, and the curve's floor, lapse and slope scale are calibrated on them afterwards. Takes no arguments — the shipped configuration is baked in; the experiment surface lives in `auxiliary/rating_dev/train_joint_mirt.py`.
```bash
./.venv/bin/python3 -m pipeline.train_mirt
```

### 11. Write per-cell axis vectors
Fans the axis projection out to every legal cell into `axes/per_cell/<size>/<tid>.npz`, which the client needs because it scores the move the user picked and not only the answer. Resumable by file existence.
```bash
./.venv/bin/python3 -m pipeline.write_mirt_per_cell
```

### 12. Map rating to rank
Fits the rating→rank curve and the per-server couplings against the rank surveys, into `ratings/rating_to_rank.json` and `ratings/rank_map.json`. Board-independent and needs no GPU, but it sits *inside* the sequence: it reads the ratings the MIRT wrote.
```bash
./.venv/bin/python3 -m pipeline.map_rating_to_rank
```

### 13. Fit population constants
Fits the retention and time-bias constants and the rank prior into `ratings/population_constants.json`. It reads the ratings and the rank curve, so it follows both; it fits no per-axis quantity, so an axis refit does not require re-running it.
```bash
./.venv/bin/python3 -m pipeline.fit_population_constants
```

### 14. Package for PWA
The single writer of `pwa/tsumegos/`: bundles tsumegos, ratings, motifs and the curve into the archives and sidecars the web app serves. Change-detects per size, so a curve-only or labels-only change publishes without rebuilding shards.
```bash
./.venv/bin/python3 -m pipeline.package_for_pwa
```

Steps 10–14 are ordered by hard data dependencies rather than convention; `docs/methodology.md` §3.1 spells out each edge and what breaks if it is skipped.

### Remote execution

The heavy generation steps can run on a rented Vast.ai GPU:

```bash
./.venv/bin/python3 -m auxiliary.vast_orchestration                    # no args prints subcommands
./.venv/bin/python3 -m auxiliary.vast_orchestration run <step> <size>  # tune / generate / benchmark are remoted (size forwarded); the rest stay local
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

## Ablations and the documents

No number in `docs/methodology.md` or the site's About page (`pwa/about.html`) is typed in by hand. Each one is a **figure** that a script under `ablations/` measures from the shipped artifacts and records in `ablations/figures.json`, and the document quotes it through a `<span data-fig="KEY">` that names the ledger entry.

```bash
./.venv/bin/python3 -m ablations --tier about        # minutes, no GPU: everything the About page quotes
./.venv/bin/python3 -m ablations --tier methodology  # adds the GPU measurements the methodology quotes
./.venv/bin/python3 -m ablations --check             # documents against the ledger, no measurement
./.venv/bin/python3 -m ablations --list              # every producer, its keys, and where each is quoted
```

This is what keeps the documents current after a refit:

- **Re-running a producer rewrites the quoted numbers in place** and prints every span it changed, so the sentences around a moved number can be reviewed before committing.
- **A sentence that leans on a figure without quoting it carries a guard**, an invisible `<!-- fig-guard: KEY OP NUMBER -->` comment. When a re-measured figure breaks the guard, the check fails and names the sentence to rewrite.
- **`npm run build` refuses to build while a document disagrees with the ledger**, so a stale or unbacked number cannot reach the site.
- **Producers abort on any surprise in their inputs**, so a run is also a check on the artifacts the pipeline just wrote.

The routine after a refit is therefore: run the tier, read the printed span diff and any violated guards, fix the sentences, and commit the ledger and the documents together. The comparisons behind the design choices (the `refit` tier, which refits the model under each alternative at several seeds) run the same way but take GPU-days; `ablations/README.md` has the full contract.

## License

The code is under the [MIT license](LICENSE). The vendored PyTorch KataGo under `utils/katago/vendor/` keeps its own license, in that directory. The problem collection is a separate repository, [onlymove-data](https://github.com/typohh/onlymove-data), licensed CC BY 4.0.
