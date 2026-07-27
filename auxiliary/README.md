# `auxiliary/`

One-off diagnostics and dev tools — recurring helpers for debugging the live rank/rating system, inspecting the embedding, and sanity-checking data. They read the artifacts the pipeline produces (`attempts/database.sqlite` / `database.sql`, `ratings/<size>/`, `activations/<size>/`, `embeddings/joint/`, `tsumegos/<size>/`, `pwa/tsumegos/`) and emit stdout reports, HTML, or PNGs. Nothing here is part of the in-order pipeline.

**Import direction.** `pipeline/` and the `rating_dev/` sub-package never import these top-level scripts — they depend only on `utils/`. The genuinely shared helpers (the activation-cache contract, attempt/survey loaders, profile/feature builders) live in `utils/rating_features.py` and `utils/rank_surveys.py`. So everything at this top level is a leaf: deletable without touching production or research code.

**Running.** Absolute imports rooted at the repo, so invoke as modules through the venv:

```bash
./.venv/bin/python3 -m auxiliary.<name> [args]
```

Board size follows the usual `--size` / `BOARD_SIZE` convention (9 or 13). `auxiliary/__init__.py` (and `rating_dev/__init__.py`) are package markers.

---

## Rank & rating QA

- **`check_mapped_vs_declared_rank.py`** — end-to-end QA of the production rating→rank map: fits each survey user's 80%-clear v2-rating, runs it through `rank_map.json`, compares to their declared server rank, and buckets the error. Reads `ratings/` + `rank_map.json`; flags include `--time-adjust`, `--time-recenter`. Run after every `map_rating_to_egf` refit.
- **`eval_strong_user_error.py`** — per-account rank error for strong (EGF 4d+) survey users across rating configs; for each `label=dir` config it reproduces `pipeline.map_rating_to_egf` exactly (joint curve + per-server couplings) and maps each EGF declaration through the fresh curve.
- **`diag_4pl_structure.py`** — structural diagnostics for 4PL rating artifacts: per-decile clamp %, slope/floor percentiles, and Spearman correlations against the empirical bot-ladder floor/steepness. `--dir` points at any `ratings*`-shaped directory (repeatable for side-by-side).
- **`eval_rank_estimators.py`** — offline bake-off of live (causal) rank estimators (momentum, causal-kernel, state-space, info-window) replaying real users' attempt streams, plus a shock test and a closed-loop forward simulation. Uses the exact client rank-index math. Modes: `--shock`, `--simulate`, `--misclick`, `--only`, `--by-cohort`, `--csv`.

- **`serve_spread_history.py`** — how far apart, in TODAY's ratings, the problems served back-to-back to one user actually were, bucketed over the project's lifetime. The selector draws from the ~50 problems nearest the serve rank, so an adjacent novel pair was equal-by-construction in the era's own rating frame and any gap visible today is that era's rating error (√2 × per-problem). Prints the spread against a no-targeting and a perfect-targeting reference, cluster-bootstrap CIs, rank/tenure strata with a fixed-mix standardization, and a lag profile that separates per-problem error (flat in lag) from serve-rank drift (rising). Reads `attempts/database.sqlite` + the shipped `pwa/tsumegos/`. `--size --bucket week --strict-adjacency --min-tenure --min-coverage --floor --csv --plot`.

## Per-user rank debugging

- **`estimate_user_rank.py`** — replicates the client's MLE rank estimate for one user from first-time attempts (`fit_user_rating` @ `CALIBRATION_P`) and maps it to a rank label. Reads `attempts/database.sqlite` + `ratings/`. Args: `user_id`, `--recent-days`/`--last-n`/`--include-repeats`.
- **`recreate_rank_graph.py`** — offline port of the client Rank-tab skill curve (`pwa/js/rank-decomposition.js`): 30-day Gaussian-pooled 4PL MLE in internal-rank space, optional near-rank filter + server display. Reads `ratings/` + `pwa/tsumegos/population_constants.json`. `recreate_rank_graph USER_ID [--server …] [--csv] [--plot out.png]`.

## Fleet & dataset stats

- **`rank_activity_buckets.py`** — recently-active users (last 24h / 7d) bucketed by rank band × activity (tsumegos/day), building the rating→rank map from `pwa/tsumegos/<size>/*.zip` shard metadata. Writes `rank_activity_buckets.png` (`--no-plot` to skip).
- **`db_stats.py`** — top-line stats from `attempts/database.sql`: totals, recent weekly activity, and a tenure × activity-rate cohort grid.

## Embedding / activation inspection

- **`inspect_extreme_problems.py`** — ranks problems by an embedding component's positive tail (post-skew-ICA = motif present) and renders HTML bands or PWA URLs. Reads `embeddings/joint/per_problem.csv` (+ `labels.json`). `--pc N` / `--all-pcs`, `--open`. Used for motif-label curation.
- **`inspect_activation_signals.py`** — raw (pre-ICA) inspector: ranks problems by a single `(layer, channel)` of the KataGo activations and renders HTML / PWA URLs. **Also the builder of the correct-move signal cache** (`activations/_signal_cache/`, via `_build_cache`) that `utils.rating_features.load_joint_activations` — and therefore the `rating_dev` trainers — read at runtime, so it must be run once per board before those consume activations. `--layer --channel --size --open`, `--rebuild-cache`.

## Tsumego QC

- **`check_tsumegos_mainline.py`** — validates that every SGF in `tsumegos/<size>/` has a complete variation (a path ending in two consecutive passes). `--board-size --workers --list-incomplete --sample`.

---

## `rating_dev/`

The rating-model research/experiment package. The production trainer (`pipeline/train_rating.py`) is intentionally CLI-free with config baked in; all experimentation happens here so a test tweak can't perturb the shipped ratings. It imports only `utils/` and its own modules — never the top-level scripts above.

- **`train_rating_staged.py`** — the live experiment surface (the production superset). Staged coordinate-descent trainer: `--staged`, `--global-stage1` (learnable global slope/floor/ceiling in phase 1), `--separate-trunks`, `--two-stage`, arbitrary `--stage-plan`, the full eval suite, and the alternative objectives. Writes per-stage `ratings/`-shaped artifacts and runs validation per stage.
- **`train_rating_research.py`** — the pre-staged surface (joint-fit feature→4PL net with the same ablation/objective flags); the historical baseline.
- **`holdout_set.py`** — deterministic held-out problem-set generator: `(seed, fraction)` → a reproducible subset plus a 16-hex content hash, so trainer and validator can prove they used the same set. `--seed --frac [--out file]`.
- **`validate_ratings.py`** — the single home of rating validation. Scores a ratings directory on a hash-confirmed holdout for held-out-problem logloss, held-out Δskill flatness, and survey-ρ (base + time-adjusted). `--ratings-dir --seed --holdout-problem-frac [--expect-hash]`.
- **`model_comparison.py`** — shared rating-model data/feature loaders for the harness. Provides `build_profiles`, `load_joint_activations`, `load_joint_bot_winrate`, `load_joint_move_features` (re-exported from `utils.rating_features`) plus `load_problem_features` (activation-PCA features for a problem set). No CLI.

## `labeler/`

A self-contained local web tool for human-labeling tsumego motif features on the embedding axes — its own mini-project. A precomputed residualized-feature cache (correct-move trunk activations, residualized on rating+size like `build_embedding`) feeds an elastic-net logistic multi-label probe that retrains after each save and reranks the queue by prediction uncertainty.

- **`precompute_features.py`** — builds the residualized correct-move feature cache (pre-PCA/ICA, full residual space) from `embeddings/joint/per_problem.csv`. `--pool 5000` → `cache/features.npy` + `cache/pool.json`.
- **`server.py`** — the labeler HTTP server (`localhost:8899`): serves `index.html`/`grid.html`, the `/api/{next,save,list,grid}` endpoints, and retrains the probe on save. `./.venv/bin/python3 -m auxiliary.labeler.server`.
- **`render.py`** — board-rendering helper (wraps `utils.board_render` and adds a rank label). Library only.
- **`index.html`** — single-problem labeling UI (board + predicted-probability checklist + Save/Focus/Add-Label).
- **`grid.html`** — batch-review UI (grid of labeled problems, filter + confirm correct/incorrect).
- **`labels.json`** — the labeler's working label store. The canonical motif registry is `embeddings/joint/labels.json` (what `build_embedding`/`package_for_pwa` consume); this copy can lag, so reconcile before treating it as authoritative.
- **`cache/`** (gitignored) — holds the `precompute_features.py` output; rebuild it if `embeddings/joint/pca.npz` changes.

---

## Generated artifacts

`rank_activity_buckets.png` is the gitignored plot output of `rank_activity_buckets.py`; regenerate by re-running it.
