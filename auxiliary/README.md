# `auxiliary/`

One-off diagnostics, research/dev experiments, and prototypes. Nothing here is part of the in-order pipeline (`pipeline/`) or shipped to the PWA — these scripts read the artifacts the pipeline produces (`attempts/database.sqlite` / `database.sql`, `ratings_v2/<size>/`, `activations/<size>/`, `embeddings/joint/`, `tsumegos/<size>/`, `pwa/tsumegos/`) and emit stdout reports, CSVs, or PNGs.

**Running.** Everything uses absolute imports rooted at the repo, so invoke scripts as modules through the venv so `PYTHONPATH` resolves:

```bash
./.venv/bin/python3 -m auxiliary.<name> [args]
```

Board size follows the usual `--size` / `BOARD_SIZE` convention (9 or 13). `auxiliary/__init__.py` (and `rating_dev/__init__.py`) are package markers so the `-m auxiliary.*` form works.

**⚠ marks scripts that target a superseded on-disk layout.** They reference a top-level `ratings/` directory (with v1 filenames like `tsumego_parameters_final.csv` / `user_profiles_final.csv` / `bot_ratings_final.csv`) that no longer exists — the rating backend is now `ratings_v2/` (`tsumego_ratings_final.csv`, `tsumego_quality_final.csv`, `tsumego_floor_final.csv`). A ⚠ script won't run unmodified against the current tree; most of them survive because other modules still import helper functions from them.

---

## Rating model — dev & diagnostics

- **`rating_dev/`** — the rating-model research/experiment package (trainers, held-out validation, shared loaders). See its section below. The production trainer is `pipeline/train_rating.py` (no CLI, config baked in); experiments happen here.
- **`diag_4pl_structure.py`** — structural diagnostics for 4PL rating artifacts: per-decile clamp %, slope/floor percentiles, and Spearman correlations against the empirical bot-ladder floor/steepness. `--dir` points at any `ratings_v2*`-shaped directory (repeatable for side-by-side).

## Rating model — activation-feature studies

- **`predict_rating_from_activations.py`** ⚠ — ridge linear probe of how much of a problem's rating is recoverable from correct-move KataGo trunk activations (K-fold R²/RMSE/r); `--contrastive` adds cross-candidate stats. Also defines the activation feature-loaders imported by `rating_dev/model_comparison.py` and `user_results_block_ablation.py`.
- **`user_results_block_ablation.py`** — logistic ablation (GroupKFold by problem) of which activation "blocks" (correct move, contrastive mean/std/max/z) carry user-success signal, standalone and marginal over the correct-move block. Reads cached activations + `attempts/database.sqlite`.

## Rank mapping, EGF calibration & surveys

- **`compare_rank_surveys.py`** — a shared helper module: `rank_to_scalar` / `scalar_to_rank` and the survey loaders are imported by `check_mapped_vs_declared_rank`, `eval_strong_user_error`, `rank_mapping_lab`, `joint_rank_model*`, and `validate_time_adjusted_rating`. Its own `main()` (declared-vs-internal rank comparison) ⚠ reads the old `ratings/` layout.
- **`check_mapped_vs_declared_rank.py`** — end-to-end QA of the production map: fits each survey user's 80%-clear v2-rating, runs it through `rank_map.json`, compares to declared server rank, and buckets the error. Reads `ratings_v2/` + `rank_map.json`. Flags include `--time-adjust`, `--time-recenter`.
- **`eval_strong_user_error.py`** — per-account rank error for strong (EGF 4d+) survey users across rating configs; for each `label=dir` config it reproduces `pipeline.map_rating_to_egf` exactly (joint curve + per-server couplings) and maps each EGF declaration through the fresh curve.
- **`rank_mapping_lab.py`** — experiment harness comparing rating→EGF curve families (free/linear/cubic/exp/quadratic) with grouped 5-fold CV, bootstrap CIs, and per-rank-band RMSE. Reads `ratings_v2/` + surveys; uses the coupling-shape fit from `fit_2018_couplings.py`.
- **`fit_2018_couplings.py`** ⚠ — fits server↔EGF coupling shapes from the 2018 cross-server survey (with a 2024 variant); imported by `rank_mapping_lab.py`. Its default input path points at `rank_survey_2018.csv` in the repo root, but the surveys live in `data/`.
- **`fit_statistics_shapes.py`** — fits candidate rank→rating shapes (linear/exp/cubic/piecewise) to the EGF even-game win-rate table in `statistics.txt` by binomial likelihood, and holds the `CubicRatingCurve` used in packaging. `--plot` writes `statistics_shape_fit.png`.

## User-rank estimation (client-equivalent)

- **`estimate_user_rank.py`** — replicates the client's MLE rank estimate for one user from first-time attempts (`fit_user_rating` @ `CALIBRATION_P`) and maps it to a rank label. Reads `attempts/database.sqlite` + `ratings_v2/`. Args: `user_id`, `--recent-days`/`--last-n`/`--include-repeats`.
- **`eval_rank_estimators.py`** — offline bake-off of live (causal) rank estimators (momentum, causal-kernel, state-space, info-window) replaying real users' attempt streams, plus a shock test and a closed-loop forward simulation. Uses the exact client rank-index math. Modes: `--shock`, `--simulate`, `--misclick`, `--only`, `--by-cohort`, `--csv`.
- **`recreate_rank_graph.py`** — offline port of the client Rank-tab skill curve (`pwa/js/rank-decomposition.js`): 30-day Gaussian-pooled 4PL MLE in internal-rank space, optional near-rank filter + server display. Reads `ratings_v2/` + `pwa/tsumegos/population_constants.json`. `recreate_rank_graph USER_ID [--server …] [--csv] [--plot out.png]`.
- **`rank_activity_buckets.py`** — recently-active users (last 24h / 7d) bucketed by rank band × activity (tsumegos/day), building the rating→rank map from `pwa/tsumegos/<size>/*.zip` shard metadata. Writes `rank_activity_buckets.png` (`--no-plot` to skip).

## Joint rank model (placement from activations)

- **`joint_rank_model.py`** ⚠ — v1 prototype: a single IRT fit over attempts + survey declarations with an activation→difficulty map. Imported by `joint_rank_model_v2.py` for shared IRT helpers; writes `joint_rank_placements.csv`.
- **`joint_rank_model_v2.py`** ⚠ — unified joint fit: outcomes + all-server ordinal cumulative-link couplings, a θ-prior to prevent scale collapse, and optional 2PL discrimination. Documented in `docs/joint_rank_methodology.md`. Writes `joint_rank_placements_v2.csv` and optionally `rank_map_v2.json`.

## Time-value-of-rank

(Conclusion of this line of work: a time→rank main effect is **not** identifiable from logs; only the time-tendency × server-speed interaction is.)

- **`time_value_analysis.py`** — between-user study of rank-equivalent ability vs difficulty-adjusted solve time (pooled-slope Rasch + ridge-shrunk per-user efficiency). Reads `ratings_v2/`; writes `time_value_per_user.csv`.
- **`time_value_from_server_ranks.py`** — cross-server natural experiment: two-way fixed-effects on `rank_surveys` isolating slow-vs-fast server rank offsets (bootstrapped).
- **`time_value_interaction.py`** — the identifiable model `native_rank ~ user_FE + server_FE + speed[server]×time_tendency[user]`; reports the interaction coefficient + bootstrap CI.
- **`analyze_time_benefit.py`** — per-user offline estimate of whether extra solving time helps on novel timed attempts (Fisher-scored time effect, difficulty-controlled). Writes `time_benefit_users.csv`; `--plot` → `time_benefit.png`.
- **`validate_time_adjusted_rating.py`** — tests whether adding solve-time features improves the rating↔survey-rank Spearman (partial-ρ diagnostic, full-data α sweep, cross-fitted honest estimate). Reads `ratings_v2/`.

## Bot calibration & handicap

- **`calibrate_handicap_ranks.py`** — calibrates humanv0 rank-Elo gaps + a handicap-effectiveness curve via standard-handicap 19×19 self-play (KataGo humanPolicy); restartable from SGFs in `calibration_handicap_sgf/`. Flags: `--threads`, `--max-games`, `--refit-interval`, `--report-only`.
- **`calibrate_ranks_19x19.py`** — sibling: humanv0 rank-Elo via adjacent-rank 19×19 self-play; SGFs in `calibration_19x19_sgf/`.
- **`analyze_bot_prediction_accuracy.py`** — how well named bots correlate with human solve patterns across overlapping user profiles (Pearson/Spearman/MSE/coverage). Writes `bot_prediction_accuracy.csv` + `bot_prediction_by_profile_count.csv`.

## Embedding / activation inspection (ICA motif axes)

- **`inspect_extreme_problems.py`** — ranks problems by an embedding component's positive tail (post-skew-ICA = motif present) and renders HTML bands or PWA URLs. Reads `embeddings/joint/per_problem.csv` (+ `labels.json`). `--pc N` / `--all-pcs`, `--open`.
- **`inspect_activation_signals.py`** — the raw (pre-ICA) counterpart: ranks problems by a single `(layer, channel)` of the KataGo activations and renders HTML / PWA URLs. Caches correct-move vectors under `activations/_signal_cache/`. `--layer --channel --size --open`.
- **`ica_stability.py`** — ICASSO stability diagnostic (bootstrap-resampled FastICA) for the joint skew-ICA embedding: how many of K=64 components are reproducible (Iq index). Reads `embeddings/joint/`, writes `ica_stability.json`.
- **`labeler/`** — local web tool for human-labeling motif axes; see its section below.

## Dataset & DB diagnostics

- **`db_stats.py`** — top-line stats from `attempts/database.sql`: totals, recent weekly activity, and a tenure × activity-rate cohort grid.
- **`analyze_attempts.py`** — users bucketed by completed-tsumego count, monthly actives, and ≥30-attempt/month users, from `attempts/database.sql`.
- **`problem_attempt_coverage.py`** — cross-tab of how many problems have ≥X first-time attempts from rank-estimable users (≥Y correct *and* incorrect), per size. Also exposes `load_first_time_attempts()`, imported by several other scripts here.
- **`recall_time_distribution.py`** — reconstructs per-user spaced-repetition recall-time distributions from the repeat-chains in `attempts/database.sql` (timing-only + outcome-based 80%-recall fit). Writes `recall_time_distribution.png`.

## Tsumego quality control & cleanup

- **`check_tsumegos_mainline.py`** — validates that every SGF in `tsumegos/<size>/` has a complete variation (a path ending in two consecutive passes). `--board-size --workers --list-incomplete --sample`.
- **`check_pass_variations.py`** — finds SGFs whose variations end in a pass and moves problems with a non-primary pass to `old_tsumegos/`.
- **`move_edge_tsumegos_13.py`** — moves 13×13 problems that place a stone on the 1st/2nd line within the first 4 moves to `tsumegos/broken/13/`. `--dry-run`.

---

## `rating_dev/`

The rating-model research/experiment package. The production trainer (`pipeline/train_rating.py`) is intentionally CLI-free with config baked in; all experimentation happens here so a test tweak can't perturb the shipped ratings.

- **`train_rating_v2_staged.py`** — the live experiment surface (the production superset). Staged coordinate-descent trainer: `--staged`, `--global-stage1` (learnable global slope/floor/ceiling in phase 1), `--separate-trunks`, `--two-stage`, arbitrary `--stage-plan`, the full eval suite, and the alternative objectives. Writes per-stage `ratings_v2/`-shaped artifacts and runs validation per stage.
- **`train_rating_v2_research.py`** — the pre-staged surface (joint-fit feature→4PL net with the same ablation/objective flags); the historical baseline.
- **`holdout_set.py`** — deterministic held-out problem-set generator: `(seed, fraction)` → a reproducible subset plus a 16-hex content hash, so trainer and validator can prove they used the same set. `--seed --frac [--out file]`.
- **`validate_ratings.py`** — the single home of rating validation. Scores a ratings directory on a hash-confirmed holdout for held-out-problem logloss, held-out Δskill flatness, and survey-ρ (base + time-adjusted). `--ratings-dir --seed --holdout-problem-frac [--expect-hash]`.
- **`model_comparison.py`** — shared rating-model data/feature loaders for the harness. Provides the cross-board loaders the trainers and validator import — `build_profiles`, `load_joint_activations`, `load_joint_bot_winrate`, `load_joint_move_features` (re-exported from `utils.rating_features`) — plus `load_problem_features` (unsupervised activation-PCA features for a problem set). No CLI.

## `labeler/`

A local active-learning web tool for human-labeling tsumego motif features on the embedding axes. A precomputed residualized-feature cache (correct-move trunk activations, residualized on rating+size like `build_embedding`) feeds an elastic-net logistic multi-label probe that retrains after each save and reranks the queue by prediction uncertainty.

- **`precompute_features.py`** — builds the residualized correct-move feature cache (pre-PCA/ICA, full residual space) from `embeddings/joint/per_problem.csv`. `--pool 5000` → `cache/features.npy` + `cache/pool.json`.
- **`server.py`** — the labeler HTTP server (`localhost:8899`): serves `index.html`/`grid.html`, the `/api/{next,save,list,grid}` endpoints, and retrains the probe on save. `./.venv/bin/python3 -m auxiliary.labeler.server`.
- **`render.py`** — board-rendering helper (wraps `utils.board_render` and adds a rank label). Library only.
- **`index.html`** — single-problem labeling UI (board + predicted-probability checklist + Save/Focus/Add-Label).
- **`grid.html`** — batch-review UI (grid of labeled problems, filter + confirm correct/incorrect).
- **`labels.json`** — the labeler's working label store. The canonical motif registry is `embeddings/joint/labels.json` (what `build_embedding`/`package_for_pwa` consume); this copy can lag, so reconcile before treating it as authoritative.
- **`cache/`** (gitignored) — holds the `precompute_features.py` output; rebuild it if `embeddings/joint/pca.npz` changes.

---

## Generated artifacts

These are regenerable outputs of the scripts above, not inputs. A few are committed as convenience snapshots:

| File | Produced by | Note |
|---|---|---|
| `bot_prediction_accuracy.csv`, `bot_prediction_by_profile_count.csv` | `analyze_bot_prediction_accuracy.py` | |
| `rank_map_v2.json`, `rank_map_v2_reweighted.json` | `joint_rank_model_v2.py` | prototype output (not the shipped `rank_map.json`) |
| `statistics_shape_fit.png` | `fit_statistics_shapes.py` | |

The remaining on-disk outputs are gitignored: `joint_rank_placements*.csv` (`joint_rank_model*.py`), `rank_activity_buckets.png`, `recall_time_distribution.png`, `time_benefit.png` / `time_benefit_users.csv`, and `time_value_per_user.csv`.
