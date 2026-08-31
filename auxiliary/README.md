# `auxiliary/`

Diagnostics, surgery tools and the rating research package — recurring helpers for debugging the live rank/rating system, inspecting the skill axes, auditing the corpus and repairing the database. They read the artifacts the pipeline produces (`db/d1.sqlite` / `d1.sql`, `ratings/<size>/`, `activations/<size>/`, `axes/`, `tsumegos/<size>/`, `pwa/tsumegos/`) and emit stdout reports, HTML or SQL. Nothing here is part of the in-order pipeline.

**What lives here and what lives in `ablations/`.** A script belongs THERE when its output is evidence for a choice the system already made — a figure a doc or `pwa/about.html` quotes, re-run after a refit. It belongs HERE when it debugs a live account, audits or repairs the corpus and database, describes the player population, or explores a decision not yet made (`rating_dev/`, the labeler). The everyday split: `ablations/` answers "was this the right choice", `auxiliary/` answers "what is going on".

**Import direction.** `pipeline/` and `utils/` never import from here. The genuinely shared helpers — the activation-cache builder and contract, the attempt/survey loaders, the profile/feature builders, the client-mirror solve-time window — live in `utils/rating/features.py` and `utils/rating/rank_surveys.py`, and the M curve, the per-server couplings and the time dock live in `utils/rating/ratings_io.py`. So everything in this directory is a leaf: deletable without touching production code. `rating_dev/` imports `utils/` and its own modules. Nothing here imports `ablations/`; one edge runs the other way, `ablations.eval_rank_endpoint_bias` importing the Rank-tab replica from `recreate_rank_graph`. **A reader for a shipped `ratings/` artifact goes in `ratings_io`, never beside its caller** — two hand-rolled copies of the `population_constants.json` time dock is how one of them ended up gating on a field that had stopped shipping, silently reporting an untimed rating in a comparison that must be time-adjusted.

**What does NOT live here.** One-off scripts are deleted once their job is done rather than kept as dead weight — the 2026-07-21 id surgery drivers, the momentum-vs-window-MLE estimator bake-off, the ICA-era motif inspector, and the closed single-hypothesis CV harnesses were all removed on 2026-08-02. Where a result matters it is stated in `CLAUDE.md` or `docs/methodology.md`; where the code matters, `git log --diff-filter=D --name-only -- auxiliary/` finds it.

**Running.** Absolute imports rooted at the repo, so invoke as modules through the venv:

```bash
./.venv/bin/python3 -m auxiliary.<name> [args]
./.venv/bin/python3 -m auxiliary.rating_dev.<name> [args]
```

Board size follows the usual `--size` / `--sizes` convention (9 or 13). `auxiliary/__init__.py` (and `rating_dev/__init__.py`) are package markers. The `.mjs` tools run under `node` from the repo root.

---

## Database surgery

The freeze → operate → unfreeze procedure these run inside is `docs/db_surgery_runbook.md`. All of them work on a LOCAL SQLite copy of the D1 export and emit SQL to apply remotely; none writes to remote D1 directly.

- **`purge_tsumegos.mjs`** — remove every trace of a set of problem ids from user history: attempt rows, recall schedule, leaderboard buffers, and the per-user digest chains (`user_seq` / `running_sum`), rebuilt with the Worker's own `rowHash` from `pwa/shared/sync-digest.js` so the repaired chains are byte-identical to what the live INSERT computes. `--db --ids --apply --emit-sql --emit-corpus-map`.
- **`corpus_surgery.py`** — the corpus-side half: consumes an id map (from `purge_tsumegos --emit-corpus-map`, or `check_corpus_ids --emit-map`) and renames/quarantines the SGF plus every per-id artifact (activations, per-cell embeddings). Dry-run by default; `--apply`, `--root` for a rehearsal tree.
- **`check_corpus_ids.py`** — re-verifies the symmetry-canonical id invariant: recomputes every SGF's orientation-invariant hash and reports misnamed files and duplicate positions. `--emit-map` regenerates a surgery map if a future defect ever needs one. Expect 0/0.
- **`update_leaderboard_ranks.mjs`** — retroactively recompute every leaderboard row's `buffer` / `pred_rank` / `pred_rank_se` / `time_delta_rating` after a re-rate. Ordering is load-bearing (`package_for_pwa` → deploy → update) — see the leaderboard note in `CLAUDE.md`. `--remote` opens its own maintenance freeze and fetches the mirror inside it, into a private copy that is discarded on exit; `--keep-refresh` puts that fetch into `db/d1.sqlite` itself. `--local` / `--remote` / dry run.

## Corpus QC

- **`check_tsumego_sequences.py`** — the corpus audit: does each winning line run to the double pass, does it stay winning at every node by both winrate and score, is the winner `variation_nodes[0]`, and how many nodes carry a self-contradictory evaluation. This is what quarantine lists are built from. `--sizes --limit --csv --workers`.

## Rank & rating experiments

The harnesses that VALIDATE the shipped mapping — `check_mapped_vs_declared_rank.py`, `eval_rank_map_cv.py`, `eval_equivalent_tournament_games.py`, `eval_recall_model.py`, `eval_rank_endpoint_bias.py` — live in `ablations/`. What stays here is the surface for mapping work not yet decided:

- **`map_rating_to_rank_dev.py`** — the experimental copy of `pipeline.map_rating_to_rank`; writes nothing by default so it can never clobber the shipped curve. Where mapping experiments (time forms, gauge, coupling structure) are run.

## Per-user replicas

Faithful offline ports of client code, for debugging one account against the local DB copy.

- **`estimate_user_rank.py`** — the client's window-MLE rank estimate for one user, mapped to a rank label. `user_id --recent-days/--last-n/--include-repeats`.
- **`recreate_rank_graph.py`** — the Rank-tab skill curve (`pwa/js/rank-decomposition.js`): 30-day Gaussian-pooled 4PL MLE in internal-rank space. `USER_ID [--server] [--csv] [--plot out.png]`.
- **`user_weakness_report.py`** — the Weaknesses pane (`weakness.js` + `stats-weaknesses.js` + `stats-compute.js`): the 2K+2 logistic, its Laplace posterior, and the two-level detection, for one user or every user with a linked profile.
- **`weekly_player_stats.py`** — one row per active ISO week for a player: problem-rank mean/sd, median solve time, attempt count. `<user_id>`.

## Fleet stats

- **`leaderboard_player_stats.mjs`** — per-player report over the leaderboard: novel attempt counts per board, median solve time, attempts/day, and the onlymove estimate against the verified EGF/AGA rating, with a summary correlation. `--with-unlinked`.
- **`rank_activity_buckets.py`** — recently-active users (24h / 7d) bucketed by rank band × tsumegos-per-day. Writes `rank_activity_buckets.png` (gitignored; `--no-plot` to skip).
- **`db_stats.py`** — top-line stats from `db/d1.sql`: totals, recent weekly activity, tenure × activity-rate cohort grid.

## Live system

- **`check_live_status.mjs`** — probe the deployed Worker: SPA routes, static assets, packaged data files, and whether writes are frozen. `pwa/deploy.mjs` runs it after every deploy and `npm run status` runs it on demand; `--expect-frozen` / `--expect-unfrozen` assert the freeze state for the runbook's two flips. It encodes the two 404s that are CORRECT and read as an outage by hand — the SPA fallback answers `/`, `/9`, `/13` only for a navigation-shaped request, and the five inlined data files are compiled into the bundle rather than served. Retries, because a fresh deploy propagates over a few seconds. Node stdlib only.

## Admin


## Skill-axis inspection

- **`rating_dev/inspect_axes.py`** — renders the extreme-loading problems on each production axis, both tails, straight from `axes/per_problem.csv`. This is the motif-labeling tool: a refit produces an axis, you eyeball its tails here, name them in `axes/labels.json`, re-run `package_for_pwa`. The boards are the whole evidence — naming an axis by what DRIVES it needs the axis to be a linear map on nameable inputs, and the MIRT's trunk means no such decomposition exists. `--per-problem --size --rows --cols --open`.
- **`labeler/`** — a local active-learning motif labeler, and the route from a human motif name to a column of `axes/axis_bank.npz`. You tick labels on rendered positions; an elastic-net logistic probe (one-vs-rest, per label) retrains after every save, pre-fills its predictions on the next position so you only correct, and re-ranks the queue by prediction uncertainty. `grid.html` is the bulk-review view — a page of one label's confident predictions, correctable in batches.

  It probes **the MIRT's own per-problem features** (`utils.rating.features.load_joint_mirt_features`, single-sourced with `pipeline.train_mirt` so the two cannot drift), which is what makes a label convertible into a bank direction: the probe is a function of exactly what the trunk reads, so a direction it finds is one the trunk can represent and therefore one the bank can teach the fit. Those features carry difficulty, so check a probe is not mostly predicting the rating before promoting it — the MIRT's factor 1 already carries that.

  ```bash
  ./.venv/bin/python3 -m auxiliary.labeler.precompute_features --pool 5000   # → labeler/cache/ (gitignored)
  ./.venv/bin/python3 -m auxiliary.labeler.server                            # → http://localhost:8899
  ```

  `--max-rank` (default `1d`) bounds the pool by difficulty, because a human cannot label a position they cannot read. The annotations live in `labeler/labels.json`, which is TRACKED — it is hand-entered work that nothing regenerates. Re-run `precompute_features` after a corpus top-up or a `cache_rating_features` rebuild; annotations for problems that have left the corpus are kept and ignored.

---

## `rating_dev/`

The rating-model research package. `pipeline/train_mirt.py` is deliberately CLI-free with its config baked in; all experimentation happens here so a test tweak cannot perturb the shipped ratings. It imports only `utils/` and its own modules.

- **`train_rating_staged.py`** — the experiment surface and production superset: `--production-flags` reproduces the shipped configuration, and the rest of the flag set (gauge, divergence barriers, EIV probes, the staged/M0b legacy configs, the eval suite, the alternative objectives incl. `--even-error`) is where the model's design decisions were measured. `--staged` fits the per-problem 4PL heads one at a time instead of jointly.
- **`validate_ratings.py`** — the single home of rating validation. Scores a ratings directory on a hash-confirmed holdout: held-out-problem logloss, held-out Δskill flatness, survey-ρ (base + time-adjusted). `--ratings-dir --seed --holdout-problem-frac [--expect-hash]`.
- **`holdout_set.py`** — deterministic held-out problem-set generator: `(seed, fraction)` → a reproducible subset plus a 16-hex content hash, so trainer and bench can prove they used the same set. `--seed --frac [--out file]`.
- **`eval_phase2_units.py`** — the certificate harness for the phase-2 global scale fit: regenerates outcomes from the fitted model at a known true scale and checks what the estimator returns. This is what established that the slope scale is real and the floor scale was a Neyman–Scott artifact. `--synthetic-outcomes`, unit-granularity and drift sweeps.
- **`model_comparison.py`** — shared data/feature loaders for the harness (re-exported from `utils.rating.features`) plus `load_problem_features`. No CLI.
