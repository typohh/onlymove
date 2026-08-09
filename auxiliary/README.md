# `auxiliary/`

Diagnostics, surgery tools and the rating research package — recurring helpers for debugging the live rank/rating system, inspecting the skill axes, auditing the corpus and repairing the database. They read the artifacts the pipeline produces (`db/d1.sqlite` / `d1.sql`, `ratings/<size>/`, `activations/<size>/`, `axes/`, `tsumegos/<size>/`, `pwa/tsumegos/`) and emit stdout reports, HTML or SQL. Nothing here is part of the in-order pipeline.

**Import direction.** `pipeline/` and `utils/` never import from here. The genuinely shared helpers — the activation-cache builder and contract, the attempt/survey loaders, the profile/feature builders — live in `utils/rating/features.py` and `utils/rating/rank_surveys.py`, so everything in this directory is a leaf: deletable without touching production code. `rating_dev/` imports `utils/` and its own modules; two top-level scripts import a sibling (`map_rating_to_rank_dev` ← `check_mapped_vs_declared_rank`, `eval_rank_endpoint_bias` ← `recreate_rank_graph`) and `rating_dev/train_residual_axes` imports `eval_score_gap_axis`.

**What does NOT live here.** One-off scripts are deleted once their job is done rather than kept as dead weight — the 2026-07-21 id surgery drivers, the momentum-vs-window-MLE estimator bake-off, the ICA-era motif labeler and inspector, and the closed single-hypothesis CV harnesses were all removed on 2026-08-02. Where a result matters it is stated in `CLAUDE.md` or `docs/methodology.md`; where the code matters, `git log --diff-filter=D --name-only -- auxiliary/` finds it.

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
- **`update_leaderboard_ranks.mjs`** — retroactively recompute every leaderboard row's `buffer` / `pred_rank` / `pred_rank_se` / `time_delta_rating` after a re-rate. Ordering is load-bearing (`package_for_pwa` → deploy → fetch → update) and its fetch must be CHAINED into the apply — see the leaderboard note in `CLAUDE.md`. `--local` / `--remote` / dry run.

## Corpus QC

- **`check_tsumego_sequences.py`** — the corpus audit: does each winning line run to the double pass, does it stay winning at every node by both winrate and score, is the winner `variation_nodes[0]`, and how many nodes carry a self-contradictory evaluation. This is what quarantine lists are built from. `--sizes --limit --csv --workers`.

## Rank & rating QA

- **`check_mapped_vs_declared_rank.py`** — end-to-end QA of the shipped rating→rank map: fits each survey user's 80%-clear rating, maps it through the curve and their server's coupling, and buckets the error against what they declared. Also runs both declaration-honesty routes against verified federation ratings. Run after every `map_rating_to_rank` refit.
- **`eval_equivalent_tournament_games.py`** — puts the shown rank's accuracy in units a Go player already has: how many tournament games a rating would need to be this good. Fits a stone-value curve to `data/reference/even_game_winrates.txt` (one joint fit across every grade × gap cell, so the +1..+4 columns bend as a stone's worth changes up the ladder — measured at roughly a quarter of a logit per stone in the mid kyus against three quarters in the dan range, so a dan game carries an order of magnitude more information), simulates a Swiss on that law, and sweeps the trailing window until its ρ / within-1 / within-2 match `check_mapped_vs_declared_rank`'s CSV. On the EGF-system declarations all five statistics land together at **16 ± 3 games** (the ± is a bootstrap over the declarations behind the target — the dominant term, since a curve point is already averaged over independent populations and rounds), and the pooled figure stays inside that margin under every sensitivity: link from the rating-indexed Se blocks instead of the declared-grade table, 20k dropped, our own user population as entrants, band-quantized vs continuous reference, peer vs ground-truth comparison. Two results about the simulation itself, both measured and both counter-intuitive: **pairing strictly by information is the wrong objective** — Fisher information peaks at equal strength, so it pairs nearest-neighbour, which leaves the field a chain that pins order while leaving the chain's own length unmeasured, and absolute rank error stops falling past ~20 games while ρ keeps rising (`--pair-mixing 0` reproduces it); and **a free windowed MLE does not exist** here, since a trailing window always separates somebody, so a population prior is required and renormalizing the mean does not substitute for it. Default `--compare peer` scores one x-game rating against a second independent one, which is like-for-like against a measurement whose other side is a self-declaration; `--compare truth` is the floor. `--targets` (the QA CSV) `--scope --link --population --games --compare`.
- **`eval_rank_map_cv.py`** — the paired held-out CV harness behind the mapping's design choices: it validated the 2-month career-block anchors and rejected errors-in-variables weighting over each rating's CI.
- **`map_rating_to_rank_dev.py`** — the experimental copy of `pipeline.map_rating_to_rank`; writes nothing by default so it can never clobber the shipped curve. Where mapping experiments (time forms, gauge, coupling structure) are run.
- **`eval_recall_model.py`** — validates the shipped retention model against the attempt log: per-rung realized retention against the window ceiling, within-user contrasts, delivery. Produces every figure in `docs/methodology.md` §2.8's closing subsection; re-run it to refresh them.
- **`eval_rank_endpoint_bias.py`** — picks `SLOPE_PRIOR_SD` for the Rank-tab graph's local-linear boundary fix and measures how much right-edge lag it removes.

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

- **`generate_invite.py`** — mint an EGD-leaderboard invitation: validates an EGD PIN against the European Go Database, inserts the `invitation` row via wrangler, and prints the `/register?t=` link. A non-numeric argument is a name search instead (no DB write). The in-app equivalent is the `GET /invite` page (`pwa/src/handlers/invite-tool.js`). `<EGD_PIN>|<name...> [--local]`. Stdlib only.

## Skill-axis inspection

- **`rating_dev/inspect_residual_axes.py`** — renders the extreme-loading problems on each production axis, both tails, straight from `axes/per_problem.csv`. This is the motif-labeling tool: `build_axes` grows an axis, you eyeball its tails here, name them in `axes/labels.json`, re-run `package_for_pwa` (which REFUSES to ship an axis with neither tail labeled). `--per-problem --size --rows --cols --open`.
- **`inspect_axis_inputs.py`** — the other half: which INPUT columns drive an axis and how concentrated that reliance is (participation ratio), each tail decomposed additively, plus correlation against every frozen anchor so a new axis can be placed as a known motif, a near-miss, or genuinely new. `--quick` reads `axes.npz` alone.
- **`eval_score_gap_axis.py`** — between-user predictive power of the best-vs-second-best score-gap axis (`--feature wr_gap` for the winrate comparator). Also the loader battery `rating_dev/train_residual_axes.py` imports.

---

## `rating_dev/`

The rating-model research package. `pipeline/train_rating.py` is deliberately CLI-free with its config baked in; all experimentation happens here so a test tweak cannot perturb the shipped ratings. It imports only `utils/` and its own modules.

- **`train_rating_staged.py`** — the experiment surface and production superset: `--production-flags` reproduces the shipped configuration, and the rest of the flag set (gauge, divergence barriers, EIV probes, the staged/M0b legacy configs, the eval suite, the alternative objectives incl. `--even-error`) is where the model's design decisions were measured. `--staged` fits the per-problem 4PL heads one at a time instead of jointly.
- **`train_residual_axes.py`** — the research surface for the skill axes; `pipeline/build_axes.py` is the production entry point and `utils/rating/residual_axes.py` the shared model.
- **`validate_ratings.py`** — the single home of rating validation. Scores a ratings directory on a hash-confirmed holdout: held-out-problem logloss, held-out Δskill flatness, survey-ρ (base + time-adjusted). `--ratings-dir --seed --holdout-problem-frac [--expect-hash]`.
- **`holdout_set.py`** — deterministic held-out problem-set generator: `(seed, fraction)` → a reproducible subset plus a 16-hex content hash, so trainer and bench can prove they used the same set. `--seed --frac [--out file]`.
- **`eval_phase2_units.py`** — the certificate harness for the phase-2 global scale fit: regenerates outcomes from the fitted model at a known true scale and checks what the estimator returns. This is what established that the slope scale is real and the floor scale was a Neyman–Scott artifact. `--synthetic-outcomes`, unit-granularity and drift sweeps.
- **`model_comparison.py`** — shared data/feature loaders for the harness (re-exported from `utils.rating.features`) plus `load_problem_features`. No CLI.
