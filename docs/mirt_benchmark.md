# MIRT benchmark: provenance and results

The fixed reference point for judging whether a change to `auxiliary/rating_dev/train_joint_mirt.py` is progress. Every number below is reproducible from the provenance in the first section; a future run that changes any of it is measuring something else and must say so.

Development only. Nothing here ships, and `pipeline.train_rating` is untouched.

## Provenance — pin these before comparing anything

```
attempts DB      db/d1.sqlite
  sha256[:16]    267d103823b924a8
  rows           387,232 tsumego_reports   max id 405,591
  window         2025-08-20T23:04:55.761Z .. 2026-08-10T13:51:57.760Z
  rank_surveys   624 rows
corpus           153,720 problems with features; 135,434 with >=1 first-time attempt

held-out PROBLEM sets — auxiliary/rating_dev/holdout_set.py, frac 0.15
  seed 0         hash 6619fdc487345fc0   20,315 problems (9: 16,189  13: 4,126)
  seed 1         hash fc33d7e4c2dffc55   20,315 problems (9: 16,206  13: 4,109)
```

**A pinned copy of that DB is kept at `db/d1.baseline-267d103823b924a8.sqlite`** (gitignored via `/db/d1.baseline-*.sqlite`, verified byte-identical at copy time). `pipeline.fetch_database` treats `db/d1.sqlite` as its merge target and overwrites it in place, so without this copy the first routine fetch would silently retire the baseline — the held-out draw would move, the hashes would change, and nothing already recorded here could be reproduced or extended. To restore: `cp db/d1.baseline-267d103823b924a8.sqlite db/d1.sqlite`, then confirm `holdout_set --seed 0 --frac 0.15` still prints `6619fdc487345fc0`.

**The DB hash is part of the identity of every result here.** `holdout_set` draws from the problems that have attempts, so a `pipeline.fetch_database` run changes the universe, changes the draw, and changes the hash — which is the intended tripwire, not a nuisance. Both models print the hash they trained under; if the two differ, the comparison is void.

Reproduce:

```bash
./.venv/bin/python3 -m auxiliary.rating_dev.holdout_set --seed 0 --frac 0.15
# production
./.venv/bin/python3 -m auxiliary.rating_dev.train_rating_staged \
    --production-flags --holdout-problem-frac 0.15 --seed 0 \
    --out-dir /tmp/prod_ho --validate
# MIRT, production's ability structure
./.venv/bin/python3 -m auxiliary.rating_dev.train_joint_mirt \
    --k 3 --seeds 0 --window-months 2 --holdout-seed 0 \
    --dump-params /tmp/joint_params_prodwindow_s0.npz
# production's axis layer on the same rows
./.venv/bin/python3 -m auxiliary.rating_dev.prod_axis_heldout \
    --prod-dir /tmp/prod_ho --seed 0 --frac 0.15 --k 3
```

## Results — held-out USERS, 2 folds, paired per fold — MEASURED 2026-08-12

The user holdout is the criterion from here; the problem- and row-holdout tables below are kept as what they are, measurements of different questions. Read the warm-start leak section first — it is one-sided and these numbers carry it.

Production is `train_rating_staged --production-flags`, which has no phase 2. All three logloss rows are PARTITIONS OF ONE one-step-ahead prequential walk from each held-out user's first attempt — ability from a running grid posterior, axis position `v` learned online from zero — split by whether training saw the problem (81% seen / 19% not). They are not three protocols.

`± sd` across two folds is fold difficulty, not the model difference; both models are scored on identical folds with identical denominators, so the PAIRED per-fold gap is the statistic.

```
rating only                f0 prod -> MIRT      f1 prod -> MIRT     MIRT-prod (mnat)
  forward walk (all)       0.4955 -> 0.4929     0.5031 -> 0.5000      -2.6 / -3.1
    seen      (81%)        0.4890 -> 0.4863     0.4928 -> 0.4889      -2.7 / -3.9
    never trained (19%)    0.5237 -> 0.5217     0.5463 -> 0.5462      -2.0 / -0.1

rating + axis
  forward walk (all)       0.4917 -> 0.4920     0.5001 -> 0.4993      +0.3 / -0.8
    seen                   0.4872 -> 0.4865     0.4913 -> 0.4896      -0.7 / -1.7
    never trained          0.5114 -> 0.5158     0.5364 -> 0.5398      +4.4 / +3.4

axis adds (mnat)              production            MIRT
  forward walk (all)         3.82 / 3.09         0.93 / 0.73
    seen                     1.85 / 1.44        -0.22 / -0.63
    never trained           12.33 / 9.96         5.95 / 6.37
```

**Joining the fits improves the rating and degrades the axes, and the two nearly cancel.** MIRT's rating head leads by ~2.9 mnat (6/6 paired), production's axis layer leads by ~2.6 mnat (6/6), and the combined systems are indistinguishable — the only comparison whose folds disagree on sign is the one where the two effects meet. The sharpest localization of the axis failure: MIRT's axes HURT on problems training saw, both folds, while production's help.

**Survey rho is not usable on this holdout.** It is restricted to held-out accounts' declarations — 92 on seed 0, 67 on seed 1, against the 374 that gave the ±0.012 resolution quoted below. Production 0.9108 / 0.8061, MIRT 0.9167 / 0.7766: the fold-to-fold swing of 0.10 dwarfs both model differences, which flip sign between folds. Do not quote it.

## THE WARM-START LEAK — found 2026-08-13, affects every MIRT number in this document

`train_joint_mirt` initializes its axis map from `eigen_directions`, which needs an axis-free 4PL to take residuals against. A single-stage model has none at initialization, so it calls `build_axis_features()` with no `params_dir` — reading the **shipped `ratings/`**, a two-stage fit trained on every user and every problem. Two independent leaks follow, both favouring the MIRT:

- **User holdout.** `--holdout-user-frac` masks rows from the SGD training set only. It never reaches `rbu`, `ax_users` or `tu_ax`, which are filtered by the *axis* fold alone, so held-out accounts warm-start the model directly: **66 of 383 warm-start users (17.2%, 31,931 attempts) on seed 0, and 52 of 383 (13.6%, 20,589) on seed 1.**
- **Problem holdout.** The shipped per-problem parameters were fitted on all 153,720 problems, the 20,315 held-out ones included, and `rbu` carries attempts on them.

Production's side is clean in both: `fit_axes_userholdout` excludes held-out users and reads the matching per-fold rating fit, and `prod_axis_heldout` excludes held-out problems. So the leak is one-sided, and the corrections run in one direction — **the MIRT's rating advantage is inflated and its axis deficit understated.** The axis conclusions therefore survive as conservative; the rating conclusions do not, and no rating figure here should be quoted as a clean measurement until a leak-free warm start is in place.

The rank curve is a second, smaller leak shared by BOTH models: `build_axis_features` reads the shipped `rating_to_rank.json` for its rank column regardless of `params_dir`.

The fix is `auxiliary/rating_dev/burn_in_warm_start.py`, wired in as `--warm-start burnin`: the warm start derived inside the pass. At `v = 0` the loading gradient is identically zero, so a burn-in with `v` pinned at zero trains exactly the axis-free baseline the residual scores are defined against, and taking them over training rows only excludes held-out users and rows by construction rather than by a filter. It reads no shipped artifact, so the rank-curve leak leaves the MIRT with it.

A second defect of the borrowed form, independent of the leak: `eigen_directions` works in the 216-column axis feature space and the result is truncated by `gi_full[:, :fdim]` to the MIRT's 212 **without renormalising**, so the initialization is a non-unit projection of directions chosen partly along four coordinates this model does not have — including the two-stage model's own rating and slope.

### The ablation — MEASURED 2026-08-13. The warm start stays; the burn-in does not replace it yet.

At the swept optimum `3e-4 / 1e-2`, held-out users, holdout seed 0:

```
                    rating            axis adds (mnat)
  eigen  (2 seeds)  0.4930 +/- 0.0003   +0.68 +/- 0.19
  burnin (2 seeds)  0.4944 +/- 0.0021   +0.22 +/- 1.20
  none   (3 seeds)  0.4932 +/- 0.0001   -2.64 +/- 0.86
```

**The warm start earns its place** — `eigen` beats random init by 3.3 mnat on the axis column, with a tight seed spread, and all three `none` seeds put the axes firmly negative. So the leak cannot be resolved by deleting `eigen_directions`. Note the value is entirely axis-side: the rating is the same to within 0.4 mnat whether or not `g` is warm-started, which is what the mechanism predicts, since at `v = 0` the loading gradient vanishes and the warm start is a statement about `g` alone.

**The burn-in at `BURN_IN_EPOCHS = 100` is not a usable replacement.** It is worse than both alternatives on rating and its seed spread is ~7x wider on rating and ~6x wider on axes. The suspected mechanism is that the residuals are taken against a baseline nowhere near converged — the stopping diagnostic puts held-out rating at 0.4980 at epoch 100 against 0.4928 at 300 — so they carry rating error rather than skill structure. Centring removes the direction in which the baseline mispredicts for EVERYONE; an undertrained baseline also mispredicts in user-specific ways correlated with ability, which centring cannot touch, and those seed `g` with directions that duplicate the rating. That fits the symptom: the damage lands on the rating, not only on the axes.

`--burn-in` makes the length a knob so this is testable. If a converged baseline closes the gap the mechanism is sound and 100 was simply wrong; if it does not, the warm start's value depends on having a converged EXTERNAL fit, which is a genuine problem for the single-stage thesis rather than a tuning matter.

Until then the default stays `eigen`, the leak stands, and **no rating figure in this document is a clean measurement.**

## Results — held-out samples, 2 folds, mean ± sd

81,439 rows (seed 0) and 81,030 rows (seed 1), identical for both models: same problems, same profile structure, same denominators.

```
                    rating only        rating + axis      axis adds (mnat)
  production      0.4952 ± 0.0000    0.4971 ± 0.0006    -1.876 ± 0.516
  MIRT            0.4986 ± 0.0009    0.4984 ± 0.0008    +0.142 ± 0.030
  MIRT shortfall    3.39 ± 0.83        1.32 ± 1.44

  per fold:  rating only   production 0.4952 / 0.4952   MIRT 0.4992 / 0.4980
             rating+axis   production 0.4967 / 0.4975   MIRT 0.4990 / 0.4978
  constant-rate baseline   0.5253 / 0.5266  (base rate 0.781 / 0.780)
```

**Rating alone: production is ahead by 3.39 ± 0.83 mnat**, both folds agreeing in direction, and production is fold-invariant to four decimals. Read it against the ~30 mnat of total signal available over a best-constant predictor: the MIRT delivers about 87% of production's discrimination.

**Rating + axis: unresolved.** The interval spans zero. Production's constructed axis layer costs it 1.88 mnat while the MIRT's gains 0.14, and that swing closes most of the gap. Per fold the combined shortfall is 2.34 and 0.31 — the two draws disagree by more than the effect.

`n = 2` makes "± sd" a range with a symbol on it. The rating column survives that (gap is 4x its spread); the combined column does not, and needs a third and fourth fold before anyone quotes 1.32.

**Production has no native rating+axis prediction** — its axes feed the client's weakness pane and never enter the rating. That column is constructed (`prod_axis_heldout.py`): its frozen 4PL with `sum g*v` added to the logit, `v` fitted per user on training rows, `v = 0` recovering rating-only exactly. Every link excludes the held-out problems — the 4PL, the axis features (which supply four feature columns AND the axis baseline), the axis fit, and `v`. One residual leak, stated not hidden: the rank feature column still comes from the shipped `rating_to_rank.json`, a global monotone map fitted on survey declarations rather than held-out outcomes.

## Survey rho — the external check

Never used in training by either model. `survey_weighted_rho`, per-server Spearman averaged with user-count weights, alpha cross-fitted per run.

```
                    base rho             time-adjusted
  production      0.8672 / 0.8690      0.8903 / 0.8931   (alpha -0.60 / -0.43)
  MIRT            0.8675 / 0.8681      0.8880 / 0.8918   (alpha -0.30 / -0.43)
  MIRT shortfall  -0.0003 / +0.0009    0.0023 / 0.0013
```

**Indistinguishable, and it cannot say more.** The paired user bootstrap (`compare_survey_rho.py`) puts this statistic's resolution at about **±0.012** with 189 users and 374 declarations, an order of magnitude above the 0.0018 difference. Quote the TIME-ADJUSTED figure: every server coupling was fitted against time-adjusted ratings, so the untimed number compares two sides in different units and fails silently.

Note what this does and does not corroborate. Survey rho reads `rating` and `slope` only — no axes — so it is a RATING-side criterion. It agrees with the rating+axis column in *verdict* (both unresolved) but for unrelated reasons: the combined column is unresolved through fold variance, survey rho through low power. It does NOT corroborate the rating-only result, which held-out logloss resolves and survey rho cannot see at all. When two criteria disagree, the one with power decides.

## Three holdout forms, and what each answers

One generator (`auxiliary/rating_dev/holdout_set.py`), one hash discipline, wired into both trainers. They are not a hierarchy — they are three different requirements, and the two models are NOT uniformly ordered across them.

```
kind      seed 0 hash          size                     asks
problem   6619fdc487345fc0     20,315 problems          can it price content nobody has played?
user      2ac648dc60813366     576 accounts             can it serve someone new, from attempt 1?
row       db89466ad768dd9b     50,997 attempts          can it predict a known user's next attempt?
```

The ROW holdout is the client's steady-state condition: the problem keeps its other attempts and the user keeps their other rows, so parameters are informed exactly as in deployment and only the scored attempt is unseen. The PROBLEM holdout is the opposite extreme (parameters from the feature prior alone) and the USER holdout is a blend — measured at **79% of its scored attempts on problems that ARE in training, 21% on problems that are not**, near-identical across two folds (79.0/21.0 and 78.8/21.2).

All evaluations here use `load_first_time_attempts`, which filters `is_repeat = 0` and then dedups to the earliest attempt per (user, problem). Training and scoring are NOVEL attempts throughout: one row per user per problem, the first time they saw it.

## Phase 2 does not survive a leak-free test — MEASURED 2026-08-12

`pipeline.train_rating`'s phase 2 fits a global slope multiplier (shipped s = 1.694) and reports 7.181 mnat, in-sample. `utils.rating.slope_calibration`'s walk-forward returns 1.667 and is described as the client-condition cross-check — **but its per-problem parameters come from a fit trained on every first-time attempt, including the rows it then scores.** Those predictions are partly memorised, memorised predictions are more accurate, and more accuracy rewards more confidence, so the optimal steepness under that protocol is biased upward. The synthetic certificate (0.993 on outcomes from TRUE parameters) does not cover this: it validates the estimator when the parameters are correct, not when they were fitted on the scored rows.

Re-run on users held out of training entirely — same WINDOW/WARMUP/EVAL_BLOCK, same mode-'slope' application, same profile-likelihood theta refit, differing in exactly one respect:

```
                                    optimal s    cost of shipping 1.694
  fold 0  (576 users, 38,639 att)      0.90           13.4 mnat
  fold 1  (576 users, 32,018 att)      0.95            9.6 mnat
```

Both folds land essentially at unity, both curves single-minimum. `auxiliary/rating_dev/walk_forward_heldout.py`.

**A caution against a tempting but wrong explanation.** It is NOT the case that attempted problems are better priced and therefore tolerate steeper slopes: per-problem parameters come from a shared FEATURE NET, not free per-item parameters, so an attempt reaches its own problem's rating only diffusely through the net. That story requires the net to fingerprint individual problems — precisely what `INPUT_DROPOUT` at column granularity exists to suppress. The leak is in the ESTIMATOR's protocol, not in per-problem parameter quality.

Note `train_rating_staged --production-flags` has no phase 2, so every production number in this document is ALREADY phase-2-free and removing the stage would not move them. What it changes is the shipped artifact: `slope_scale` in `rating_constants.json`, shipped as `quality = s*slope`. Dropping it divides every shipped slope by 1.694, cutting client-side information ~2.9x, which moves `se`, rank convergence, the SE floor curve and the serve gap for every user.

## K is hardcoded in the MIRT

`build_axes` selects K incrementally (K_START 3 to K_MAX 12, growing while held-out gain stays within STOP_FRACTION of the peak; currently selects 3). The MIRT takes `--k`, default 3 — no search, no criterion. It is inheriting production's selected answer, which is right today and silently wrong as soon as the data supports a fourth axis. Matched-K comparisons here are fair as comparisons of ARCHITECTURE; they are not comparisons of the two SYSTEMS, one of which grows K and one of which does not.

## The framework

```
holdout_set.py            the shared problem holdout; hash-confirmed on both sides
train_rating_staged.py    --production-flags --holdout-problem-frac  (production)
train_joint_mirt.py       --holdout-seed --window-months --dump-params  (MIRT)
prod_axis_heldout.py      production's constructed rating+axis on the same rows
compare_heldout_logloss.py  paired logloss, user-CLUSTERED bootstrap
compare_survey_rho.py     paired survey rho, user bootstrap
```

Both bootstraps are **paired** and cluster by user. Attempts within a user share an ability estimate, so a row-level resample understates the interval — in the direction that makes a difference look real.
