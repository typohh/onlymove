"""Per-user estimate of whether spending more solving time helps — controlling
for the 3PL probability of solving the problem at all.

Motivation
----------
The PWA's "you rushed it" / "you didn't review enough" overlays currently fire
on the *sign* of a per-user time coefficient (point estimate > 0), which is a
coin-flip threshold. The honest question is: for each user, what is the
probability that *additional solving time actually benefits them*, and how
tightly is that pinned down (standard error)? This script answers that offline
from the logged attempts, so we can see the population of per-user time effects
and decide a sane confidence gate.

The key confound is difficulty: harder problems take longer AND are failed more,
which by itself makes "more time" look harmful. So we don't regress success on
time naively. We give every attempt a 3PL baseline from the *shipped* problem
parameters and the user's own fitted ability, then ask whether time explains
residual variation on top of that baseline.

Model (per user u, over their NOVEL timed attempts i)
-----------------------------------------------------
    η_i = slope_i · (θ_u − rating_i) + κ(floor_i) + β_u · z_i
    P_i = floor_i + (1 − floor_i) · σ(η_i)

  - rating_i, slope_i, floor_i: the per-problem 3PL params shipped to the PWA
    (shard metadata.json: `rating`, `slope`, `floor`), used in RAW-rating space
    (no rating→rank curve needed — we never leave raw units).
  - κ(floor) = logit((0.8 − floor)/(1 − floor)) — the same `clearKappa` the
    client uses so P = 0.8 at θ == rating (CALIBRATION_P = 0.8). floor clamped
    to [0, 0.78] (FLOOR_MAX) exactly as rank-decomposition.js does.
  - θ_u: the user's ability (raw-rating units), a free parameter.
  - z_i: the user's solving time for attempt i, as winsorized log10(seconds),
    CENTERED WITHIN USER (so θ_u absorbs their average pace) and scaled by the
    GLOBAL sd of log-time (so β_u is in common units across users).
  - β_u: THE QUANTITY OF INTEREST — the effect of spending one global-sd more
    log-time than the user's own norm, in latent-logit units. β_u > 0 ⇒ the user
    benefits from extra time after difficulty is accounted for.

η is linear in (θ_u, β_u) with offset (κ_i − slope_i·rating_i) and design
[slope_i, z_i], so we fit by Fisher scoring (IRLS) with the 3PL lower asymptote.
SE(β_u) comes from the inverse Fisher information at convergence.

We deliberately do NOT replicate the client rank machinery (kernel-weighted
time-varying R, recall model, picked-move embeddings). That's both unnecessary
here and impossible historically — solving time was only logged starting May
2026 and the move/embedding fields are sparse. θ_u here is a single static
ability fit from the timed window, which is the right baseline for "did time
help on these attempts".

Caveats (printed in the report too):
  - Observational. β_u is an association, not a causal time-dose effect. Reverse
    causation survives the difficulty control: a problem that *feels* easy to a
    user is answered fast AND correctly (pushes β down); one they flail on eats
    time AND fails (also β down). The 3PL removes the part explained by the
    problem's objective difficulty + the user's ability, not the subjective part.
  - Single month of data (time logging began 2026-05); ~300 users qualify.

Run:
  ./.venv/bin/python3 -m auxiliary.analyze_time_benefit
  ./.venv/bin/python3 -m auxiliary.analyze_time_benefit --size 9 --min-attempts 30
  ./.venv/bin/python3 -m auxiliary.analyze_time_benefit --plot
"""

import argparse
import json
import os
import sqlite3
import sys
import zipfile
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Client-matched 3PL constants (pwa/js/rank-decomposition.js) ---
CALIBRATION_P = 0.8
FLOOR_MAX = 0.78


def clear_kappa(c):
    """Inner-sigmoid offset so the full 3PL P == CALIBRATION_P at θ == rating."""
    c = np.clip(c, 0.0, FLOOR_MAX)
    if not np.any(c > 0):
        return np.full_like(np.asarray(c, dtype=float),
                            np.log(CALIBRATION_P / (1 - CALIBRATION_P)))
    q = np.clip((CALIBRATION_P - c) / (1 - c), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_attempts(db_path, size_filter):
    """Novel, timed attempts. Returns list of dicts."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    q = ("SELECT user_id, problem, success, size, time_used "
         "FROM tsumego_reports "
         "WHERE is_repeat = 0 AND time_used IS NOT NULL AND time_used > 0")
    params = ()
    if size_filter:
        q += " AND size = ?"
        params = (size_filter,)
    rows = con.execute(q, params).fetchall()
    con.close()
    out = []
    for r in rows:
        out.append({
            "user": r["user_id"],
            "tid": r["problem"],
            "y": 1 if r["success"] else 0,
            "size": int(r["size"]),
            "sec": r["time_used"] / 1000.0,
        })
    return out


def load_problem_params(tsumegos_dir, needed):
    """Scan shard zips for (size, tid) in `needed`; return {(size,tid): (rating,
    slope, floor)}. `needed` is a set of (size, tid)."""
    params = {}
    by_size = defaultdict(set)
    for (size, tid) in needed:
        by_size[size].add(tid)
    for size, tids in by_size.items():
        size_dir = os.path.join(tsumegos_dir, str(size))
        if not os.path.isdir(size_dir):
            print(f"  WARNING: {size_dir} missing, skipping size {size}")
            continue
        remaining = set(tids)
        shards = [f for f in os.listdir(size_dir) if f.endswith(".zip")]
        for shard in shards:
            if not remaining:
                break
            try:
                with zipfile.ZipFile(os.path.join(size_dir, shard)) as zf:
                    meta = json.loads(zf.read("metadata.json"))
            except (KeyError, zipfile.BadZipFile, json.JSONDecodeError):
                continue
            for item in meta:
                tid = item.get("filename", "")[:-4]  # strip ".sgf"
                if tid in remaining:
                    slope = item.get("slope")
                    rating = item.get("rating")
                    floor = item.get("floor", 0.0)
                    if (slope is None or rating is None
                            or not np.isfinite(slope) or slope <= 0
                            or not np.isfinite(rating)):
                        remaining.discard(tid)
                        continue
                    params[(size, tid)] = (
                        float(rating), float(slope),
                        float(np.clip(floor, 0.0, FLOOR_MAX)))
                    remaining.discard(tid)
        if remaining:
            print(f"  size {size}: {len(remaining)} attempted problems had no "
                  f"shard params (dropped)")
    return params


# ---------------------------------------------------------------------------
# Per-user fit
# ---------------------------------------------------------------------------
def fit_user(theta_design, z, y, rating, slope, floor, kappa,
             lam_theta, lam_beta, max_iter=60):
    """Fisher-scoring fit of (theta, beta). Returns (theta, beta, se_beta, ok).

    Linear predictor η = slope*theta + z*beta + (kappa - slope*rating).
    Design column for theta is `slope` (=theta_design), for beta is `z`.
    """
    offset = kappa - slope * rating
    one_mc = 1.0 - floor
    w = np.array([0.0, 0.0])  # [theta, beta]
    Lam = np.diag([lam_theta, lam_beta])
    X = np.column_stack([theta_design, z])  # n x 2
    for _ in range(max_iter):
        eta = X @ w + offset
        praw = sigmoid(eta)
        P = floor + one_mc * praw
        P = np.clip(P, 1e-9, 1 - 1e-9)
        g = one_mc * praw * (1 - praw)            # dP/dη
        denom = P * (1 - P)
        s = (y - P) * g / denom                   # per-obs score wrt η
        v = g * g / denom                         # per-obs Fisher info wrt η
        score = X.T @ s - Lam @ w
        info = (X * v[:, None]).T @ X + Lam
        try:
            step = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            return (np.nan, np.nan, np.inf, False)
        # damp theta steps so a near-separable user can't blow up
        step = np.clip(step, -5.0, 5.0)
        w = w + step
        w[0] = np.clip(w[0], -25.0, 25.0)
        if np.max(np.abs(step)) < 1e-7:
            break
    # Covariance from inverse Fisher info at convergence (includes weak prior).
    eta = X @ w + offset
    praw = sigmoid(eta)
    P = np.clip(floor + one_mc * praw, 1e-9, 1 - 1e-9)
    g = one_mc * praw * (1 - praw)
    v = g * g / (P * (1 - P))
    info = (X * v[:, None]).T @ X + Lam
    try:
        cov = np.linalg.inv(info)
        se_beta = float(np.sqrt(max(cov[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        return (float(w[0]), float(w[1]), np.inf, False)
    return (float(w[0]), float(w[1]), se_beta, True)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def quantile_line(label, arr, qs=(5, 10, 25, 50, 75, 90, 95)):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return f"{label:<22} (no data)"
    vals = np.percentile(arr, qs)
    cells = "  ".join(f"p{q}={v:+.3f}" for q, v in zip(qs, vals))
    return f"{label:<22} n={arr.size:<5d} {cells}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.join(PROJECT_ROOT, "attempts",
                                                 "database.sqlite"))
    ap.add_argument("--tsumegos-dir", default=os.path.join(PROJECT_ROOT, "pwa",
                                                           "tsumegos"))
    ap.add_argument("--size", type=int, default=None,
                    help="restrict to one board size (9 or 13); default both")
    ap.add_argument("--min-attempts", type=int, default=20,
                    help="min novel timed attempts (with shard params) per user")
    ap.add_argument("--winsor", type=float, default=1.0,
                    help="winsorize solving seconds to [p, 100-p] percentiles")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="HARD-drop attempts slower than this (AFK guard, true "
                         "exclusion not clipping)")
    ap.add_argument("--min-seconds", type=float, default=None,
                    help="HARD-drop attempts faster than this (misclick guard)")
    ap.add_argument("--lam-theta", type=float, default=0.01,
                    help="weak ridge on ability (prior sd ~10 rating units)")
    ap.add_argument("--lam-beta", type=float, default=0.01,
                    help="weak ridge on the time effect (numerical PD)")
    ap.add_argument("--out-csv", default=os.path.join(PROJECT_ROOT, "auxiliary",
                                                      "time_benefit_users.csv"))
    ap.add_argument("--plot", action="store_true",
                    help="also write histogram PNGs to auxiliary/")
    args = ap.parse_args()

    print(f"Loading attempts from {args.db} ...")
    attempts = load_attempts(args.db, args.size)
    print(f"  {len(attempts)} novel timed attempts")
    if not attempts:
        print("No usable attempts. (time_used is sparse — logged from 2026-05.)")
        return

    needed = {(a["size"], a["tid"]) for a in attempts}
    print(f"Loading 3PL params for {len(needed)} distinct problems ...")
    params = load_problem_params(args.tsumegos_dir, needed)
    print(f"  resolved {len(params)} / {len(needed)}")

    # Attach params; drop attempts whose problem has no shipped params, and
    # HARD-exclude AFK / misclick times if requested (true exclusion, applied
    # BEFORE winsorization so the percentile anchors are computed on clean data).
    rows = []
    n_afk = n_fast = 0
    for a in attempts:
        if args.max_seconds is not None and a["sec"] > args.max_seconds:
            n_afk += 1
            continue
        if args.min_seconds is not None and a["sec"] < args.min_seconds:
            n_fast += 1
            continue
        p = params.get((a["size"], a["tid"]))
        if p is None:
            continue
        rating, slope, floor = p
        rows.append((a["user"], a["sec"], a["y"], rating, slope, floor))
    if args.max_seconds is not None or args.min_seconds is not None:
        print(f"  hard-excluded {n_afk} attempts > {args.max_seconds}s (AFK), "
              f"{n_fast} attempts < {args.min_seconds}s (misclick)")
    print(f"  {len(rows)} attempts with params")

    # Global winsor + log of solving time.
    secs = np.array([r[1] for r in rows])
    lo = np.percentile(secs, args.winsor)
    hi = np.percentile(secs, 100 - args.winsor)
    secs_w = np.clip(secs, lo, hi)
    log_t = np.log10(secs_w)
    global_sd = float(np.std(log_t))
    print(f"  winsor seconds to [{lo:.1f}, {hi:.1f}]; "
          f"log10-time global sd = {global_sd:.3f}")

    # Group by user.
    by_user = defaultdict(list)
    for i, r in enumerate(rows):
        by_user[r[0]].append(i)

    log_t = np.asarray(log_t)
    y_all = np.array([r[2] for r in rows])
    rating_all = np.array([r[3] for r in rows])
    slope_all = np.array([r[4] for r in rows])
    floor_all = np.array([r[5] for r in rows])
    kappa_all = clear_kappa(floor_all)

    results = []  # dicts per fitted user
    n_skip_small = n_skip_novar = n_skip_fit = 0
    for user, idx in by_user.items():
        idx = np.array(idx)
        if idx.size < args.min_attempts:
            n_skip_small += 1
            continue
        L = log_t[idx]
        # within-user center, global-sd scale
        z = (L - L.mean()) / max(global_sd, 1e-9)
        if np.std(L) < 1e-6:
            n_skip_novar += 1
            continue
        y = y_all[idx]
        rating = rating_all[idx]
        slope = slope_all[idx]
        floor = floor_all[idx]
        kappa = kappa_all[idx]
        theta, beta, se, ok = fit_user(slope, z, y, rating, slope, floor, kappa,
                                       args.lam_theta, args.lam_beta)
        if not ok or not np.isfinite(se) or se <= 0:
            n_skip_fit += 1
            continue
        results.append({
            "user": user,
            "n": int(idx.size),
            "success_rate": float(y.mean()),
            "theta": theta,
            "median_sec": float(np.median(secs[idx])),
            "log_t_sd": float(np.std(L)),
            "beta": beta,
            "se_beta": se,
            "z_beta": beta / se,
        })

    print(f"\nFitted {len(results)} users "
          f"(skipped: {n_skip_small} <{args.min_attempts} attempts, "
          f"{n_skip_novar} no time variance, {n_skip_fit} fit failures)")
    if not results:
        return

    from scipy.special import ndtr  # standard normal CDF

    beta = np.array([r["beta"] for r in results])
    se = np.array([r["se_beta"] for r in results])
    zscore = beta / se
    # Per-user probability the true time-benefit is positive, under the normal
    # sampling distribution N(estimate, se): P(>0) = Φ(estimate / se).
    prob_pos = ndtr(zscore)
    for r, pp in zip(results, prob_pos):
        r["prob_positive"] = float(pp)

    # ---- Write CSV ----
    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Wrote per-user estimates → {args.out_csv}")

    # ---- SE distribution ----
    print("\n" + "=" * 78)
    print("STANDARD-ERROR DISTRIBUTION  (SE of the per-user time effect β)")
    print("=" * 78)
    print(quantile_line("SE(beta)", se))
    print(f"  Interpretation: β is in latent-logit units per (global-sd of\n"
          f"  log10 solving time). Smaller SE = the user has enough timed\n"
          f"  attempts AND enough spread in their own pace to pin the effect.")

    # ---- β distribution (unfiltered) ----
    print("\n" + "=" * 78)
    print("USER TIME-EFFECT DISTRIBUTION  (β; >0 ⇒ more time helps)")
    print("=" * 78)
    print(quantile_line("beta (all fitted)", beta))
    # inverse-variance (fixed-effect) pooled mean = "does the average user benefit"
    wts = 1.0 / se**2
    pooled = float(np.sum(wts * beta) / np.sum(wts))
    pooled_se = float(np.sqrt(1.0 / np.sum(wts)))
    print(f"\n  Inverse-variance pooled β = {pooled:+.4f} ± {pooled_se:.4f}  "
          f"(z = {pooled/pooled_se:+.2f})")
    print(f"  Fraction with point estimate β > 0: "
          f"{100*np.mean(beta > 0):.1f}%")

    # ---- Min-SE requirement sweep ----
    print("\n" + "=" * 78)
    print("β DISTRIBUTION UNDER MINIMUM-PRECISION (max-SE) REQUIREMENTS")
    print("=" * 78)
    print(f"{'max SE':>8}  {'n users':>7}  {'med β':>7}  {'%β>0':>6}  "
          f"{'%sig+':>6}  {'%sig-':>6}  {'pooled β':>9}")
    print(f"{'(95% CI excludes 0 ⇒ sig+/sig-)':^58}")
    for se_max in [np.inf, 2.0, 1.0, 0.7, 0.5, 0.3, 0.2]:
        m = se <= se_max
        if not np.any(m):
            print(f"{se_max:>8}  {'0':>7}")
            continue
        b = beta[m]
        s = se[m]
        sig_pos = np.mean(b - 1.96 * s > 0) * 100
        sig_neg = np.mean(b + 1.96 * s < 0) * 100
        wm = float(np.sum(b / s**2) / np.sum(1.0 / s**2))
        tag = "  inf" if not np.isfinite(se_max) else f"{se_max:.2f}"
        print(f"{tag:>8}  {m.sum():>7d}  {np.median(b):>+7.3f}  "
              f"{100*np.mean(b > 0):>5.0f}%  {sig_pos:>5.0f}%  "
              f"{sig_neg:>5.0f}%  {wm:>+9.4f}")

    # ---- Empirical distribution of P(time-benefit > 0) ----
    print("\n" + "=" * 78)
    print("EMPIRICAL DISTRIBUTION OF P(time-benefit > 0)  (per-user, = Φ(est/se))")
    print("=" * 78)

    def prob_pos_block(mask, label):
        pp = prob_pos[mask]
        if pp.size == 0:
            print(f"{label}: (no users)")
            return
        print(f"\n{label}  (n={pp.size})")
        print("  " + quantile_line("P(>0) quantiles", pp))
        # binned histogram across [0,1]
        edges = np.linspace(0, 1, 11)
        counts, _ = np.histogram(pp, bins=edges)
        print("  histogram (bin width 0.1):")
        for k in range(10):
            bar = "#" * int(round(40 * counts[k] / max(counts.max(), 1)))
            print(f"    [{edges[k]:.1f},{edges[k+1]:.1f})  {counts[k]:>4d}  {bar}")
        print("  share of users with P(>0) at least:")
        for thr in (0.5, 0.8, 0.9, 0.95, 0.99):
            print(f"    ≥ {thr:.2f}: {np.mean(pp >= thr)*100:5.1f}%  "
                  f"({int(np.sum(pp >= thr))} users)")
        # symmetric: confidently NEGATIVE (benefit < 0)
        for thr in (0.95, 0.99):
            print(f"    ≤ {1-thr:.2f} (i.e. P(<0) ≥ {thr:.2f}): "
                  f"{np.mean(pp <= 1-thr)*100:5.1f}%  "
                  f"({int(np.sum(pp <= 1-thr))} users)")

    prob_pos_block(np.ones(len(prob_pos), dtype=bool), "ALL FITTED USERS")
    prob_pos_block(se <= 0.5, "WELL-MEASURED USERS (SE ≤ 0.5)")

    # ---- Structural estimate of the difficulty-confounding bias ----
    # The omitted variable is the per-attempt RESIDUAL difficulty (how much
    # harder/easier this problem is for this user than its 3PL rating says).
    # On the inner-logit / rating scale, within user, the bias on the time
    # coefficient is (omitted-variable formula):
    #   bias = -(success_slope) * (time_vs_difficulty_sens) * var(resid_diff)
    #          / var(time after removing measured difficulty)
    # success_slope and time_vs_difficulty_sens are measured here; the residual
    # difficulty sd is taken from prior calibration (bot-rating overdispersion
    # for real users ~ 0.5 rating units, project_bot_rating_accuracy_vs_users)
    # and swept. Identifying assumption: residual difficulty drives solving time
    # with the same per-rating sensitivity as measured difficulty does.
    print("\n" + "=" * 78)
    print("STRUCTURAL ESTIMATE OF THE DIFFICULTY-CONFOUNDING BIAS")
    print("=" * 78)
    ability = {r["user"]: r["theta"] for r in results}
    g_meas, logt_b, slope_b, ukeys = [], [], [], []
    for i, r in enumerate(rows):
        user = r[0]
        if user not in ability:
            continue
        ukeys.append(user)
        g_meas.append(r[3] - ability[user])   # measured difficulty = rating - ability
        logt_b.append(log_t[i])               # winsorized log10 solving seconds
        slope_b.append(r[4])
    g_meas = np.array(g_meas); logt_b = np.array(logt_b); slope_b = np.array(slope_b)
    ukeys = np.array(ukeys)
    # within-user demean both regressors
    gd = np.empty_like(g_meas); ld = np.empty_like(logt_b)
    for u in np.unique(ukeys):
        m = ukeys == u
        gd[m] = g_meas[m] - g_meas[m].mean()
        ld[m] = logt_b[m] - logt_b[m].mean()
    # sensitivity of log-time to a rating unit of measured difficulty (within user)
    time_vs_diff = float(np.sum(gd * ld) / np.sum(gd * gd))
    time_resid = ld - time_vs_diff * gd          # time variation NOT explained by difficulty
    var_time_resid = float(np.var(time_resid))
    success_slope = float(np.mean(slope_b))       # mean 3PL slope = logit per rating
    pooled_headline = pooled                       # inverse-variance pooled per-user coefficient
    print(f"  measured success-slope (logit per rating)            = {success_slope:.3f}")
    print(f"  time-vs-difficulty sensitivity (log10-sec per rating) = {time_vs_diff:+.3f}")
    print(f"  within-user var of difficulty (rating²)              = "
          f"{float(np.var(gd)):.3f}")
    print(f"  var of time after removing difficulty (log10-sec²)   = {var_time_resid:.4f}")
    print(f"  global sd of log10 solving time                      = {global_sd:.3f}")
    print(f"\n  Observed pooled time coefficient (headline)          = {pooled_headline:+.3f}"
          f"  (standardized-time units)")
    print(f"\n  Implied bias under residual-difficulty sd assumptions:")
    print(f"  {'resid-diff sd':>13}  {'bias (std units)':>16}  "
          f"{'% of headline':>13}")
    for sd_d in (0.3, 0.4, 0.5, 0.6):
        bias_logt = -success_slope * time_vs_diff * (sd_d ** 2) / var_time_resid
        bias_std = bias_logt * global_sd     # convert to the headline's standardized-time units
        pct = 100 * bias_std / pooled_headline if pooled_headline != 0 else float('nan')
        print(f"  {sd_d:>13.2f}  {bias_std:>+16.3f}  {pct:>12.0f}%")
    # Inversion: residual-difficulty sd that would explain the ENTIRE headline
    # through this channel (at the measured time-vs-difficulty sensitivity).
    sd_full = float(np.sqrt(abs(pooled_headline) * var_time_resid
                            / (success_slope * abs(time_vs_diff) * global_sd)))
    print(f"\n  Residual-difficulty sd needed to explain the ENTIRE headline through\n"
          f"  this channel = {sd_full:.2f} rating units (vs calibrated ~0.50).")
    print(f"\n  Reading: at the calibrated residual-difficulty sd≈0.5 the MEASURABLE\n"
          f"  channel accounts for ~1/4 of the observed negative — a firm LOWER bound\n"
          f"  on the bias (it omits the user×problem residual, same sign). It looks\n"
          f"  small only because the sensitivity of time to MEASURED difficulty\n"
          f"  ({time_vs_diff:+.3f}) is itself suppressed: problems are SERVED\n"
          f"  difficulty-matched (~80%-clear), so measured difficulty is compressed\n"
          f"  by design and is further attenuated by error in the ability/rating\n"
          f"  estimates. The difficulty a user actually FEELS is dominated by the\n"
          f"  residual the server can't match on — so residual difficulty plausibly\n"
          f"  drives time MORE strongly than measured difficulty, pushing the true\n"
          f"  bias well above 1/4. Explaining ALL of the headline needs residual sd\n"
          f"  ~{sd_full:.1f} (2x calibrated) OR that stronger sensitivity. Net: the\n"
          f"  data are consistent with the negative coefficient being mostly — and\n"
          f"  possibly entirely — confounding, not a causal 'time hurts' effect.\n"
          f"  Single-attempt logs cannot close the gap; a forced-minimum-time A/B can.")
    # ---- Bias-CORRECTED share who benefit ----
    # Shift every user's estimate UP by the (population, additive) bias we
    # removed, then re-ask: what share clear each confidence level for a
    # POSITIVE causal time effect? The bias is partially identified, so sweep
    # the correction from the firm measurable lower bound up to "the entire
    # average negative is bias" (= -pooled_headline). SE per user is unchanged
    # (it's the sampling SE of the estimate); sweeping the correction is exactly
    # how the bias-magnitude uncertainty is surfaced.
    bias_lb = success_slope * abs(time_vs_diff) * (0.5 ** 2) / var_time_resid * global_sd
    corrections = [
        (0.0, "uncorrected"),
        (bias_lb, "measurable lower bound (resid sd 0.5)"),
        (abs(time_vs_diff) * success_slope * (0.6 ** 2) / var_time_resid * global_sd,
         "resid sd 0.6"),
        (0.5 * abs(pooled_headline), "half the headline is bias"),
        (abs(pooled_headline), "ALL the headline is bias (zero mean causal)"),
    ]
    thr = [0.5, 0.8, 0.9, 0.95]
    print("\n" + "=" * 78)
    print("BIAS-CORRECTED: SHARE OF USERS WHO BENEFIT FROM MORE TIME, BY CONFIDENCE")
    print("=" * 78)
    print(f"  (per-user P(causal effect > 0) = Φ((estimate + correction)/se);  "
          f"n={len(beta)} users)")
    print(f"\n  {'correction':>10}  {'≥50%':>6}  {'≥80%':>6}  {'≥90%':>6}  "
          f"{'≥95%':>6}   assumption")
    for corr, label in corrections:
        ppc = ndtr((beta + corr) / se)
        cells = "  ".join(f"{100*np.mean(ppc >= t):5.1f}%" for t in thr)
        print(f"  {corr:>+10.3f}  {cells}   {label}")
    print(f"\n  Same, restricted to WELL-MEASURED users (SE ≤ 0.5, n={int(np.sum(se<=0.5))}):")
    wm = se <= 0.5
    print(f"  {'correction':>10}  {'≥50%':>6}  {'≥80%':>6}  {'≥90%':>6}  {'≥95%':>6}")
    for corr, label in corrections:
        ppc = ndtr((beta[wm] + corr) / se[wm])
        cells = "  ".join(f"{100*np.mean(ppc >= t):5.1f}%" for t in thr)
        print(f"  {corr:>+10.3f}  {cells}")
    print(f"\n  Reading: even granting the FULL headline as bias (last row — the most\n"
          f"  generous, recentering the average causal effect to zero), only a\n"
          f"  minority clear 80% confidence and almost none clear 95% — that's just\n"
          f"  the noise spread around zero, i.e. no evidence of a real benefiting\n"
          f"  subpopulation. At the defensible lower-bound correction the share is\n"
          f"  tiny. So 'who provably benefits from more time' stays near zero at any\n"
          f"  honest confidence level, whichever bias correction you accept.")

    # ---- Composition of the overlay's ACTUAL firing cell (wrong + fast) ----
    # The overlay only fires on WRONG answers below the user's median time. So
    # the relevant question is not the average time effect but: are fast FAILURES
    # on problems the user could have solved (high 3PL-predicted success →
    # careless rush, nudge valid) or on problems beyond them (low predicted
    # success → fast give-up, nudge misfires)? Compare predicted success and
    # difficulty-gap across the {fast,slow}×{correct,wrong} cells, per user.
    print("\n" + "=" * 78)
    print("COMPOSITION OF THE FIRING CELL  (overlay fires only on WRONG + fast)")
    print("=" * 78)
    ru = defaultdict(list)
    for i, r in enumerate(rows):
        if r[0] in ability:
            ru[r[0]].append(i)

    def kappa_s(c):
        c = min(max(c, 0.0), FLOOR_MAX)
        if c <= 0:
            return np.log(CALIBRATION_P / (1 - CALIBRATION_P))
        q = min(max((CALIBRATION_P - c) / (1 - c), 1e-6), 1 - 1e-6)
        return np.log(q / (1 - q))

    cells = {"fast_wrong": [], "slow_wrong": [], "fast_correct": [], "slow_correct": []}
    gaps = {k: [] for k in cells}
    for u, idxs in ru.items():
        a = ability[u]
        med = np.median([rows[i][1] for i in idxs])   # user's median solving seconds
        for i in idxs:
            _, sec, y, rating, slope, floor = rows[i]
            pred = floor + (1 - floor) * sigmoid(slope * (a - rating) + kappa_s(floor))
            key = ("fast_" if sec < med else "slow_") + ("correct" if y else "wrong")
            cells[key].append(pred)
            gaps[key].append(rating - a)            # >0 = problem above user's ability

    print(f"  {'cell':>13}  {'n':>7}  {'mean pred. success':>18}  "
          f"{'mean difficulty gap':>19}")
    for k in ("fast_wrong", "slow_wrong", "fast_correct", "slow_correct"):
        arr = np.array(cells[k]); g = np.array(gaps[k])
        print(f"  {k:>13}  {arr.size:>7d}  {arr.mean():>18.3f}  {g.mean():>+19.3f}")
    fw = np.array(cells["fast_wrong"])
    sw = np.array(cells["slow_wrong"])
    print(f"\n  Among FAST failures (the cell the nudge targets):")
    print(f"    predicted success ≥0.5 (clearly should-have-solved): "
          f"{100*np.mean(fw >= 0.5):.1f}%")
    print(f"    predicted success ≥0.7:                              "
          f"{100*np.mean(fw >= 0.7):.1f}%")
    print(f"    predicted success <0.3 (was likely to fail anyway):  "
          f"{100*np.mean(fw < 0.3):.1f}%")
    print(f"  vs SLOW failures predicted success ≥0.5: {100*np.mean(sw >= 0.5):.1f}%")
    print(f"\n  Reading: a fast failure on a high-predicted-success problem is a\n"
          f"  careless miss — the one case where 'slow down' is plausibly right. A\n"
          f"  fast failure on a low-predicted-success problem is a quick give-up on\n"
          f"  something beyond reach, where more time wouldn't have helped. The split\n"
          f"  above is the share of the nudge that is well- vs mis-targeted. Note the\n"
          f"  client ALSO gates on a positive difficulty-controlled time coefficient,\n"
          f"  which is negative for ~85% of users — so today it rarely fires at all.")

    print("\nCaveats: observational (β is association, not a causal time dose);\n"
          "reverse causation survives the 3PL control (a subjectively easy\n"
          "problem is answered fast AND correctly); single month of data\n"
          "(solving time logged only from 2026-05).")

    # ---- Optional plots ----
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        axes[0].hist(beta, bins=40, color="#4C72B0")
        axes[0].axvline(0, color="k", lw=1)
        axes[0].set_title("Per-user time effect (all fitted)")
        axes[0].set_xlabel("time-benefit coefficient")
        axes[1].hist(se, bins=40, color="#55A868")
        axes[1].set_title("Standard-error distribution")
        axes[1].set_xlabel("standard error of time-benefit")
        axes[2].hist(prob_pos, bins=np.linspace(0, 1, 21), color="#8172B3")
        axes[2].axvline(0.5, color="k", lw=1)
        axes[2].set_title("P(time-benefit > 0), all fitted")
        axes[2].set_xlabel("probability positive")
        tight = se <= 0.5
        if np.any(tight):
            axes[3].hist(prob_pos[tight], bins=np.linspace(0, 1, 21),
                         color="#C44E52")
            axes[3].axvline(0.5, color="k", lw=1)
            axes[3].set_title(f"P(>0) | SE≤0.5  (n={tight.sum()})")
            axes[3].set_xlabel("probability positive")
        fig.tight_layout()
        out_png = os.path.join(PROJECT_ROOT, "auxiliary", "time_benefit.png")
        fig.savefig(out_png, dpi=110)
        print(f"\nWrote plots → {out_png}")


if __name__ == "__main__":
    main()
