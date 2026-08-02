"""Validation bench — score a directory of rating outputs on a hash-confirmed held-out set.

THE single home of rating validation: held-out-problem logloss, held-out Δskill flatness, and
survey-ρ (base + time-adjusted). The rating tests (auxiliary.rating_dev.train_rating_staged) carry
NO inbuilt validation; they either chain this bench afterward or pass --validate to call
run_validation on the just-written CSVs (same code, no duplication).

Pointed at a ratings dir (the per-size tsumego_{ratings,slopes,floors}.csv + rating_constants.json
+ user_theta.csv a training run writes) plus a (seed, held-out fraction), it:

  1. Derives the SAME held-out problem set the test run excluded, via auxiliary.rating_dev.holdout_set, and prints
     its hash. With --expect-hash it ASSERTS the hash matches (bench and test provably agree on the
     set; mismatch => different set or a drifted attempts DB => nonzero exit).
  2. Reports, on the held-out (UNSEEN) problems — which the rated model never trained on, so their
     b/a/c come purely from the feature prior:
       - held-out-PROBLEM logloss            (feature-prior generalization)
       - held-out Δskill flatness            (per-rating-bin even-error, out of sample)
  3. Reports, on the whole rating map (held-out-independent):
       - survey-ρ (per-server Spearman, n-weighted), base and time-adjusted (fit on THIS run).

Predictions are reconstructed from the SHIPPED rating files (no model / no activation load):
  b_p = rating_p − clear_logit(c_p, λ)/a_p ;  P = c_p + (1 − c_p − λ)·σ(a_p·(θ_u − b_p))
where a_p=quality, c_p=floor, λ=fat_finger_rate, θ_u from user_theta.csv (the (user, 2-month) profile
ability). This judges the ratings EXACTLY AS PUBLISHED. Rating winsorization is part of the shipped
rating — it's what the trainer writes, what package_for_pwa puts in metadata.json, and what every
downstream consumer reads — so scoring the winsorized value is the AUTHORITATIVE measure of the
artifact, not an approximation of some un-winsorized internal b_p (which never ships). The inversion
b_p = rating − clear_logit(c,λ)/a is exact given (rating, a, c, λ), so the reconstructed 4PL curve is
exactly the one whose 80%-clear point is the published rating. Consequence: this held-out logloss can
legitimately differ from a trainer's in-process number (computed on un-winsorized params) — the
bench's is the one that reflects what ships, and it correctly charges any predictive cost/benefit of
the winsorization step itself.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.validate_ratings --ratings-dir /tmp/rv2_ho --seed 0 --holdout-problem-frac 0.1
  ./.venv/bin/python3 -m auxiliary.rating_dev.validate_ratings --ratings-dir /tmp/rv2_ho --seed 0 --holdout-problem-frac 0.1 --expect-hash e83ecdbf5c2e3dae
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.rating_dev.holdout_set import held_out_keys  # noqa: E402
from utils.rating_features import load_first_time_attempts  # noqa: E402
from auxiliary.rating_dev.model_comparison import build_profiles  # noqa: E402
from utils.fit_rank_curves import CALIBRATION_P  # noqa: E402
from utils.ratings_io import load_size_value_maps  # noqa: E402

EPS = 1e-6
SIZES = [9, 13]


def clear_logit_vec(c, lapse=0.0):
    """Per-problem 80%-clear logit anchor under the 4PL: at the rating P(clear)=CALIBRATION_P, so
    σ(a(θ−b))=(P*−c)/(1−c−λ) and rating = b + logit((P*−c)/(1−c−λ))/a. Vector over problems. Must
    match the trainer's clear_logit_vec so the inversion b = rating − clear_logit(c,λ)/a is exact."""
    c = np.asarray(c, dtype=np.float64)
    q = np.clip((CALIBRATION_P - c) / (1.0 - c - lapse), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


# ── survey-ρ metrics (the single definition; the trainer no longer carries a copy) ──────────────
# Time-adjust yardstick knobs + the weighted-Spearman helper, inherited verbatim from the retired
# fit_population_constants time fit (removed at 3c1f57521b when the production time fit moved into
# map_rating_to_rank's joint objective). They live here now as EVAL-side constants: this bench
# still scores a fresh run's own cross-fitted time-adjusted rho, independent of the shipped dock.
TA_MIN_TIMED = 20          # timed novel attempts needed for a user's time feature
TA_MIN_PAIRS = 50          # (user, server) survey pairs needed to fit alpha
TA_MIN_SERVER_USERS = 5    # per-server minimum for a Spearman term
TA_ALPHA_GRID = np.linspace(-1.5, 0.5, 81)   # rating units per log10-minute
TA_CV_FOLDS = 5
TA_CV_SEEDS = 10
TA_AFK_CAP_S = 180.0       # drop AFK solving times from the median-time feature


def _weighted_spearman(pairs, alpha):
    """pairs: [(user, server, rating, rank_value, centered_feature)]. Per-server
    Spearman(rating + alpha·feature, declared rank), user-count-weighted mean."""
    from collections import defaultdict
    from scipy.stats import spearmanr
    by_server = defaultdict(list)
    for _, server, rating, sc, f in pairs:
        by_server[server].append((rating + alpha * f, sc))
    acc, w = 0.0, 0.0
    for rows_ in by_server.values():
        if len(rows_) < TA_MIN_SERVER_USERS:
            continue
        r = np.array([x for x, _ in rows_])
        s = np.array([y for _, y in rows_])
        if np.ptp(r) == 0 or np.ptp(s) == 0:
            continue
        rho = spearmanr(r, s).statistic
        if np.isfinite(rho):
            acc += rho * len(rows_)
            w += len(rows_)
    return acc / w if w else np.nan


def _user_log10_median_solve_min(sql_path, prefixes, afk_cap_ms, min_timed):
    """{user_prefix8: log10(median solving minutes)} over each user's NON-REPEAT, AFK-capped timed
    attempts (the same feature the client and fit_population_constants compute). Built from the
    canonical D1 .sql dump (parsing ~289k rows costs a few seconds — fine for a once-per-run eval,
    and correct even when the sibling .sqlite is stale). Users below `min_timed` timed attempts are
    omitted (no reliable time feature)."""
    import sqlite3
    from collections import defaultdict
    con = sqlite3.connect(":memory:")
    with open(sql_path) as f:
        con.executescript(f.read())
    rows = con.execute(
        "SELECT user_id, time_used FROM tsumego_reports "
        "WHERE is_repeat=0 AND time_used>0 AND time_used<=?", (afk_cap_ms,)).fetchall()
    con.close()
    buckets = defaultdict(list)
    for uid, t in rows:
        p = str(uid)[:8]
        if p in prefixes:
            buckets[p].append(t)
    return {p: float(np.log10(np.median(ts) / 60000.0))
            for p, ts in buckets.items() if len(ts) >= min_timed}


def _time_adjusted_rho(surveys, user_rating, sql_path):
    """Time-adjusted survey rho with EVERY coefficient fit on the CURRENT run — never the stale
    ratings/population_constants.json `time_adjust` block (its alpha/g0/g1 were fit on a prior
    run's ratings, which a fresh train invalidates). Each survey user's median solving time
    (log10 min) is residualized on THIS run's 80%-clear rating R (t = g0 + g1·R), so only solve
    time UNEXPLAINED by skill drives the adjustment — and the residualization needs no rating→rank
    curve M (also stale for a fresh run). A single alpha (rating units per log10-minute, clamped
    ≤0 so slow-for-skill is never rated higher) is picked by a per-server survey-Spearman sweep,
    cross-fitted (alpha chosen on train folds, scored on held-out users) so the reported number is
    not optimistically self-tuned. Returns a dict (rho_base/rho/rho_insample/alpha/g1/n_users/
    n_pairs) or {note} when too sparse to fit."""
    tfeat = _user_log10_median_solve_min(sql_path, set(user_rating), TA_AFK_CAP_S * 1000.0,
                                         TA_MIN_TIMED)
    common = [p for p in user_rating if p in tfeat]
    if len(common) < TA_MIN_PAIRS:
        return {"note": f"only {len(common)} survey users with rating+time (<{TA_MIN_PAIRS})"}
    # Residualize the time feature on THIS run's rating (no stale M): t = g0 + g1·R.
    R = np.array([user_rating[p] for p in common])
    T = np.array([tfeat[p] for p in common])
    g1, g0 = (float(x) for x in np.polyfit(R, T, 1))
    resid = {p: tfeat[p] - (g0 + g1 * user_rating[p]) for p in common}
    pairs = []
    for _, row in surveys.iterrows():
        pref = str(row["user_id"])[:8]
        g = row["rank_value"]
        if pref in resid and g is not None and not (isinstance(g, float) and np.isnan(g)):
            pairs.append((pref, row["server"], user_rating[pref], float(g), resid[pref]))
    if len(pairs) < TA_MIN_PAIRS:
        return {"note": f"only {len(pairs)} timed survey pairs (<{TA_MIN_PAIRS})"}
    base = _weighted_spearman(pairs, 0.0)
    sweep = [_weighted_spearman(pairs, a) for a in TA_ALPHA_GRID]
    a_best = float(TA_ALPHA_GRID[int(np.nanargmax(sweep))])
    users_list = sorted({p[0] for p in pairs})
    cv = []
    for seed in range(TA_CV_SEEDS):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(users_list))
        fold = {users_list[j]: i % TA_CV_FOLDS for i, j in enumerate(order)}
        a_fold = []
        for kf in range(TA_CV_FOLDS):
            train = [p for p in pairs if fold[p[0]] != kf]
            sw = [_weighted_spearman(train, a) for a in TA_ALPHA_GRID]
            a_fold.append(float(TA_ALPHA_GRID[int(np.nanargmax(sw))]))
        adj = [(u, srv, rt + a_fold[fold[u]] * f, sc, 0.0) for u, srv, rt, sc, f in pairs]
        cv.append(_weighted_spearman(adj, 0.0))
    return {"rho_base": base, "rho": float(np.mean(cv)), "rho_insample": float(max(sweep)),
            "alpha": min(a_best, 0.0), "g1": g1, "n_users": len(users_list), "n_pairs": len(pairs)}


def survey_weighted_rho(ratemap, slopemap, min_attempts=20, min_server=4, time_adjust=True):
    """Composite survey objective (calibration to real ranks). For each survey user, fit their MLE
    80%-clear rating (fit_user_rating over their attempts with these per-problem ratings/slopes); then,
    PER SERVER, the Spearman against the declared rank, and the n-weighted average over servers.

    Per-server Spearman is offset- AND scale-invariant, so it is robust to server-population imbalance
    (EGF skews strong, OGS skews weak) — unlike mean-centered pooling, which only removes a server's
    offset and still mixes their differing within-server spread. The surveys are NEVER used in
    training; they enter only here, as an external validation signal.

    When `time_adjust` (default), ALSO computes a TIME-ADJUSTED survey rho via _time_adjusted_rho —
    each user's median solve time residualized on THIS run's rating and a cross-fitted alpha, all
    coefficients fit on the current run (the shipped population_constants.json alpha is stale for a
    fresh train). Returns (weighted_rho, {server: (n, rho)}, time_adjust_dict | None)."""
    from collections import defaultdict
    from utils.rank_surveys import (
        load_rank_surveys, latest_per_user_server, load_attempts_for_users, SQL_PATH_DEFAULT)
    from utils.fit_rank_curves import fit_user_rating
    surveys = latest_per_user_server(load_rank_surveys(SQL_PATH_DEFAULT))
    # first_time_only: mirror the client / production anchor (first-time attempts);
    # repeats' spaced-repetition success would inflate the 80%-clear rating this rho uses.
    attempts = load_attempts_for_users(SQL_PATH_DEFAULT, set(surveys["user_id"].unique()),
                                       first_time_only=True)
    user_rating = {}
    for uid in surveys["user_id"].unique():
        pooled = []
        for s in SIZES:
            pooled.extend(attempts.get((uid[:8], s), []))
        if len(pooled) < min_attempts:
            continue
        R, nj = fit_user_rating(pooled, ratemap, min_n=min_attempts, slopes=slopemap)
        if R is not None and nj >= min_attempts:
            user_rating[uid[:8]] = R
    serv = defaultdict(list)
    for _, row in surveys.iterrows():
        g = row["rank_value"]; pref = str(row["user_id"])[:8]
        if g is None or (isinstance(g, float) and np.isnan(g)) or pref not in user_rating:
            continue
        serv[row["server"]].append((user_rating[pref], float(g)))

    def spear(pairs):
        x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
        return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
    per = {s: (len(p), spear(p)) for s, p in serv.items() if len(p) >= min_server}
    if not per:
        return float("nan"), {}, None
    wn = sum(n for n, _ in per.values())
    wrho = sum(n * r for n, r in per.values()) / wn
    ta = _time_adjusted_rho(surveys, user_rating, SQL_PATH_DEFAULT) if time_adjust else None
    return wrho, per, ta


def load_ratings(ratings_dir):
    """{tid: rating}, {tid: slope}, {tid: floor}, λ, {profile: θ}. tids are globally unique (the
    symmetry-canonical hash differs across board sizes), so a flat per-tid dict is unambiguous."""
    rat, slp, flr = {}, {}, {}
    for s in SIZES:
        r, q, fl = load_size_value_maps(ratings_dir, s)
        rat.update(r)
        slp.update(q)
        flr.update(fl)
    lam = float(json.load(open(os.path.join(ratings_dir, "rating_constants.json")))["fat_finger_rate"])
    ut = pd.read_csv(os.path.join(ratings_dir, "user_theta.csv"))
    theta = dict(zip(ut["profile"].astype(str), ut["theta"].astype(float)))
    return rat, slp, flr, lam, theta


def heldout_metrics(rat, slp, flr, lam, theta, ho_tids, nbins=10):
    """Dict of held-out metrics over first-time attempts on held-out problems, restricted to scorable
    problems (in `rat`) and trained profiles (in `theta`) — exactly the held-out rows the trainer's
    prof_df eval saw. Predictions are reconstructed from the shipped CSVs (see the module docstring on
    winsorization). Returns None if no scorable held-out attempts. Keys:
      logloss          model 4PL held-out-problem logloss (the backbone-transfer metric)
      constant_logloss the BEST CONSTANT predictor's logloss = H(p), p = held-out solve rate. p is the
                       optimal constant on this set, so this is the lowest logloss reachable WITHOUT
                       per-problem information; the gap below is purely the model's discrimination.
                       NB p is itself shaped by difficulty-matched serving (served ≈80%-clear via the
                       PRIOR ratings), so the served stream is range-restricted and the achievable gap
                       is compressed — read `improvement` as a within-served-band, version-over-version
                       discrimination figure, not an absolute accuracy. See module docstring.
      base_rate        p (held-out solve rate)
      improvement      constant_logloss − logloss (≥0 ⇒ model beats the best constant; in nats)
      flatness         (std, worst, spread) of per-rating-bin even-error, or None
      n_rows/n_problems counts
    """
    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)]
    prof = build_profiles(df)                       # (profile, problem, success), exploded into windows
    prof = prof[prof["problem"].isin(rat) & prof["profile"].astype(str).isin(theta)]
    sub = prof[prof["problem"].isin(ho_tids)]
    if len(sub) == 0:
        return None
    tids = sub["problem"].to_numpy()
    a = np.array([slp[t] for t in tids])
    c = np.array([flr[t] for t in tids])
    r = np.array([rat[t] for t in tids])
    th = sub["profile"].astype(str).map(theta).to_numpy(float)
    b = r - clear_logit_vec(c, lam) / a             # invert rating = b + clear_logit(c,λ)/a (exact)
    P = np.clip(c + (1 - c - lam) / (1 + np.exp(-(a * (th - b)))), EPS, 1 - EPS)
    yy = sub["success"].to_numpy(float)
    ho_ll = float(-(yy * np.log(P) + (1 - yy) * np.log(1 - P)).mean())
    # constant-rate reference: predict p (the held-out solve rate) for EVERY row. The constant that
    # minimizes logloss is the label mean, and its logloss equals the Bernoulli entropy H(p), so this
    # is the no-per-problem-information floor; model `improvement` = H(p) − ho_ll.
    base_rate = float(np.clip(yy.mean(), EPS, 1 - EPS))
    const_ll = float(-(base_rate * np.log(base_rate) + (1 - base_rate) * np.log(1 - base_rate)))
    # held-out Δskill flatness: per equal-count rating bin, bin LL − bin base-rate entropy.
    edges = np.quantile(r, np.linspace(0, 1, nbins + 1)); edges[-1] += 1e-6
    bk = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, nbins - 1)
    d = []
    for i in range(nbins):
        sm = bk == i
        if sm.sum() < 50:
            continue
        base = min(max(float(yy[sm].mean()), EPS), 1 - EPS)
        bll = -(base * np.log(base) + (1 - base) * np.log(1 - base))
        d.append(float(-(yy[sm] * np.log(P[sm]) + (1 - yy[sm]) * np.log(1 - P[sm])).mean()) - bll)
    d = np.array(d)
    flat = (float(d.std()), float(d.max()), float(d.max() - d.min())) if len(d) else None
    return {"logloss": ho_ll, "constant_logloss": const_ll, "base_rate": base_rate,
            "improvement": const_ll - ho_ll, "flatness": flat,
            "n_rows": len(sub), "n_problems": int(np.unique(tids).size)}


def run_validation(ratings_dir, seed, frac, expect_hash=None, no_survey=False):
    """Derive the held-out set (seed, frac), confirm/print its hash, and report held-out logloss +
    Δskill flatness (frac>0) and survey-ρ. Prints results and returns a dict. Shared by the CLI and
    by auxiliary.rating_dev.train_rating_staged --validate (one definition, no duplication)."""
    ho_keys, ho_hash = held_out_keys(seed, frac)
    print(f"held-out set: {len(ho_keys)} problems | seed {seed} | frac {frac:g} | hash {ho_hash}")
    if expect_hash and expect_hash != ho_hash:
        sys.exit(f"HASH MISMATCH: expected {expect_hash}, derived {ho_hash} — "
                 "different held-out set (or a drifted attempts DB). Aborting.")

    rat, slp, flr, lam, theta = load_ratings(ratings_dir)
    print(f"ratings: {len(rat)} problems | λ {lam:.5f} | {len(theta)} user profiles")
    out = {"hash": ho_hash, "n_holdout": len(ho_keys)}

    print(f"\n=== validation | {ratings_dir} | seed {seed} | holdout {frac:g} ===")
    if frac > 0:
        ho_tids = {t for _, t in ho_keys}
        hm = heldout_metrics(rat, slp, flr, lam, theta, ho_tids)
        if hm is None:
            out.update(holdout_logloss=None, flatness=None, n_rows=0, n_problems=0)
            print("  held-out: no scorable held-out attempts found (check ratings dir / seed / frac)")
        else:
            out.update(holdout_logloss=hm["logloss"], holdout_constant_logloss=hm["constant_logloss"],
                       holdout_base_rate=hm["base_rate"], holdout_improvement=hm["improvement"],
                       flatness=hm["flatness"], n_rows=hm["n_rows"], n_problems=hm["n_problems"])
            print(f"  held-out-PROBLEM logloss : {hm['logloss']:.4f}  ({hm['n_rows']} attempt-rows on "
                  f"{hm['n_problems']} scorable held-out problems)")
            print(f"  constant-rate baseline   : {hm['constant_logloss']:.4f}  (predict the {hm['base_rate']:.3f} "
                  f"held-out solve rate; model improvement {hm['improvement']:+.4f} nats)")
            if hm["flatness"] is not None:
                fl = hm["flatness"]
                print(f"  held-out Δskill flatness : std {fl[0]:.4f} worst {fl[1]:+.4f} "
                      f"spread {fl[2]:.4f}  (rating bins; lower = flatter held-out error)")

    if not no_survey:
        # weighted (non-time-adjusted) survey rho is still computed + returned in `out` (and the
        # per-server breakdown drives the time-adjustment), but no longer PRINTED — the time-adjusted
        # rho is the metric that matters; the raw weighted number was misleading noise next to it.
        wrho, per, ta = survey_weighted_rho(rat, slp)
        out.update(survey_rho=wrho, survey_per_server=per, time_adjust=ta)
        if ta is not None and "rho" in ta:
            print(f"  survey rho (time-adjusted): {ta['rho']:.4f} cross-fitted vs base {ta['rho_base']:.4f} "
                  f"(Δ {ta['rho'] - ta['rho_base']:+.4f}; in-sample {ta['rho_insample']:.4f}) | "
                  f"alpha {ta['alpha']:+.2f}/log10-min | {ta['n_users']} timed users, {ta['n_pairs']} pairs")
        elif ta is not None:
            print(f"  survey rho (time-adjusted): skipped ({ta['note']})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings-dir", required=True,
                    help="Dir of rating outputs (per-size CSVs + rating_constants.json + user_theta.csv).")
    ap.add_argument("--seed", type=int, required=True, help="Held-out seed (must match the test run).")
    ap.add_argument("--holdout-problem-frac", type=float, required=True,
                    help="Held-out fraction (must match the test run). 0 => survey-ρ only.")
    ap.add_argument("--expect-hash", default=None,
                    help="Assert the derived held-out-set hash equals this (the test run's printed hash).")
    ap.add_argument("--no-survey", action="store_true", help="Skip the survey-ρ metrics.")
    args = ap.parse_args()
    run_validation(args.ratings_dir, args.seed, args.holdout_problem_frac,
                   expect_hash=args.expect_hash, no_survey=args.no_survey)


if __name__ == "__main__":
    main()
