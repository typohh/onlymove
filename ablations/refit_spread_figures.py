#!/usr/bin/env python3
"""How far each problem's true success rate departs from the fitted curve
(methodology §1.5), read from the `problem_spread` refit panel.

THE QUESTION. The client serves at the calibration probability and the attempt
log clears at about that rate — which says nothing about whether each served
problem is truly at it, or whether some are truly far harder and some trivial and
the two average out. A single outcome cannot tell those apart: a Bernoulli's
variance is fixed by its mean, so the pooled outcomes look the same under every
mixture with the same mean. What separates them is outcomes that AGREE across
different users on one problem. If problem j's true logit sits u_j off the fitted
one, with u_j of spread tau over problems, two users' residuals on it covary by
about w_1*w_2*tau^2, w being dP/dlogit at each attempt; with no departure they
are independent. So

    tau^2 = sum over problems, pairs i<k of r_i*r_k  /  sum of the same w_i*w_k

a moment estimator over within-problem cross-user residual products. It is
unbiased at tau = 0 by construction — checked here on every arm by rescoring
outcomes simulated from the arm's own predictions — and needs no distributional
assumption for the VARIANCE; the implied distribution of true success rates below
does assume the departures are normal on the logit.

WHY A PANEL AND NOT THE ATTEMPT LOG AFTER A FIT. Rows attempted after a shipped
fit are held out of it, but their users' abilities are then either stale or
missing, and re-fitting them on the scored rows spends those rows twice. The
panel's arms hold rows out BEFORE the fit, so each user's ability comes from
their other attempts, and the draw is chosen for the question: only problems with
several training rows are eligible, because a problem with one held-out attempt
carries no pair. Two disjoint sets per arm (`pipeline.train_mirt`, stop unit
`spread`):

  WHOLE   every row of the problem held out — its parameters come from the
          features alone, which is the position of every never-attempted problem.
  HALVED  half its rows held out, half trained on — what the fit makes of a
          problem it has seen.

The two answer different questions and their difference is the claim §1.5 turns
on: whether a problem's own attempts move it. Both sets are drawn among OFTEN-
attempted problems, which is the population a player is served most — not the
corpus at large, half of which nobody has attempted.

A MAGNITUDE, SO THE DRAW MOVES. Each seed draws its own holdout and fits its own
model; the figures are means over seeds with the per-seed range beside them.
Nothing is subtracted across seeds except within one arm, where the two sets
share the fit.

  spread.seeds                       arms behind everything below.
  spread.min_attempts                training rows a problem needs to be eligible.
  spread.problems_whole              held-out problems per arm, per set (mean).
  spread.problems_halved
  spread.tau_whole                   tau on the logit, mean over seeds; the
  spread.tau_whole_lo / _hi          lowest lower and highest upper end of the
  spread.tau_halved                  per-seed 95% problem bootstrap.
  spread.tau_halved_lo / _hi
  spread.halving_gain                tau_whole - tau_halved, paired within each
                                     arm and averaged: what seeing half a
                                     problem's attempts removes.
  spread.halving_gain_every_seed     1 when that gain is positive at every seed.
  spread.null_bias_sd                the estimator on outcomes simulated with no
                                     departure: its mean in units of its own
                                     spread — near zero or the estimator is off.
  spread.predicted_p05 / _p95        the model's own predicted success rates on
                                     the WHOLE set's rows, 5th and 95th percentile.
  spread.true_p05 / _p95             the implied TRUE success rates on those rows
                                     (normal departures of spread tau_whole).
  spread.true_below60_pct            share of those rows truly below 60%.
  spread.info_loss_pct               Fisher information per attempt lost against
                                     every attempt sitting at its prediction.
  spread.recoverable_pct             for an attempt drawn from the log, the share
                                     of its problem's departure variance a
                                     shrunken per-problem estimate built from the
                                     OTHER attempts on that problem would recover
                                     — tau^2*I/(1 + tau^2*I), I being those
                                     attempts' information about the offset. What
                                     per-problem parameters would buy at the
                                     density the log has.
  spread.rank_err_median / _p90      the same departure in RANKS: how far a
                                     problem's true calibration point sits from
                                     the fitted one, median and 90th percentile
                                     of its absolute value over the whole set's
                                     attempts. A logit offset u on a problem of
                                     slope a moves its difficulty by u/a, worth
                                     u*M'(theta)/a ranks to the player attempting
                                     it, M' the shipped curve's local slope at
                                     that player's time-adjusted rating.
  spread.tau_<band>                  the departure spread read INSIDE a player
                                     band — from cross-user pairs whose players
                                     both sit in it — since a problem's offset
                                     need not be the same for weak and strong
                                     players. Each band's rank figures use its
                                     own spread, never the pooled one.
  spread.ranks_per_logit_<band>      median M'(theta)/a over the band's attempts.
  spread.rank_err_median_<band>      per-band median and 90th percentile.
  spread.rank_err_p90_<band>

    ./.venv/bin/python3 -m ablations.refit_spread_figures
"""

import json
import os
import sqlite3
import statistics
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require, panel_regime  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402

TIER = "refit"

PANEL = "problem_spread"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")
SETS = {0: "whole", 1: "halved"}
BOOTSTRAP = 2000
NULL_REPS = 40
IMPLIED_DRAWS = 20
ROWS_KEYS = ("set", "tid", "user_id", "y", "z", "floor", "lapse", "p", "w")


def load_panel():
    require(os.path.exists(MANIFEST), f"no {MANIFEST}")
    manifest = json.load(open(MANIFEST))
    arms = {}
    for arm, cfg in manifest["arms"].items():
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the problem_spread panel has not finished. Run it "
                f"with `python -m ablations.run_panel {PANEL}` (or "
                f"`auxiliary/vast_orchestration.py ablate {PANEL}`).")
        rec = json.load(open(path))
        require(rec["config"] == cfg,
                f"{arm} ran {rec['config']!r}, the manifest declares {cfg!r} — "
                f"the record is not this arm")
        require(cfg.get("stop_unit") == "spread",
                f"{arm} does not hold out the spread sets")
        arms[arm] = rec
    budget, schedule = panel_regime(arms)
    for arm, r in arms.items():
        require(r["metrics"]["trace"]
                and r["metrics"]["trace"][-1]["epoch"] == budget,
                f"{arm}'s trace does not reach the {budget}-epoch budget")
    base = next(iter(arms.values()))["pins"]
    for arm, r in arms.items():
        for pin in ("db_max_id", "labels_sha", "proxies_sha", "blocks",
                    "pca_dim", "k_axes"):
            require(r["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} differs — the arms are not one panel")
    splits = [r["metrics"]["split_seed"] for r in arms.values()]
    require(len(set(splits)) == len(splits),
            f"the arms share a held-out draw ({splits}) — this panel estimates "
            f"a magnitude and wants the draw moving")
    require(len(arms) >= 2, "one arm is one draw — a magnitude needs several")

    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute("SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"measured at attempt-log frontier {base['db_max_id']}, the log now "
            f"ends at {now} — re-run the panel")
    return arms, budget, schedule, base["gpu"]


def load_rows(arm, rec):
    path = os.path.join(PANEL_DIR, arm, "ratings", "spread_rows.npz")
    require(os.path.exists(path), f"no {path}")
    d = np.load(path)
    require(set(d.files) == set(ROWS_KEYS),
            f"{path} carries {sorted(d.files)}, expected {sorted(ROWS_KEYS)}")
    rows = {k: d[k] for k in ROWS_KEYS}
    n = len(rows["y"])
    require(all(len(rows[k]) == n for k in ROWS_KEYS if k != "lapse"),
            f"{path}: columns of unequal length")
    require(n == rec["metrics"]["stop_rows"],
            f"{path} holds {n} rows, the arm held aside "
            f"{rec['metrics']['stop_rows']}")
    require(set(np.unique(rows["set"]).tolist()) == set(SETS),
            f"{path} carries sets {np.unique(rows['set']).tolist()}")
    # A problem in both sets would be one held out whole and halved at once.
    whole = set(rows["tid"][rows["set"] == 0].tolist())
    halved = set(rows["tid"][rows["set"] == 1].tolist())
    require(not whole & halved, f"{path}: a problem sits in both sets")
    # Each row is one user's first attempt at one problem; a repeated pair would
    # put one person's two outcomes into a cross-user product.
    pair = np.char.add(np.char.add(rows["tid"], "|"), rows["user_id"])
    require(len(np.unique(pair)) == n, f"{path}: a (user, problem) pair repeats")
    return rows


def pair_sums(prob, r, w):
    """Per problem, the sum over its unordered row pairs of r_i*r_k and of
    w_i*w_k — ((sum)^2 - sum of squares) / 2 each. Problems with one row
    contribute zero to both and are dropped."""
    u, inv = np.unique(prob, return_inverse=True)
    R = np.bincount(inv, r); R2 = np.bincount(inv, r * r)
    W = np.bincount(inv, w); W2 = np.bincount(inv, w * w)
    cnt = np.bincount(inv)
    keep = cnt >= 2
    return (R[keep] ** 2 - R2[keep]) / 2.0, (W[keep] ** 2 - W2[keep]) / 2.0


def tau2(num, den):
    return float(num.sum() / den.sum())


def measure_set(rows, s, rng):
    sel = rows["set"] == s
    prob, y, p, w = rows["tid"][sel], rows["y"][sel], rows["p"][sel], rows["w"][sel]
    num, den = pair_sums(prob, y - p, w)
    require(len(num) > 500, f"set {SETS[s]}: only {len(num)} problems carry a pair")
    t2 = tau2(num, den)
    idx = rng.integers(0, len(num), (BOOTSTRAP, len(num)))
    boots = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    nulls = []
    for _ in range(NULL_REPS):
        ys = (rng.random(len(p)) < p).astype(float)
        nulls.append(tau2(*pair_sums(prob, ys - p, w)))
    return {"problems": len(num), "rows": int(sel.sum()), "tau2": t2,
            "tau": float(np.sqrt(max(t2, 0.0))),
            "tau_lo": float(np.sqrt(max(lo, 0.0))),
            "tau_hi": float(np.sqrt(max(hi, 0.0))),
            "null_mean": float(np.mean(nulls)), "null_sd": float(np.std(nulls)),
            "predicted_p": p}


def implied(rows, s, tau, rng):
    """True success rates on the set's rows if each problem's logit sits a
    normal draw of spread `tau` off the fitted one, one draw per problem."""
    sel = rows["set"] == s
    prob, z, c = rows["tid"][sel], rows["z"][sel], rows["floor"][sel]
    lam = float(rows["lapse"])
    top = 1.0 - c - lam
    u, inv = np.unique(prob, return_inverse=True)
    true_p, info_true = [], []
    for _ in range(IMPLIED_DRAWS):
        off = rng.normal(0.0, tau, len(u))[inv]
        sg = 1.0 / (1.0 + np.exp(-(z + off)))
        pt = c + top * sg
        true_p.append(pt)
        info_true.append((top * sg * (1 - sg)) ** 2 / (pt * (1 - pt)))
    true_p = np.concatenate(true_p)
    sg0 = 1.0 / (1.0 + np.exp(-z))
    p0 = c + top * sg0
    info_nom = float(np.mean((top * sg0 * (1 - sg0)) ** 2 / (p0 * (1 - p0))))
    return {"p05": float(np.percentile(true_p, 5)),
            "p95": float(np.percentile(true_p, 95)),
            "below60": float(np.mean(true_p < 0.6)),
            "info_loss": 1.0 - float(np.mean(np.concatenate(info_true))) / info_nom}


def recoverable(tau, info_per_attempt):
    """Attempt-weighted share of the departure variance a normal-prior
    per-problem estimate recovers from the other attempts on the problem: the
    posterior variance of an offset with prior tau^2 after information I is
    tau^2/(1 + tau^2*I), so the recovered share is tau^2*I/(1 + tau^2*I). The
    count is taken over the same rows the fit reads (first-time attempts by
    users it keeps), at the frontier the panel was pinned to."""
    from utils.rating.features import load_first_time_attempts
    df = load_first_time_attempts()
    g = df.groupby("user_id")["success"].agg(["sum", "count"])
    keep = g.index[(g["sum"] >= 1) & (g["count"] - g["sum"] >= 1)]
    n = df[df["user_id"].isin(keep)].groupby("problem").size().to_numpy()
    others = np.repeat(n - 1, n)
    t2i = tau * tau * info_per_attempt * others
    return float(np.mean(t2i / (1.0 + t2i)))


def shipped_rank_scale():
    """Per user, the internal rank and the ranks one unit of rating is worth there;
    per problem, its slope — all from the SHIPPED fit and its M curve. Ranks per
    logit, M'(theta)/slope, is the same in any gauge only when the curve, the
    abilities and the slopes come from one fit, and the panel's arms ship no curve
    of their own, so the conversion is the shipped fit's while the departure it
    converts is the panel's. The rank is read at the time-adjusted rating, as every
    rank the mapping produces is."""
    from utils.rating.features import RATINGS_DIR
    from utils.rating.ratings_io import (load_M, load_size_value_maps,
                                         load_time_adjust, load_value_map,
                                         interp_extrap_vec)
    from pipeline.map_rating_to_rank import load_user_median_solve_minutes
    mr, mi = load_M(RATINGS_DIR)
    ta = load_time_adjust(RATINGS_DIR)
    require(ta is not None, f"no time_adjust in {RATINGS_DIR}/population_constants.json")
    alpha, center = ta
    theta = load_value_map(os.path.join(RATINGS_DIR, "user_theta.csv"), "theta")
    require(theta, f"no fitted abilities in {RATINGS_DIR}/user_theta.csv")
    med = load_user_median_solve_minutes(DB_SQLITE)
    users = sorted(theta)
    t = np.array([theta[u] + (alpha * (np.log10(med[u[:8]]) - center)
                              if u[:8] in med else 0.0) for u in users])
    # Central difference through the shipped interpolator itself, so the slope is
    # the curve's own segment slope, extrapolated past the end knots as it is.
    h = 1e-6
    per_rating = (interp_extrap_vec(t + h, mr, mi)
                  - interp_extrap_vec(t - h, mr, mi)) / (2 * h)
    rank = interp_extrap_vec(t, mr, mi)
    slope = {}
    for size in ("9", "13"):
        slope.update(load_size_value_maps(RATINGS_DIR, size)[1])
    return dict(zip(users, rank)), dict(zip(users, per_rating)), slope


def abs_quantile(scale, q):
    """The q-quantile of |u*R| over rows, u ~ N(0, 1) per row and R = `scale`
    (tau times ranks per logit): the mixture's CDF is the row mean of
    2*Phi(x/scale) - 1, solved by bisection — exact, no sampling."""
    from scipy.special import erf
    lo, hi = 0.0, float(scale.max()) * 10.0
    for _ in range(100):
        x = (lo + hi) / 2
        if np.mean(erf(x / (scale * np.sqrt(2.0)))) < q:
            lo = x
        else:
            hi = x
    return (lo + hi) / 2


def rank_error(per, rng):
    """How many ranks a problem's true calibration point sits from the fitted one,
    per player band, on the WHOLE set's rows. The departure's spread is estimated
    INSIDE each band — from cross-user pairs whose players both sit in it — because
    a problem's offset need not be the same for a weak and a strong player, and a
    pooled tau applied to every band would assume it is. Each band's moment sums
    are pooled over the fits (their problem draws are disjoint samples of one
    population), so a thin band reads off every pair the panel holds. The row's
    error is then that band's departure times its own ranks per logit."""
    rank, per_rating, slope = shipped_rank_scale()
    rows = []
    for a in sorted(per):
        r = per[a]["rows"]
        sel = r["set"] == 0
        users, tids = r["user_id"][sel], r["tid"][sel]
        missing = [t for t in set(tids.tolist()) if t not in slope]
        require(not missing, f"{a}: {len(missing)} whole-set problems have no "
                f"shipped slope, e.g. {missing[:3]}")
        known = np.array([u in rank for u in users])
        rows.append({
            "arm": a, "dropped": int((~known).sum()), "rows": int(sel.sum()),
            "tid": tids[known], "r": (r["y"] - r["p"])[sel][known],
            "w": r["w"][sel][known],
            "rank": np.array([rank[u] for u in users[known]]),
            "rpl": np.array([per_rating[u] / slope[t]
                             for u, t in zip(users[known], tids[known])]),
        })
    dropped = sum(x["dropped"] for x in rows) / sum(x["rows"] for x in rows)
    bands = []
    scale_all = []
    for key, label, lo, hi in RANK_BANDS:
        nums, dens, scales = [], [], []
        for x in rows:
            b = (x["rank"] > lo) & (x["rank"] <= hi)
            num, den = pair_sums(x["tid"][b], x["r"][b], x["w"][b])
            nums.append(num); dens.append(den); scales.append(x["rpl"][b])
        num, den = np.concatenate(nums), np.concatenate(dens)
        rpl = np.concatenate(scales)
        require(len(num) > 100, f"band {label}: only {len(num)} problems carry a "
                f"within-band pair — coarsen the bands")
        t2 = tau2(num, den)
        require(t2 > 0, f"band {label}: departure variance {t2:+.3f} is not "
                f"positive — too few pairs to read a spread from")
        idx = rng.integers(0, len(num), (BOOTSTRAP, len(num)))
        boots = num[idx].sum(1) / den[idx].sum(1)
        tlo, thi = np.percentile(boots, [2.5, 97.5])
        tau = float(np.sqrt(t2))
        scale = tau * rpl
        scale_all.append(scale)
        bands.append({"key": key, "label": label, "rows": len(rpl),
                      "problems": len(num), "tau": tau,
                      "tau_lo": float(np.sqrt(max(tlo, 0.0))),
                      "tau_hi": float(np.sqrt(max(thi, 0.0))),
                      "rpl": float(np.median(rpl)),
                      "median": abs_quantile(scale, 0.5),
                      "p90": abs_quantile(scale, 0.9)})
    scale_all = np.concatenate(scale_all)
    return {"bands": bands, "dropped": dropped,
            "median": abs_quantile(scale_all, 0.5),
            "p90": abs_quantile(scale_all, 0.9)}


# Player bands on the internal rank scale (0 = the 1k/1d boundary, one unit per
# rank, integers are band edges), lower edge exclusive. The strongest bands are
# merged into one because few problems carry two dan players' outcomes.
RANK_BANDS = (
    ("k21", "21k and weaker", -np.inf, -20.0),
    ("k20_16", "20k–16k", -20.0, -15.0),
    ("k15_11", "15k–11k", -15.0, -10.0),
    ("k10_6", "10k–6k", -10.0, -5.0),
    ("k5_1", "5k–1k", -5.0, 0.0),
    ("dan", "dan", 0.0, np.inf),
)


def measure():
    from pipeline.train_mirt import SPREAD_MIN_ATTEMPTS
    arms, budget, schedule, gpu = load_panel()
    per = {}
    for arm, rec in sorted(arms.items()):
        rows = load_rows(arm, rec)
        rng = np.random.default_rng(0)
        per[arm] = {SETS[s]: measure_set(rows, s, rng) for s in SETS}
        per[arm]["whole"]["predicted"] = (
            float(np.percentile(per[arm]["whole"]["predicted_p"], 5)),
            float(np.percentile(per[arm]["whole"]["predicted_p"], 95)))
        per[arm]["rows"] = rows
    names = sorted(per)
    mean = lambda f: statistics.fmean(f(per[a]) for a in names)  # noqa: E731
    tau_w = mean(lambda c: c["whole"]["tau"])
    rng = np.random.default_rng(1)
    imp = [implied(per[a]["rows"], 0, tau_w, rng) for a in names]
    gains = [per[a]["whole"]["tau"] - per[a]["halved"]["tau"] for a in names]
    info = statistics.fmean(
        float(np.mean(per[a]["rows"]["w"] ** 2
                      / (per[a]["rows"]["p"] * (1 - per[a]["rows"]["p"]))))
        for a in names)
    null_z = [c["null_mean"] / c["null_sd"] for a in names
              for c in (per[a]["whole"], per[a]["halved"])]
    return {
        "per": per, "arms": names, "budget": budget, "schedule": schedule,
        "gpu": gpu, "min_attempts": SPREAD_MIN_ATTEMPTS,
        "problems_whole": mean(lambda c: c["whole"]["problems"]),
        "problems_halved": mean(lambda c: c["halved"]["problems"]),
        "tau_whole": tau_w,
        "tau_whole_lo": min(per[a]["whole"]["tau_lo"] for a in names),
        "tau_whole_hi": max(per[a]["whole"]["tau_hi"] for a in names),
        "tau_halved": mean(lambda c: c["halved"]["tau"]),
        "tau_halved_lo": min(per[a]["halved"]["tau_lo"] for a in names),
        "tau_halved_hi": max(per[a]["halved"]["tau_hi"] for a in names),
        "gains": gains, "halving_gain": statistics.fmean(gains),
        "halving_gain_every_seed": int(all(g > 0 for g in gains)),
        "null_bias_sd": float(np.mean(null_z)),
        "predicted_p05": mean(lambda c: c["whole"]["predicted"][0]),
        "predicted_p95": mean(lambda c: c["whole"]["predicted"][1]),
        "true_p05": statistics.fmean(i["p05"] for i in imp),
        "true_p95": statistics.fmean(i["p95"] for i in imp),
        "true_below60": statistics.fmean(i["below60"] for i in imp),
        "info_loss": statistics.fmean(i["info_loss"] for i in imp),
        "info_per_attempt": info,
        "recoverable": recoverable(tau_w, info),
        "rank_error": rank_error(per, np.random.default_rng(2)),
    }


def figures():
    m = measure()
    n = len(m["arms"])
    re_ = m["rank_error"]
    return [
        Figure("spread.seeds", n, "d",
               "fits behind the spread figures, each with its own held-out draw"),
        Figure("spread.min_attempts", m["min_attempts"], "d",
               "training rows a problem needs to be eligible for the holdout"),
        Figure("spread.problems_whole", m["problems_whole"], "~100",
               "problems held out whole per fit, with at least one cross-user pair"),
        Figure("spread.problems_halved", m["problems_halved"], "~100",
               "problems held out by half per fit, with at least one cross-user pair"),
        Figure("spread.tau_whole", m["tau_whole"], ".2f",
               "spread of the problem's true logit about the fitted one, problems "
               "held out whole (features only), mean over fits"),
        Figure("spread.tau_whole_lo", m["tau_whole_lo"], ".2f",
               "lowest per-fit 95% bootstrap lower bound of that spread"),
        Figure("spread.tau_whole_hi", m["tau_whole_hi"], ".2f",
               "highest per-fit 95% bootstrap upper bound of that spread"),
        Figure("spread.tau_halved", m["tau_halved"], ".2f",
               "the same spread on problems whose other half of attempts was "
               "trained on, mean over fits"),
        Figure("spread.tau_halved_lo", m["tau_halved_lo"], ".2f",
               "lowest per-fit 95% bootstrap lower bound, halved set"),
        Figure("spread.tau_halved_hi", m["tau_halved_hi"], ".2f",
               "highest per-fit 95% bootstrap upper bound, halved set"),
        Figure("spread.halving_gain", m["halving_gain"], "+.2f",
               "whole minus halved spread, paired within each fit and averaged: "
               "what training on half a problem's attempts removes"),
        Figure("spread.halving_gain_every_seed", m["halving_gain_every_seed"], "d",
               "1 when that gain is positive in every fit. Backs a fig-guard"),
        Figure("spread.null_bias_sd", m["null_bias_sd"], "+.2f",
               "the estimator's mean on outcomes simulated with no departure, in "
               "units of its own spread; near zero when it is unbiased"),
        Figure("spread.predicted_p05", m["predicted_p05"], "%.0f",
               "5th percentile of the model's predicted success on the whole "
               "set's rows, percent"),
        Figure("spread.predicted_p95", m["predicted_p95"], "%.0f",
               "95th percentile of that prediction, percent"),
        Figure("spread.true_p05", m["true_p05"], "%.0f",
               "5th percentile of the implied true success on those rows, percent"),
        Figure("spread.true_p95", m["true_p95"], "%.0f",
               "95th percentile of the implied true success, percent"),
        Figure("spread.true_below60_pct", m["true_below60"], "%.0f",
               "share of those rows whose implied true success is below 60%, "
               "percent"),
        Figure("spread.info_loss_pct", m["info_loss"], "%.0f",
               "Fisher information per attempt lost to the spread against every "
               "attempt sitting at its prediction, percent"),
        Figure("spread.recoverable_pct", m["recoverable"], "%.0f",
               "attempt-weighted share of a problem's departure variance that a "
               "shrunken per-problem estimate from the other attempts on it "
               "would recover, percent"),
        Figure("spread.rank_err_median", re_["median"], ".1f",
               "median distance in ranks between a problem's true calibration "
               "point and the fitted one, over the whole set's attempts, each "
               "band's own departure spread times the attempt's ranks per logit"),
        Figure("spread.rank_err_p90", re_["p90"], ".1f",
               "90th percentile of that distance, ranks"),
    ] + [f for b in re_["bands"] for f in (
        Figure(f"spread.tau_{b['key']}", b["tau"], ".2f",
               f"departure spread on the logit read from pairs of {b['label']} "
               f"players only, problems held out whole, pooled over fits"),
        Figure(f"spread.ranks_per_logit_{b['key']}", b["rpl"], ".0f",
               f"median ranks one logit is worth on {b['label']} players' "
               f"attempts: the ability-to-rank curve's local slope over the "
               f"problem's own slope"),
        Figure(f"spread.rank_err_median_{b['key']}", b["median"], ".1f",
               f"median rank distance for {b['label']} players' attempts"),
        Figure(f"spread.rank_err_p90_{b['key']}", b["p90"], ".1f",
               f"90th percentile rank distance for {b['label']} players' attempts"),
    )]


def main():
    m = measure()
    print(f"problem_spread panel  ({len(m['arms'])} fits, {m['budget']} epochs "
          f"{m['schedule']}; {m['gpu']}; eligible = at least {m['min_attempts']} "
          f"training rows)\n")
    print("tau = spread of a problem's true logit about the fitted one, from "
          "within-problem cross-user residual covariance on held-out rows;\n"
          "  whole  = problem held out entirely (features only)\n"
          "  halved = half its rows trained on, half scored\n"
          "null = the estimator on outcomes simulated from the fit's own "
          "predictions (true tau 0)\n")
    print(f"{'fit':>5}{'set':>8}{'problems':>10}{'rows':>8}{'tau':>7}"
          f"{'95% boot':>14}{'null tau^2':>18}")
    for a in m["arms"]:
        for s in ("whole", "halved"):
            c = m["per"][a][s]
            print(f"{a:>5}{s:>8}{c['problems']:>10}{c['rows']:>8}{c['tau']:>7.2f}"
                  f"   [{c['tau_lo']:.2f},{c['tau_hi']:.2f}]"
                  f"{c['null_mean']:>+10.3f} ± {c['null_sd']:.3f}")
    print(f"\nmean tau: whole {m['tau_whole']:.2f}, halved {m['tau_halved']:.2f}; "
          f"whole - halved per fit: "
          + " ".join(f"{g:+.2f}" for g in m["gains"]))
    print(f"estimator null bias: {m['null_bias_sd']:+.2f} of its own sd")
    print(f"\nwhole set, success rate 5th-95th percentile:\n"
          f"  predicted  {m['predicted_p05']:.2f} – {m['predicted_p95']:.2f}\n"
          f"  implied    {m['true_p05']:.2f} – {m['true_p95']:.2f}   "
          f"({m['true_below60']:.0%} of rows truly below 0.60)\n"
          f"information per attempt lost to the spread: {m['info_loss']:.0%}")
    print(f"a per-problem estimate from the other attempts on an attempt's "
          f"problem would recover {m['recoverable']:.0%} of its departure "
          f"variance (information per attempt about the offset "
          f"{m['info_per_attempt']:.3f})")
    re_ = m["rank_error"]
    print(f"\nrank distance of a problem's true calibration point from the fitted "
          f"one, whole set, tau read within each band ({re_['dropped']:.1%} of "
          f"rows dropped: player not in the shipped fit)")
    print(f"{'band':>16}{'rows':>8}{'problems':>10}{'tau':>7}{'95% boot':>14}"
          f"{'ranks/logit':>13}{'median':>8}{'p90':>7}")
    for b in re_["bands"]:
        print(f"{b['label']:>16}{b['rows']:>8}{b['problems']:>10}{b['tau']:>7.2f}"
              f"   [{b['tau_lo']:.2f},{b['tau_hi']:.2f}]{b['rpl']:>13.1f}"
              f"{b['median']:>8.1f}{b['p90']:>7.1f}")
    print(f"{'all':>16}{'':>8}{'':>10}{'':>7}{'':>14}{'':>13}"
          f"{re_['median']:>8.1f}{re_['p90']:>7.1f}")


if __name__ == "__main__":
    main()
