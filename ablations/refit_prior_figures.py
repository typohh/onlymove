#!/usr/bin/env python3
"""What the ability prior and the base-rate start each buy (methodology §1.5,
§1.8), read from the `prior_start` refit panel: the shipped fit against the
same fit with the prior off, the start off, and both off.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` and hard-fails when the panel is missing, short of
budget, internally inconsistent or stale against the attempt log.

WHAT IS MEASURED. The claim both switches stand on is CONSISTENCY — the served
attempts barely inform the curve below each problem's calibration point, so
what they are for is that the parts the data does not pin come out the same
from one seed to the next and that no user is explained by floors. So each
configuration is read for:

  - cross-seed agreement of the per-problem floors, slopes and ratings (the
    Spearman correlation between two seeds' values over every problem,
    averaged over all seed pairs);
  - the spread across seeds of the floor's centre (the median floor) and of
    the lapse;
  - how much of a user's predicted success the floors carry: over a user's
    attempts after their first PARK_SKIP, the floor's share is
    sum(floor * (1 - curve)) / sum(prediction), curve being
    sigmoid(slope * (ability - difficulty)) — the part of each prediction that
    is there whatever the ability. Read over users with at least
    PARK_MIN_ATTEMPTS such attempts: the count whose share passes PARK_SHARE
    (explained mostly by floors) and the largest share, worst seed; and the
    attempt-weighted share over all of them, lowest and highest seed;
  - problems whose floor reaches the calibration probability and so ship no
    defined rating (section 1.9). Worst seed;

and beside those its held-out-problem level, PAIRED against the shipped arm at
the same seed (same initialization, same held-out draw), with a t-interval,
a paired t-test and the count of seeds on the better side.

  prior.<cfg>_gain_mnat           shipped level minus the arm's, paired per
  prior.<cfg>_gain_ci_mnat        seed; positive = the arm predicts better;
  prior.<cfg>_gain_p              its 95% half-width, paired t p, and seeds
  prior.<cfg>_seeds_better        on the better side (non-shipped configs)
  prior.<cfg>_floor_agree         cross-seed Spearman of per-problem floors
  prior.<cfg>_slope_agree         ... of slopes
  prior.<cfg>_rating_agree        ... of ratings
  prior.<cfg>_floor_median_lo/hi  lowest / highest median floor over seeds
  prior.<cfg>_lapse_lo/hi         lowest / highest lapse over seeds
  prior.<cfg>_parked_max          users whose floor share passes PARK_SHARE,
                                  worst seed
  prior.<cfg>_parked_attempts_max their attempts, worst seed
  prior.<cfg>_user_share_max      the largest single user's floor share,
                                  worst seed
  prior.<cfg>_floor_share_lo/hi   attempt-weighted floor share, lowest /
                                  highest seed
  prior.<cfg>_above_calibration_max  floors at or above 0.8, worst seed
  prior.seeds / prior.park_min_attempts / prior.park_skip / prior.park_share
  prior.level_sd_mnat             run-to-run sd pooled over configurations

    ./.venv/bin/python3 -m ablations.refit_prior_figures
"""

import itertools
import json
import math
import os
import sqlite3
import statistics
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require, panel_regime  # noqa: E402
from ablations.refit_ramp_figures import paired_stats  # noqa: E402
from ablations.refit_schedule_figures import level  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.model_constants import CALIBRATION_P, SIZES  # noqa: E402

TIER = "refit"

PANEL = "prior_start"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")
SEEDS = tuple(range(4))
# Configuration -> arm pattern and the (theta_laplace, init_base_rate) its
# records must carry.
CONFIGS = {
    "shipped": ("shipped_s{}", (True, True)),
    "no_prior": ("no_prior_s{}", (False, True)),
    "no_start": ("no_start_s{}", (True, False)),
    "neither": ("neither_s{}", (False, False)),
}
CFG_NOUN = {
    "shipped": "the shipped fit (ability prior and base-rate start)",
    "no_prior": "the ability prior off",
    "no_start": "the base-rate start off",
    "neither": "both off",
}
# Who counts as explained by floors. The skip keeps a user's opening run,
# which clears far above the calibration rate before serving has found their
# level, out of the reading.
PARK_SKIP = 30
PARK_MIN_ATTEMPTS = 30
PARK_SHARE = 0.5


def load_panel():
    require(os.path.exists(MANIFEST), f"no {MANIFEST}")
    manifest = json.load(open(MANIFEST))
    want = {pat.format(s) for pat, _ in CONFIGS.values() for s in SEEDS}
    require(set(manifest["arms"]) == want,
            f"{MANIFEST} declares {sorted(manifest['arms'])}, this producer "
            f"reads {sorted(want)}")
    arms = {}
    for arm, cfg in manifest["arms"].items():
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the prior_start panel has not finished. Run it "
                f"with `python -m ablations.run_panel prior_start`.")
        rec = json.load(open(path))
        require(rec["config"] == cfg,
                f"{arm} ran {rec['config']!r}, the manifest declares {cfg!r}")
        require(cfg.get("stop_unit") == "problem", f"{arm} holds aside rows")
        arms[arm] = rec
    budget, schedule = panel_regime(arms, prior_may_vary=True)
    for arm, r in arms.items():
        require(r["metrics"]["trace"]
                and r["metrics"]["trace"][-1]["epoch"] == budget,
                f"{arm}'s trace does not reach the {budget}-epoch budget")
    base = next(iter(arms.values()))["pins"]
    for arm, r in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha", "blocks",
                    "pca_dim", "k_axes", "floor_mode"):
            require(r["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} differs — the arms are not one panel")
    for field in ("split_seed", "pca_seed"):
        vals = {r["metrics"].get(field) for r in arms.values()}
        require(len(vals) == 1 and None not in vals,
                f"the arms record {len(vals)} different {field} values")
    rows = {(r["metrics"]["stop_rows"], r["metrics"]["rows"]) for r in arms.values()}
    require(len(rows) == 1, f"the arms held aside different row counts {rows}")
    for cfg, (pat, (lap, start)) in CONFIGS.items():
        for s in SEEDS:
            m = arms[pat.format(s)]["metrics"]
            require(bool(m.get("theta_laplace", False)) == lap
                    and bool(m.get("init_base_rate", False)) == start,
                    f"{pat.format(s)} records theta_laplace="
                    f"{m.get('theta_laplace')} init_base_rate="
                    f"{m.get('init_base_rate')}")

    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute("SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"measured at attempt-log frontier {base['db_max_id']}, the log now "
            f"ends at {now} — re-run the panel")
    return arms, budget, schedule, base["gpu"]


def load_fit(arm):
    """Per-problem (rating, slope, floor, difficulty) keyed by (size, id), the
    abilities keyed by user id, and the lapse."""
    d = os.path.join(PANEL_DIR, arm, "ratings")
    lam = json.load(open(os.path.join(d, "rating_constants.json")))["fat_finger_rate"]
    parts = []
    for z in SIZES:
        f = None
        for kind in ("ratings", "slopes", "floors"):
            path = os.path.join(d, str(z), f"tsumego_{kind}.csv")
            require(os.path.exists(path), f"no {path}")
            t = pd.read_csv(path, dtype={"tsumego_id": str})
            f = t if f is None else f.merge(t, on="tsumego_id", how="outer")
        require(not f.isna().any().any(), f"{arm} {z}: a problem lacks a parameter")
        parts.append(f.assign(size=z))
    f = pd.concat(parts).set_index(["size", "tsumego_id"])
    require(not f.index.has_duplicates, f"{arm}: a duplicate problem key")
    # The rating is the 80% point; the difficulty is recovered from it.
    q = ((CALIBRATION_P - f.floor) / (1 - f.floor - lam)).clip(1e-6, 1 - 1e-6)
    f["difficulty"] = f.rating - np.log(q / (1 - q)) / f.slope
    th = pd.read_csv(os.path.join(d, "user_theta.csv"))
    th = th.set_index(th.profile.astype(str)).theta
    return f, th, float(lam)


def late_attempts():
    from pipeline.train_mirt import load_first_time_attempts
    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)].copy()
    df["user_id"] = df.user_id.astype(str)
    df["problem"] = df.problem.astype(str)
    df = df.sort_values(["user_id", "created_at"])
    df["k"] = df.groupby("user_id").cumcount()
    return df[df.k >= PARK_SKIP]


def parked(f, th, lam, late):
    # The fit's own row filter: attempts on problems outside the rated set
    # (retired ones) and users the trainer drops for lacking both outcomes
    # never entered it.
    key = pd.MultiIndex.from_arrays([late["size"].values, late.problem.values])
    keep = key.isin(f.index) & late.user_id.isin(th.index).values
    late, key = late[keep], key[keep]
    pf = f.reindex(key)
    ab = late.user_id.map(th).values
    curve = 1 / (1 + np.exp(-np.clip(pf.slope.values * (ab - pf.difficulty.values), -35, 35)))
    c = pf.floor.values
    g = (pd.DataFrame({"u": late.user_id.values, "fl": c * (1 - curve),
                       "p": c + (1 - c - lam) * curve})
         .groupby("u").agg(n=("p", "size"), fl=("fl", "sum"), p=("p", "sum")))
    g = g[g.n >= PARK_MIN_ATTEMPTS]
    share = g.fl / g.p
    hit = g[share > PARK_SHARE]
    return {"parked": int(len(hit)), "parked_attempts": int(hit.n.sum()),
            "user_share_max": float(share.max()),
            "floor_share": float(g.fl.sum() / g.p.sum()), "users": int(len(g))}


def agreement(fits, col):
    vals = []
    for a, b in itertools.combinations(fits, 2):
        x, y = a[col].align(b[col], join="inner")
        vals.append(x.rank().corr(y.rank()))
    return float(np.mean(vals))


def measure():
    arms, budget, schedule, gpu = load_panel()
    late = late_attempts()
    base_lv = {s: level(arms[CONFIGS["shipped"][0].format(s)]["metrics"]) for s in SEEDS}
    per = {}
    users = None
    for cfg, (pat, _) in CONFIGS.items():
        names = [pat.format(s) for s in SEEDS]
        loaded = [load_fit(a) for a in names]
        fits = [f for f, _, _ in loaded]
        park = [parked(f, th, lam, late) for f, th, lam in loaded]
        users = park[0]["users"]
        meds = [float(f.floor.median()) for f in fits]
        lams = [lam for _, _, lam in loaded]
        lv = {s: level(arms[names[s]]["metrics"]) for s in SEEDS}
        c = {"levels": lv, "level": statistics.fmean(lv.values()),
             "floor_agree": agreement(fits, "floor"),
             "slope_agree": agreement(fits, "slope"),
             "rating_agree": agreement(fits, "rating"),
             "floor_median_lo": min(meds), "floor_median_hi": max(meds),
             "lapse_lo": min(lams), "lapse_hi": max(lams),
             "parked_max": max(p["parked"] for p in park),
             "parked_attempts_max": max(p["parked_attempts"] for p in park),
             "parked": [p["parked"] for p in park],
             "user_share_max": max(p["user_share_max"] for p in park),
             "floor_share_lo": min(p["floor_share"] for p in park),
             "floor_share_hi": max(p["floor_share"] for p in park),
             "above_calibration_max": max(int((f.floor >= CALIBRATION_P).sum()) for f in fits)}
        if cfg != "shipped":
            gains = [1e3 * (base_lv[s] - lv[s]) for s in SEEDS]
            c["gains"] = gains
            c["gain"], c["gain_ci"], c["gain_p"], c["seeds_better"] = paired_stats(gains)
        per[cfg] = c
    ss = sum((x - per[c]["level"]) ** 2 for c in CONFIGS for x in per[c]["levels"].values())
    pooled = 1e3 * math.sqrt(ss / (len(CONFIGS) * (len(SEEDS) - 1)))
    return {"per": per, "budget": budget, "schedule": schedule, "gpu": gpu,
            "users": users, "pooled_sd": pooled}


def figures():
    m = measure()
    figs = [
        Figure("prior.seeds", len(SEEDS), "d",
               "fits per configuration, paired by seed at one pinned draw"),
        Figure("prior.park_min_attempts", PARK_MIN_ATTEMPTS, "d",
               "attempts after the opening run a user needs to be read for "
               "floor explanation"),
        Figure("prior.park_skip", PARK_SKIP, "d",
               "opening attempts skipped before that reading"),
        Figure("prior.park_share", PARK_SHARE, ".1f",
               "floor share of a user's predicted success above which they "
               "count as explained mostly by floors"),
        Figure("prior.users_read", m["users"], "d",
               "users with enough attempts to be read for floor explanation"),
        Figure("prior.level_sd_mnat", m["pooled_sd"], ".1f",
               "run-to-run sd of the held-out-problem level, pooled over the "
               "configurations, millinats"),
    ]
    for cfg, c in m["per"].items():
        noun = CFG_NOUN[cfg]
        figs += [
            Figure(f"prior.{cfg}_floor_agree", c["floor_agree"], ".2f",
                   f"{noun}: cross-seed Spearman of the per-problem floors, "
                   f"averaged over seed pairs"),
            Figure(f"prior.{cfg}_slope_agree", c["slope_agree"], ".2f",
                   f"{noun}: the same for slopes"),
            Figure(f"prior.{cfg}_rating_agree", c["rating_agree"], ".3f",
                   f"{noun}: the same for ratings"),
            Figure(f"prior.{cfg}_floor_median_lo", c["floor_median_lo"], ".2f",
                   f"{noun}: lowest median floor over the seeds"),
            Figure(f"prior.{cfg}_floor_median_hi", c["floor_median_hi"], ".2f",
                   f"{noun}: highest median floor over the seeds"),
            Figure(f"prior.{cfg}_lapse_lo", c["lapse_lo"], ".3f",
                   f"{noun}: lowest lapse over the seeds"),
            Figure(f"prior.{cfg}_lapse_hi", c["lapse_hi"], ".3f",
                   f"{noun}: highest lapse over the seeds"),
            Figure(f"prior.{cfg}_parked_max", c["parked_max"], "d",
                   f"{noun}: users explained mostly by floors, worst seed"),
            Figure(f"prior.{cfg}_user_share_max", c["user_share_max"], ".2f",
                   f"{noun}: the largest single user's floor share, worst seed"),
            Figure(f"prior.{cfg}_floor_share_lo", c["floor_share_lo"], ".3f",
                   f"{noun}: attempt-weighted floor share of predicted "
                   f"success, lowest seed"),
            Figure(f"prior.{cfg}_floor_share_hi", c["floor_share_hi"], ".3f",
                   f"{noun}: the same, highest seed"),
            Figure(f"prior.{cfg}_parked_attempts_max", c["parked_attempts_max"], "d",
                   f"{noun}: the attempts those users hold, worst seed"),
            Figure(f"prior.{cfg}_above_calibration_max", c["above_calibration_max"], "d",
                   f"{noun}: problems whose floor reaches the calibration "
                   f"probability, worst seed"),
        ]
        if cfg == "shipped":
            continue
        figs += [
            Figure(f"prior.{cfg}_gain_mnat", c["gain"], "+.1f",
                   f"{noun}: shipped level minus this arm's on held-out "
                   f"problems, paired per seed and averaged, millinats; "
                   f"positive = this arm predicts better"),
            Figure(f"prior.{cfg}_gain_ci_mnat", c["gain_ci"], ".1f",
                   "half-width of that gain's 95% t-interval, millinats"),
            Figure(f"prior.{cfg}_gain_p", c["gain_p"], ".3f",
                   "two-sided paired t-test p for that gain"),
            Figure(f"prior.{cfg}_seeds_better", c["seeds_better"], "d",
                   f"of {len(SEEDS)} seeds, how many this arm beats the "
                   f"shipped fit on"),
        ]
    return figs


def main():
    m = measure()
    print(f"prior_start panel  (full budget {m['budget']} epochs, {m['schedule']} "
          f"rate; {len(SEEDS)} seeds per configuration at one pinned draw; "
          f"{m['gpu']}; {m['users']} users read for floor explanation; pooled "
          f"run-to-run sd {m['pooled_sd']:.1f} mnat)\n")
    print(f"{'configuration':<10}{'level':>8}{'gain':>7}{'±95%':>6}{'p':>7}{'better':>8}"
          f"{'floorρ':>8}{'slopeρ':>8}{'ratingρ':>9}{'med floor':>12}{'lapse':>13}"
          f"{'parked':>14}{'max share':>10}{'floor share':>14}{'≥0.8':>6}")
    for cfg, c in m["per"].items():
        g = (f"{c['gain']:>+7.2f}{c['gain_ci']:>6.2f}{c['gain_p']:>7.3f}"
             f"{c['seeds_better']:>5}/{len(SEEDS)}"
             if cfg != "shipped" else f"{'—':>7}{'':>6}{'':>7}{'':>8}")
        print(f"{cfg:<10}{c['level']:>8.4f}{g}{c['floor_agree']:>8.2f}"
              f"{c['slope_agree']:>8.2f}{c['rating_agree']:>9.3f}"
              f"{c['floor_median_lo']:>6.2f}–{c['floor_median_hi']:<5.2f}"
              f"{c['lapse_lo']:>6.3f}–{c['lapse_hi']:<6.3f}"
              f"{'/'.join(map(str, c['parked'])):>14}{c['user_share_max']:>10.2f}"
              f"{c['floor_share_lo']:>7.3f}–{c['floor_share_hi']:<6.3f}"
              f"{c['above_calibration_max']:>6}")
    print("\nper-seed gains vs shipped (mnat, + = arm better):")
    for cfg, c in m["per"].items():
        if cfg != "shipped":
            print(f"  {cfg:<10}" + " ".join(f"{g:+.2f}" for g in c["gains"]))


if __name__ == "__main__":
    main()
