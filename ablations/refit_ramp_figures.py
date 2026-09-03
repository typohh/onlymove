#!/usr/bin/env python3
"""The opening learning-rate ramp (methodology §1.8), read from the `ramp` refit
panel: the shipped fit at several ramp LENGTHS, many seeds each, at one pinned
held-out draw and feature basis, so the difference between two lengths at one
seed is the ramp alone.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` and hard-fails when the panel is missing, short of
budget, internally inconsistent, stale against the attempt log, or when the
sweep cannot answer the question it was run for.

WHAT IS MEASURED. Per length: the held-out-PROBLEM level of each seed (the mean
of a run's last 2% of checkpoints, the reading the schedule panel settled on),
and its PAIRED difference from the no-ramp arm at the same seed — mean, a
t-interval, a paired t-test and the count of seeds on the better side, because
"same sign on every seed" is a sign test whose p is 2^(1-n) and eight seeds is
what a millinat-sized effect needs. The same paired difference with the axis
term zeroed says whether a gain is the rating's. And, since a ramp may buy
level by moving the basis, each length's arms are read for cross-seed identity
(matched-axis |correlation| over all seed pairs) and for agreement with the
taught rule ends (the trainer's own identity report).

THE SWEEP MUST BRACKET. A length that wins at the sweep's longest point is a
boundary, not an optimum: `ramp.best_at_boundary` is 1 in that case and the
panel is to be extended with longer ramps before any length is named. The
SHIPPED length is read from the trainer (`LR_WARMUP_EPOCHS`, 0 while no ramp
ships) and is required to be one of the panel's lengths, so the figures that
describe production describe an arm that was actually run.

  ramp.lengths / ramp.seeds           the sweep, and seeds per length
  ramp.shipped_length                 what production runs
  ramp.best_length                    the length with the best mean level
  ramp.best_at_boundary               1 if that is the longest length tried
  ramp.l<R>_gain_mnat                 no-ramp level minus length R's, paired per
                                      seed and averaged; positive = R better
  ramp.l<R>_gain_ci_mnat              half-width of its 95% t-interval
  ramp.l<R>_gain_p                    paired t-test p
  ramp.l<R>_seeds_better              seeds on which R beats no ramp
  ramp.l<R>_rating_gain_mnat          the same gain with the axis term zeroed
  ramp.l<R>_basis_mean                cross-seed matched-axis agreement
  ramp.l<R>_taught_agreement          mean agreement with the taught rule ends
  ramp.l<R>_work_epochs               constant-rate epochs of work delivered
  ramp.level_sd_mnat                  run-to-run sd pooled over lengths

    ./.venv/bin/python3 -m ablations.refit_ramp_figures
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
# Shared readings, imported rather than restated: the terminal-window level
# from the schedule panel, the agreement statistics from the axis-space panel.
from ablations.refit_axis_space_figures import (  # noqa: E402
    EMBED_COLS, attempted_keys, pair_stats, standardized)
from ablations.refit_schedule_figures import level  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.model_constants import K_AXES  # noqa: E402

TIER = "refit"

PANEL = "ramp"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")


def shipped_length():
    """The ramp production runs, from the trainer itself — 0 while none ships."""
    from pipeline import train_mirt
    return int(getattr(train_mirt, "LR_WARMUP_EPOCHS", 0))


def load_panel():
    require(os.path.exists(MANIFEST), f"no {MANIFEST}")
    manifest = json.load(open(MANIFEST))
    arms = {}
    for arm, cfg in manifest["arms"].items():
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the ramp panel has not finished. Run it with "
                f"`python -m ablations.run_panel ramp` (or "
                f"`vast_orchestration.py ablate ramp`).")
        rec = json.load(open(path))
        require(rec["config"] == cfg,
                f"{arm} ran {rec['config']!r}, the manifest declares {cfg!r} — "
                f"the record is not this arm")
        require(cfg.get("stop_unit") == "problem",
                f"{arm} holds aside rows, not problems")
        arms[arm] = rec
    budget, schedule = panel_regime(arms)
    for arm, r in arms.items():
        require(r["metrics"]["trace"]
                and r["metrics"]["trace"][-1]["epoch"] == budget,
                f"{arm}'s trace does not reach the {budget}-epoch budget")

    base = next(iter(arms.values()))["pins"]
    for arm, r in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha", "blocks",
                    "pca_dim", "k_axes"):
            require(r["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} differs — the arms are not one panel")
    require(int(base["k_axes"]) == int(K_AXES),
            f"fitted at K={base['k_axes']}, the shipping constant is {K_AXES}")
    for field in ("split_seed", "pca_seed"):
        vals = {r["metrics"].get(field) for r in arms.values()}
        require(len(vals) == 1 and None not in vals,
                f"the arms record {len(vals)} different {field} values — one "
                f"pinned draw and basis is this panel's design")
    rows = {(r["metrics"]["stop_rows"], r["metrics"]["rows"]) for r in arms.values()}
    require(len(rows) == 1, f"the arms held aside different row counts {rows}")

    # Group by ramp length, read off the RECORDS; every length needs the same
    # seed set, else a paired contrast has nothing to pair.
    by_len = {}
    for arm, r in arms.items():
        R = int(r["metrics"].get("lr_warmup_epochs", 0))
        require(R == int(r["config"].get("lr_warmup_epochs", 0)),
                f"{arm} recorded a {R}-epoch ramp against its manifest entry")
        by_len.setdefault(R, {})[int(r["config"].get("seed", 0))] = (arm, r)
    require(0 in by_len, "the panel carries no no-ramp arm to pair against")
    require(len(by_len) >= 2, "the panel carries a single length — no sweep")
    seed_sets = {R: tuple(sorted(d)) for R, d in by_len.items()}
    require(len(set(seed_sets.values())) == 1,
            f"the lengths ran different seed sets: {seed_sets}")
    seeds = seed_sets[0]
    require(len(seeds) >= 2, "a length needs at least two seeds")
    ships = shipped_length()
    require(ships in by_len,
            f"production runs a {ships}-epoch ramp and the panel's lengths are "
            f"{sorted(by_len)} — the shipping configuration has to be an arm")

    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute("SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"measured at attempt-log frontier {base['db_max_id']}, the log now "
            f"ends at {now} — re-run the panel")
    return by_len, seeds, budget, schedule, ships, base["gpu"]


def load_loadings(arm):
    path = os.path.join(PANEL_DIR, arm, "axes", "per_problem.csv")
    require(os.path.exists(path), f"no {path}")
    df = pd.read_csv(path, dtype={"tsumego_id": str})
    require(list(df.columns) == ["tsumego_id", "size"] + EMBED_COLS,
            f"{path} has an unexpected layout")
    df = df.set_index(df["size"].astype(str) + ":" + df["tsumego_id"])
    require(not df.index.has_duplicates, f"{path} carries a duplicate key")
    return df


def taught_agreement(metrics):
    """Mean agreement of each axis's TAUGHT end with its rule, from the
    trainer's own identity report: a one-sided rule is the positive tail, an
    `A/B` rule is both."""
    vals = []
    for row, spec in zip(metrics["identity"], metrics["bank_spec"]):
        if "/" in spec:
            vals.append((row["pos_tail"] + row["neg_tail"]) / 2.0)
        else:
            vals.append(row["pos_tail"])
    require(len(vals) == K_AXES, "the identity report does not cover every axis")
    return float(np.mean(vals))


def paired_stats(deltas):
    """mean, 95% t half-width, two-sided paired-t p, count > 0."""
    from scipy import stats
    d = np.asarray(deltas, float)
    n = len(d)
    mean = float(d.mean())
    if n < 2 or d.std(ddof=1) == 0:
        return mean, float("nan"), float("nan"), int((d > 0).sum())
    se = d.std(ddof=1) / math.sqrt(n)
    ci = float(stats.t.ppf(0.975, n - 1) * se)
    p = float(stats.ttest_1samp(d, 0.0).pvalue)
    return mean, ci, p, int((d > 0).sum())


def measure():
    by_len, seeds, budget, schedule, ships, gpu = load_panel()
    lengths = sorted(by_len)
    base_levels = {s: level(by_len[0][s][1]["metrics"]) for s in seeds}
    base_noax = {s: by_len[0][s][1]["metrics"]["stop_bce_no_axes"] for s in seeds}

    keys = None
    loads = {}
    per = {}
    for R in lengths:
        arms = by_len[R]
        levels = {s: level(arms[s][1]["metrics"]) for s in seeds}
        gains = [1e3 * (base_levels[s] - levels[s]) for s in seeds]
        rgains = [1e3 * (base_noax[s] - arms[s][1]["metrics"]["stop_bce_no_axes"])
                  for s in seeds]
        work = {round(arms[s][1]["metrics"]["lr_integral_epochs"]) for s in seeds}
        require(len(work) == 1, f"length {R}'s arms delivered different work {work}")
        mean, ci, p, better = paired_stats(gains)
        rmean, _, _, _ = paired_stats(rgains)
        for s in seeds:
            loads[arms[s][0]] = load_loadings(arms[s][0])
        per[R] = {
            "levels": levels, "level": statistics.fmean(levels.values()),
            "sd_mnat": 1e3 * statistics.stdev(levels.values()),
            "gains": gains, "gain": mean, "gain_ci": ci, "gain_p": p,
            "seeds_better": better, "rating_gain": rmean,
            "work": work.pop(),
            "taught_agreement": statistics.fmean(
                taught_agreement(arms[s][1]["metrics"]) for s in seeds),
            "elapsed_s": statistics.median(arms[s][1]["metrics"]["elapsed_s"]
                                           for s in seeds),
        }
    common = set(attempted_keys())
    for df in loads.values():
        common &= set(df.index)
    keys = sorted(common)
    require(len(keys) > 1000, "too few attempted problems shared by every arm")
    Z = {arm: standardized(df, keys, arm) for arm, df in loads.items()}
    for R in lengths:
        names = [by_len[R][s][0] for s in seeds]
        vals = [pair_stats(Z[a], Z[b])["basis"].mean()
                for a, b in itertools.combinations(names, 2)]
        per[R]["basis_mean"] = float(np.mean(vals))
        per[R]["basis_min"] = float(np.min(vals))

    ss = sum((x - per[R]["level"]) ** 2 for R in lengths for x in per[R]["levels"].values())
    df_ = sum(len(seeds) - 1 for _ in lengths)
    pooled_sd = 1e3 * math.sqrt(ss / df_)
    best = min(lengths, key=lambda R: per[R]["level"])
    return {"per": per, "lengths": lengths, "seeds": seeds, "budget": budget,
            "schedule": schedule, "ships": ships, "gpu": gpu, "keys": len(keys),
            "pooled_sd": pooled_sd, "best": best,
            "best_at_boundary": int(best == max(lengths))}


def figures():
    m = measure()
    figs = [
        Figure("ramp.lengths", len(m["lengths"]), "d",
               f"ramp lengths swept: {', '.join(str(R) for R in m['lengths'])} "
               f"epochs (0 = no ramp)"),
        Figure("ramp.seeds", len(m["seeds"]), "d",
               "fits per length, differing only in initialization and batch "
               "order — held-out draw and feature basis pinned across the panel"),
        Figure("ramp.shipped_length", m["ships"], "d",
               "the ramp production runs, read from the trainer"),
        Figure("ramp.best_length", m["best"], "d",
               "the length with the best mean held-out-problem level"),
        Figure("ramp.best_at_boundary", m["best_at_boundary"], "d",
               "1 if the best length is the longest tried — a boundary, not an "
               "optimum, and the sweep has to be extended"),
        Figure("ramp.level_sd_mnat", m["pooled_sd"], ".1f",
               "run-to-run sd of the held-out-problem level, pooled over "
               "lengths, millinats"),
        Figure("ramp.problems_scored", m["keys"], "~1000",
               "attempted problems the identity figures are measured over"),
    ]
    for R in m["lengths"]:
        c = m["per"][R]
        figs += [
            Figure(f"ramp.l{R}_work_epochs", c["work"], "d",
                   f"constant-rate epochs of work a {R}-epoch ramp delivers"),
            Figure(f"ramp.l{R}_taught_agreement", c["taught_agreement"], ".2f",
                   f"{R}-epoch ramp: mean agreement of each axis's taught end "
                   f"with its rule, over the seeds"),
            Figure(f"ramp.l{R}_basis_mean", c["basis_mean"], ".2f",
                   f"{R}-epoch ramp: cross-seed matched-axis |correlation|, "
                   f"averaged over all seed pairs"),
        ]
        if R == 0:
            continue
        figs += [
            Figure(f"ramp.l{R}_gain_mnat", c["gain"], "+.1f",
                   f"no-ramp level minus the {R}-epoch ramp's, paired per seed "
                   f"on held-out problems and averaged, millinats; positive = "
                   f"the ramp predicts better"),
            Figure(f"ramp.l{R}_gain_ci_mnat", c["gain_ci"], ".1f",
                   "half-width of that gain's 95% t-interval, millinats"),
            Figure(f"ramp.l{R}_gain_p", c["gain_p"], ".3f",
                   "two-sided paired t-test p for that gain"),
            Figure(f"ramp.l{R}_seeds_better", c["seeds_better"], "d",
                   f"of {len(m['seeds'])} seeds, how many the {R}-epoch ramp "
                   f"beats no ramp on"),
            Figure(f"ramp.l{R}_rating_gain_mnat", c["rating_gain"], "+.1f",
                   "the same paired gain with the axis term zeroed — the "
                   "rating's share of it"),
        ]
    return figs


def main():
    m = measure()
    print(f"ramp panel  (full budget {m['budget']} epochs, {m['schedule']} rate; "
          f"{len(m['seeds'])} seeds per length at one pinned draw and basis; "
          f"{m['gpu']}; identity over {m['keys']} attempted problems)")
    print(f"production ramp: {m['ships']} epochs; run-to-run sd pooled over "
          f"lengths: {m['pooled_sd']:.1f} mnat\n")
    print(f"{'ramp':>6}{'work':>6}{'level':>9}{'sd':>6}{'gain vs 0':>11}{'±95%':>7}"
          f"{'p':>8}{'better':>8}{'rating':>8}{'basis':>7}{'taught':>8}{'s/arm':>7}")
    for R in m["lengths"]:
        c = m["per"][R]
        g = "" if R == 0 else f"{c['gain']:>+11.2f}{c['gain_ci']:>7.2f}{c['gain_p']:>8.3f}{c['seeds_better']:>5}/{len(m['seeds'])}{c['rating_gain']:>+8.2f}"
        if R == 0:
            g = f"{'—':>11}{'':>7}{'':>8}{'':>8}{'':>8}"
        print(f"{R:>6}{c['work']:>6}{c['level']:>9.4f}{c['sd_mnat']:>6.1f}{g}"
              f"{c['basis_mean']:>7.2f}{c['taught_agreement']:>8.2f}{c['elapsed_s']:>7.0f}")
    print(f"\nper-seed gains vs no ramp (mnat):")
    for R in m["lengths"][1:]:
        print(f"  {R:>5}: " + " ".join(f"{g:+.2f}" for g in m["per"][R]["gains"]))
    print(f"\nbest length {m['best']}"
          + ("  — AT THE BOUNDARY: extend the sweep with longer ramps before naming a length"
             if m["best_at_boundary"] else "  (interior)"))


if __name__ == "__main__":
    main()
