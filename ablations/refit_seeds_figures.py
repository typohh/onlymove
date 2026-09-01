#!/usr/bin/env python3
"""What fixes the axis basis (methodology §1.14), read from the `seeds` refit
panel: how well two fits of one configuration, differing only in the random
seed, agree on their DIRECTIONS — with the teaching pull off, and with it on.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` (`ablations.run_panel`, locally or via
`vast_orchestration.py ablate`) and hard-fails when the panel is missing, short
of budget, internally inconsistent, or STALE against the current attempt log —
the pins rule `docs/mirt_benchmark.md` states, mechanized. Guards and spans keep
evaluating against the last-measured ledger values, so the panel's cost is paid
only when it is deliberately re-run.

WHY THE PANEL EXISTS. The likelihood reads the loadings and the user positions
only through their product, so rotating the axis space leaves every predicted
probability unchanged and nothing in the objective prefers one basis to another.
Left alone the fit lands wherever initialization drops it; §1.14 quotes 0.56 /
0.25 / 0.20, a mean of 0.34, for two seeds of one configuration — a K=3-era
figure, re-measured here at the shipped K. The taught arms are the same
comparison with the constant pull toward the rule columns switched back on,
which is the mechanism §1.14 credits with fixing the basis.

  seeds.untaught_cross_seed_mean  cross-seed direction agreement with the pull
  seeds.untaught_cross_seed_min   off: the mean and the weakest of the K matched
                                  axes, averaged over the three seed pairs.
  seeds.taught_cross_seed_mean    the same two statistics for the shipped,
  seeds.taught_cross_seed_min     taught configuration at two seeds.
  seeds.teaching_gap              taught minus untaught, on the means — what the
                                  pull buys in basis stability.
  seeds.axes_matched              axes matched per pair (K as shipped).
  seeds.problems_scored           the population all of the above run over.

THE AGREEMENT STATISTIC, stated because every part of it is a choice:

  1. Each arm's per-problem loadings are read from its own
     `axes/per_problem.csv`, joined across arms by (size, tsumego_id) — never
     positionally. Two fits write the same rows, but nothing in the format
     guarantees the same ORDER, and a positional join would silently score one
     fit's axes against another fit's problems.
  2. The two arms' K axes are matched ONE-TO-ONE by maximum total |correlation|
     over that population (Hungarian, `scipy.optimize.linear_sum_assignment`),
     because with the pull off the axis INDEX carries no meaning: nothing ties
     arm A's axis 3 to arm B's axis 3, so comparing them by index would measure
     the arbitrary ordering rather than the subspace.
  3. Each matched pair's agreement is the ABSOLUTE correlation, which is the
     sign alignment: a direction and its negation are the same direction, the
     sign being absorbed by the user positions.
  4. Per pair that gives K numbers; the pair's summary is their mean and their
     minimum, and the untaught figure is those two averaged over the three
     untaught seed pairs. The three pairs are not independent (three fits, three
     pairings), so the average is a summary and not a standard error.

MEASURED ON ATTEMPTED PROBLEMS ONLY. About half the corpus is never attempted,
and §1.14 states why that matters here: those loadings are unconstrained
extrapolation whose extremes churn freely between fits, so including them counts
free churn as a failure to reproduce. It is also the population the shipped fit's
own identity report scores on (`report_bank_identity`). "Attempted" is every
problem carrying at least one FIRST-TIME attempt in the local DB at the fit's
board sizes, read through `utils.rating.features.load_first_time_attempts` — a
slight superset of the trainer's own identity population, which additionally
drops users without both outcomes. The set is intersected with every arm's rows
once, so all four pairs are scored over exactly the same problems.

NO LIKELIHOOD DELTA IS EMITTED FROM THIS PANEL. The stop rows are drawn from the
seed, so arms at different seeds hold aside DIFFERENT rows and a cross-arm
held-out comparison is unpaired — the one thing `docs/mirt_benchmark.md` says
voids a comparison. Held-out costs belong to a panel whose arms share a seed.

    ./.venv/bin/python3 -m ablations.refit_seeds_figures
"""

import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
from pipeline.train_mirt import EPOCHS  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.features import load_first_time_attempts  # noqa: E402
from utils.rating.model_constants import K_AXES, SIZES  # noqa: E402

TIER = "refit"

PANEL = "seeds"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")
ARMS = ("taught_s0", "taught_s1",
        "untaught_s0", "untaught_s1", "untaught_s2")
TAUGHT_PAIR = ("taught_s0", "taught_s1")
UNTAUGHT_PAIRS = (("untaught_s0", "untaught_s1"),
                  ("untaught_s0", "untaught_s2"),
                  ("untaught_s1", "untaught_s2"))


# --- the panel -------------------------------------------------------------

def load_panel():
    require(os.path.exists(MANIFEST),
            f"no {MANIFEST} — the panel manifest is what the arm records are "
            f"checked against, so it cannot be read without it")
    manifest = json.load(open(MANIFEST))
    require(set(manifest["arms"]) == set(ARMS),
            f"{MANIFEST} declares arms {sorted(manifest['arms'])}, this "
            f"producer reads {sorted(ARMS)}")

    arms = {}
    for arm in ARMS:
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the seeds panel has not run. Run it with "
                f"`python -m ablations.run_panel seeds` (or "
                f"`vast_orchestration.py ablate seeds`) first.")
        arms[arm] = json.load(open(path))

    for arm, r in arms.items():
        require(r["metrics"]["epochs"] == EPOCHS,
                f"{arm} ran {r['metrics']['epochs']} epochs, not the production "
                f"budget of {EPOCHS} — a smoke arm is not quotable")
        require(r["metrics"]["trace"]
                and r["metrics"]["trace"][-1]["epoch"] == EPOCHS,
                f"{arm}'s convergence trace does not reach its budget")
        # A record run with a different flag is a well-formed record of a
        # different experiment; every figure below would be attributed to a
        # configuration that was not fitted.
        require(r["config"] == manifest["arms"][arm],
                f"{arm} was run with {r['config']!r}, but the manifest declares "
                f"{manifest['arms'][arm]!r} — the record is not this arm")

    # The SEED is what this panel varies, so it is the one thing not pinned
    # across arms. Everything else has to agree.
    base = arms["taught_s0"]["pins"]
    for arm, r in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha",
                    "blocks", "pca_dim", "k_axes"):
            require(r["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} ({r['pins'][pin]!r}) differs from the "
                    f"taught_s0 arm's ({base[pin]!r}) — the arms are not one "
                    f"panel and their comparison is void")
    require(int(base["k_axes"]) == int(K_AXES),
            f"the panel was fitted at K={base['k_axes']} but the shipping "
            f"constant is K_AXES={K_AXES} — this measurement is quoted as being "
            f"at the shipped K, so it has to have been")

    seeds = {arm: int(r["config"].get("seed", 0)) for arm, r in arms.items()}
    for a, b in (TAUGHT_PAIR,) + UNTAUGHT_PAIRS:
        require(seeds[a] != seeds[b],
                f"{a} and {b} ran at the same seed ({seeds[a]}) — they are the "
                f"same fit, and their agreement measures nothing")

    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute("SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"the panel was measured at attempt-log frontier {base['db_max_id']} "
            f"but the log now ends at {now} — the arms were fitted against a "
            f"different attempt set than the attempted population read below. "
            f"Re-run the panel.")
    return arms, seeds


# --- the loadings ----------------------------------------------------------

EMBED_COLS = [f"embed_{k}" for k in range(K_AXES)]


def _key(size, tid):
    return size.astype(str) + ":" + tid.astype(str)


def load_loadings(arm):
    """One arm's per-problem axis loadings, indexed by "<size>:<tsumego_id>".

    The id is read as TEXT: it is eight hex characters and an all-digit one
    would otherwise arrive as an integer with its leading zeros gone, which
    joins against nothing."""
    path = os.path.join(PANEL_DIR, arm, "axes", "per_problem.csv")
    require(os.path.exists(path),
            f"no {path} — the arm's record exists but the axis loadings it "
            f"wrote are gone. A record without its artifacts cannot be read.")
    df = pd.read_csv(path, dtype={"tsumego_id": str})
    want = ["tsumego_id", "size"] + EMBED_COLS
    require(list(df.columns) == want,
            f"{path} has columns {list(df.columns)}, expected {want} — the fit "
            f"wrote a different K or a different layout than this producer reads")
    df = df.set_index(_key(df["size"], df["tsumego_id"]))
    require(not df.index.has_duplicates,
            f"{path} carries the same (size, tsumego_id) twice")
    return df


def attempted_keys():
    """"<size>:<tsumego_id>" for every problem with at least one first-time
    attempt at the fit's board sizes."""
    att = load_first_time_attempts()
    att = att[att["size"].isin(SIZES)]
    require(len(att), "no first-time attempts at the fitted board sizes")
    return set(_key(att["size"], att["problem"]).unique())


def matched_agreement(a, b, keys):
    """Per-axis |correlation| after matching the two arms' axes one-to-one.

    Returned in matched order, longest-total-|correlation| assignment first;
    the caller summarises. Axis INDEX is meaningless across an untaught pair, so
    the matching is what makes the comparison well-posed at all."""
    A = a.loc[keys, EMBED_COLS].to_numpy(float)
    B = b.loc[keys, EMBED_COLS].to_numpy(float)
    for name, M in (("A", A), ("B", B)):
        require(np.all(np.isfinite(M)),
                f"arm {name}'s loadings carry a non-finite value over the "
                f"scored population")
        require(np.all(M.std(0) > 0),
                f"arm {name} has a constant axis over the scored population — "
                f"a collapsed axis has no direction to agree with")
    Az = (A - A.mean(0)) / A.std(0)
    Bz = (B - B.mean(0)) / B.std(0)
    corr = np.abs(Az.T @ Bz) / len(keys)
    require(np.all(np.isfinite(corr)), "the correlation matrix is not finite")
    rows, cols = linear_sum_assignment(-corr)
    return corr[rows, cols]


# --- the measurement -------------------------------------------------------

def measure():
    arms, seeds = load_panel()
    load = {arm: load_loadings(arm) for arm in ARMS}

    # ONE population for every pair: attempted, and present in all five arms.
    # Scoring different pairs on different problems would make the taught and
    # untaught numbers incomparable, which is the whole comparison.
    common = set(attempted_keys())
    for df in load.values():
        common &= set(df.index)
    keys = sorted(common)
    require(len(keys) > 1000,
            f"only {len(keys)} attempted problems are shared by every arm — "
            f"too few for a direction correlation to mean anything")

    per_pair = {(a, b): matched_agreement(load[a], load[b], keys)
                for a, b in (TAUGHT_PAIR,) + UNTAUGHT_PAIRS}
    taught = per_pair[TAUGHT_PAIR]
    untaught = [per_pair[p] for p in UNTAUGHT_PAIRS]

    m = {
        "seeds": seeds,
        "keys": len(keys),
        "per_pair": per_pair,
        "taught_mean": float(taught.mean()),
        "taught_min": float(taught.min()),
        "untaught_mean": float(np.mean([p.mean() for p in untaught])),
        "untaught_min": float(np.mean([p.min() for p in untaught])),
        "gpu": arms["taught_s0"]["pins"]["gpu"],
    }
    m["gap"] = m["taught_mean"] - m["untaught_mean"]
    require(m["gap"] > 0,
            f"the taught pair agrees no better across seeds than the untaught "
            f"pairs ({m['taught_mean']:.3f} vs {m['untaught_mean']:.3f}) — the "
            f"claim that the constant pull is what fixes the basis is refuted, "
            f"not just its magnitude")
    return m


def figures():
    m = measure()
    return [
        Figure("seeds.untaught_cross_seed_mean", m["untaught_mean"], ".2f",
               "cross-seed direction agreement with the teaching pull off: axes "
               f"matched one-to-one between two fits by maximum total "
               f"|correlation| of their per-problem loadings, each matched "
               f"pair's |correlation| meaned over the {K_AXES} axes and then "
               f"over the three untaught seed pairs, on attempted problems"),
        Figure("seeds.untaught_cross_seed_min", m["untaught_min"], ".2f",
               "the same agreement at each untaught pair's weakest-matched "
               "axis, averaged over the three pairs"),
        Figure("seeds.taught_cross_seed_mean", m["taught_mean"], ".2f",
               "the same statistic for two fits of the SHIPPED configuration "
               "differing only in the seed — the teaching pull on"),
        Figure("seeds.taught_cross_seed_min", m["taught_min"], ".2f",
               "the taught pair's weakest-matched axis"),
        Figure("seeds.teaching_gap", m["gap"], ".2f",
               "taught minus untaught cross-seed agreement, on the means — what "
               "the constant pull toward the rule columns buys in basis "
               "stability, the two measured on the same problems"),
        Figure("seeds.axes_matched", K_AXES, "d",
               "axes matched per pair — K as shipped"),
        Figure("seeds.problems_scored", m["keys"], "~1000",
               "problems every agreement above is measured over: attempted at "
               "least once (first-time attempts) and carried by every arm"),
    ]


def main():
    m = measure()
    print(f"seeds panel  (all arms full budget, one machine: {m['gpu']})")
    print(f"  agreement = |correlation| of loadings over {m['keys']} attempted "
          f"problems shared by every arm,")
    print(f"              after matching the {K_AXES} axes one-to-one by "
          f"maximum total |correlation|")
    print()
    print(f"{'pair':>28}{'seeds':>10}{'mean':>8}{'min':>8}   per-axis, matched")
    for (a, b), vals in m["per_pair"].items():
        axes = " ".join(f"{v:.2f}" for v in sorted(vals, reverse=True))
        label = f"{a} vs {b}"
        pair_seeds = f"{m['seeds'][a]}/{m['seeds'][b]}"
        print(f"{label:>28}{pair_seeds:>10}"
              f"{vals.mean():>8.2f}{vals.min():>8.2f}   {axes}")
    print()
    print(f"  untaught (3 pairs averaged)   mean {m['untaught_mean']:.2f}   "
          f"min {m['untaught_min']:.2f}")
    print(f"  taught   (shipped config)     mean {m['taught_mean']:.2f}   "
          f"min {m['taught_min']:.2f}")
    print(f"  the pull is worth {m['gap']:+.2f} in mean cross-seed agreement")
    print()
    print("  no held-out delta is quotable from this panel: the arms run at "
          "different seeds,")
    print("  so they hold aside different stop rows and the comparison would "
          "be unpaired.")


if __name__ == "__main__":
    main()
