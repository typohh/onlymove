#!/usr/bin/env python3
"""How well behaved the free axis space is (methodology §1.14), read from the
`axis_space` refit panel: fits of one configuration that differ ONLY in the
network's initialization and minibatch order, with the held-aside draw and the
feature basis pinned, compared on their axes — with the teaching pull off, with
it on, and each of those with and without an opening learning-rate ramp.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` (`ablations.run_panel`, locally or via
`vast_orchestration.py ablate`) and hard-fails when the panel is missing, short
of budget, internally inconsistent, or STALE against the current attempt log.

WHY THE PANEL EXISTS. The likelihood reads the loadings and the user positions
only through their product, so rotating the axis space leaves every predicted
probability unchanged. §1.14 claims more than a free choice of basis: that two
fits of one configuration can hold different SUBSPACES, and that even their
best-matched axis shares few of its extreme problems. Those readings were taken
under a constant learning rate, whose endpoint is one draw from a wandering
band; the decay that ships freezes a fit where it stands. Whether the space is
underdetermined, or was merely noisy, is what this panel decides — and whether
a gentler opening (the ramp) makes two free fits land closer, which is the
obvious remedy if the endpoint is decided by the early path.

  axis_space.<cfg>_basis_mean        per-axis |correlation| of the loadings
  axis_space.<cfg>_basis_min         after one-to-one matching: the mean and
                                     the weakest of the K matched axes,
                                     averaged over the six seed pairs.
  axis_space.<cfg>_subspace_lead     the strongest and the weakest canonical
  axis_space.<cfg>_subspace_min      correlation between the two fits' K-dim
                                     spans, averaged over the pairs. A basis
                                     that rotates inside a shared subspace
                                     reads low on the first pair and high here;
                                     a subspace that moves reads low on both.
  axis_space.<cfg>_tail_shared_best  shared fraction of the TAIL_N most extreme
  axis_space.<cfg>_tail_shared_mean  problems at each end of a matched axis,
                                     sign-aligned, both ends averaged: on the
                                     best-matched axis, and over all K.
  axis_space.<cfg>_level_sd_mnat     run-to-run sd of the held-out-problem
                                     level across the seeds, at one pinned draw.
  axis_space.<cfg>_effective_dim     how many DISTINCT directions a fit's K axes
  axis_space.<cfg>_max_within_r      span: the participation ratio of its axis
                                     correlation matrix's eigenvalues, and the
                                     most correlated pair of its own axes,
                                     averaged over the seeds. A fit whose axes
                                     duplicate each other reads low on the
                                     first and near one on the second.
  axis_space.warm_*                  the ramp against the free fit: what it
                                     buys in basis and subspace agreement, and
                                     its PAIRED held-out cost per seed.
  axis_space.taught_warm_*           the same on the shipped, taught fit.
  axis_space.teaching_*              the pull's paired cost, and what it buys.
  axis_space.same_init_basis_mean    free fits at the SAME seed under the two
                                     schedules — whether the initialization
                                     alone decides the basis.

<cfg> is one of `untaught` (teaching pull off), `untaught_warm`, `taught`
(the shipped configuration), `taught_warm`.

THE AGREEMENT STATISTICS, each stated because every part is a choice:

  1. Each arm's per-problem loadings are read from its own
     `axes/per_problem.csv`, joined across arms by (size, tsumego_id) — never
     positionally — and scored on ATTEMPTED problems only (at least one
     first-time attempt at the fitted sizes), because the never-attempted half
     of the corpus is unconstrained extrapolation whose extremes churn between
     fits, and counting that churn would report free extrapolation as a failure
     to reproduce. One population, intersected with every arm's rows, so every
     pair is scored over the same problems.
  2. BASIS: the two arms' K axes are matched one-to-one by maximum total
     |correlation| (Hungarian), since with the pull off the axis index carries
     no meaning; each matched pair's agreement is the absolute correlation, the
     sign being absorbed by the user positions.
  3. SUBSPACE: the canonical correlations between the two K-column loading
     matrices, standardized per column — the cosines between the two spans'
     principal directions, invariant to any rotation inside either span. This
     is what separates a basis rotating inside a fixed subspace (all K near one)
     from the subspace itself moving (the weakest near zero). The weakest one
     is degenerate for a fit whose axes DUPLICATE each other — a span of fewer
     than K distinct directions has trailing canonical correlations near zero
     whatever the other fit holds — so it is read beside the redundancy
     figures (item 4), and the leading ones carry the subspace claim.
  4. REDUNDANCY, within one fit: the eigenvalues of the K x K correlation
     matrix of its own axes over the population. Their participation ratio
     (sum^2 / sum of squares) is the effective number of distinct directions
     — K when the axes are uncorrelated, one when they all coincide — and the
     most correlated pair says how close the worst duplication is. Nothing in
     the shipped objective asks the axes to differ from each other: the ridge
     on positions is isotropic and each axis is normalized to unit scale, so
     splitting one direction over two axes halves the ridge cost at identical
     predictions, and only the teaching pull, holding each axis to its own
     rule, keeps them apart.
  5. TAILS: for each matched, sign-aligned axis pair, the TAIL_N problems with
     the largest loadings in each fit and the TAIL_N with the smallest; the
     shared fraction at each end, averaged. A name is a claim about an axis's
     extremes, so this is the statistic a name's survival actually rests on.
  6. Per pair each statistic gives K numbers (or K canonical correlations); the
     pair's summary is a mean and a minimum, and the configuration's figure is
     those averaged over its six seed pairs. The pairs are not independent (four
     fits, six pairings), so the average is a summary and not a standard error.

THE LEVELS ARE PAIRED, and that is what pinning the draw buys: every arm holds
aside the same problems, so the ramp's cost at a seed is the difference between
the two arms at that seed — same initialization, same held-out problems, the
schedule the only thing moving — and the same for teaching. A run's level is
the mean over its last 2% of checkpoints, the reading the schedule panel settled
on because the final epoch is what ships.

    ./.venv/bin/python3 -m ablations.refit_axis_space_figures
"""

import itertools
import json
import os
import sqlite3
import statistics
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require, panel_regime  # noqa: E402
# The terminal-window level, shared with the schedule panel rather than restated:
# both read "where a run ended" the same way, so a change there moves both.
from ablations.refit_schedule_figures import level  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.features import load_first_time_attempts  # noqa: E402
from utils.rating.model_constants import K_AXES, SIZES  # noqa: E402

TIER = "refit"

PANEL = "axis_space"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")
SEEDS = (0, 1, 2, 3)
# Configuration -> arm name pattern. The order is the printout's.
CONFIGS = {
    "untaught": "untaught_i{}",
    "untaught_warm": "untaught_warm_i{}",
    "untaught_ortho": "untaught_ortho_i{}",
    "taught": "taught_i{}",
    "taught_warm": "taught_warm_i{}",
}
WARM_OF = {"untaught": "untaught_warm", "taught": "taught_warm"}
ARMS = tuple(pat.format(s) for pat in CONFIGS.values() for s in SEEDS)
PAIRS = tuple(itertools.combinations(SEEDS, 2))
# Extreme problems per axis end. A thousand is the count §1.14 states its tail
# claims on, and the tail-overlap statistic is quoted at that count.
TAIL_N = 1000
EMBED_COLS = [f"embed_{k}" for k in range(K_AXES)]


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
                f"no {path} — the axis_space panel has not run. Run it with "
                f"`python -m ablations.run_panel axis_space` (or "
                f"`vast_orchestration.py ablate axis_space`) first.")
        arms[arm] = json.load(open(path))

    # ONE PANEL IS ONE REGIME, read off the arms and not off the trainer's
    # constants: the arms are only ever compared to each other.
    budget, schedule = panel_regime(arms)
    for arm, r in arms.items():
        require(r["metrics"]["trace"]
                and r["metrics"]["trace"][-1]["epoch"] == budget,
                f"{arm}'s convergence trace does not reach the panel's "
                f"{budget}-epoch budget")
        require(r["config"] == manifest["arms"][arm],
                f"{arm} was run with {r['config']!r}, but the manifest declares "
                f"{manifest['arms'][arm]!r} — the record is not this arm")
        require(r["config"].get("stop_unit") == "problem",
                f"{arm} holds aside rows, not problems — the level here is a "
                f"generalization reading and needs whole problems held out")

    base = arms[ARMS[0]]["pins"]
    for arm, r in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha",
                    "blocks", "pca_dim", "k_axes"):
            require(r["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} ({r['pins'][pin]!r}) differs from "
                    f"{ARMS[0]}'s ({base[pin]!r}) — the arms are not one panel "
                    f"and their comparison is void")
    require(int(base["k_axes"]) == int(K_AXES),
            f"the panel was fitted at K={base['k_axes']} but the shipping "
            f"constant is K_AXES={K_AXES} — this measurement is quoted as being "
            f"at the shipped K, so it has to have been")

    # EVERYTHING BUT THE INITIALIZATION IS PINNED, and it is checked on the
    # RECORDED values rather than trusted to the manifest: the draw, the feature
    # basis and the row count every arm scored on.
    for field in ("split_seed", "pca_seed"):
        vals = {r["metrics"].get(field) for r in arms.values()}
        require(len(vals) == 1 and None not in vals,
                f"the arms record {len(vals)} different {field} values "
                f"({vals}) — this panel's design is one pinned draw and one "
                f"pinned feature basis, and without them a between-arm "
                f"difference is not the initialization moving")
    rows = {(r["metrics"]["stop_rows"], r["metrics"]["rows"])
            for r in arms.values()}
    require(len(rows) == 1,
            f"the arms held aside different row counts ({rows}) at one pinned "
            f"draw — they are not scored on the same problems")

    # The seed is what varies, per configuration, and the ramp is what the
    # warm configurations add — both read off the records.
    warm = {}
    for cfg, pat in CONFIGS.items():
        seeds = [int(arms[pat.format(s)]["config"].get("seed", 0))
                 for s in SEEDS]
        require(seeds == list(SEEDS),
                f"{cfg}'s arms ran seeds {seeds}, expected {list(SEEDS)}")
        ramps = {int(arms[pat.format(s)]["metrics"].get("lr_warmup_epochs", 0))
                 for s in SEEDS}
        require(len(ramps) == 1,
                f"{cfg}'s arms ran different ramps {sorted(ramps)}")
        warm[cfg] = ramps.pop()
    for cfg in ("untaught", "untaught_ortho", "taught"):
        require(warm[cfg] == 0,
                f"{cfg} ran a {warm[cfg]}-epoch ramp — the shipped schedule "
                f"has none")
    # The mutual penalty is carried by exactly one configuration, read off the
    # records: a shipped-path arm that recorded it would be a different fit.
    for cfg, pat in CONFIGS.items():
        ws = {float(arms[pat.format(s)]["metrics"].get("ortho_axes", 0.0))
              for s in SEEDS}
        require(len(ws) == 1, f"{cfg}'s arms ran different mutual penalties {ws}")
        w = ws.pop()
        require((w > 0) == (cfg == "untaught_ortho"),
                f"{cfg} recorded a mutual-orthogonality weight of {w}")
        if cfg == "untaught_ortho":
            ortho_w = w
    for cfg, warm_cfg in WARM_OF.items():
        require(warm[warm_cfg] > 0,
                f"{warm_cfg} ran no ramp — it is {cfg}'s ramp arm")
    require(warm["untaught_warm"] == warm["taught_warm"],
            f"the two ramp configurations ran different ramps "
            f"({warm['untaught_warm']} and {warm['taught_warm']} epochs)")

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
    return arms, budget, schedule, warm["untaught_warm"], ortho_w


# --- the loadings ----------------------------------------------------------

def _key(size, tid):
    return size.astype(str) + ":" + tid.astype(str)


def load_loadings(arm):
    """One arm's per-problem axis loadings, indexed by "<size>:<tsumego_id>".

    The id is read as TEXT: it is eight hex characters and an all-digit one
    would otherwise arrive as an integer with its leading zeros gone."""
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


def standardized(df, keys, name):
    M = df.loc[keys, EMBED_COLS].to_numpy(float)
    require(np.all(np.isfinite(M)),
            f"arm {name}'s loadings carry a non-finite value over the scored "
            f"population")
    require(np.all(M.std(0) > 0),
            f"arm {name} has a constant axis over the scored population — a "
            f"collapsed axis has no direction to agree with")
    return (M - M.mean(0)) / M.std(0)


def pair_stats(A, B):
    """The three agreement statistics for one pair of standardized loading
    matrices (n x K each), as a dict of per-axis arrays in matched order,
    best-matched axis first."""
    n = len(A)
    corr = A.T @ B / n
    require(np.all(np.isfinite(corr)), "the correlation matrix is not finite")
    rows, cols = linear_sum_assignment(-np.abs(corr))
    matched = np.abs(corr[rows, cols])
    order = np.argsort(-matched)

    # Canonical correlations between the two spans: the singular values of
    # Q_A^T Q_B for orthonormal bases of each. Columns are centred already.
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    canon = np.linalg.svd(qa.T @ qb, compute_uv=False)
    require(len(canon) == K_AXES and np.all(np.isfinite(canon)),
            "the canonical correlations are malformed")

    shared = []
    for i, j in zip(rows, cols):
        x = A[:, i]
        y = np.sign(corr[i, j]) * B[:, j]
        hi = len(set(np.argsort(-x)[:TAIL_N]) & set(np.argsort(-y)[:TAIL_N]))
        lo = len(set(np.argsort(x)[:TAIL_N]) & set(np.argsort(y)[:TAIL_N]))
        shared.append((hi + lo) / (2.0 * TAIL_N))
    shared = np.array(shared)
    return {"basis": matched[order], "subspace": np.sort(canon)[::-1],
            "tail": shared[order]}


def redundancy(Z):
    """(effective dimension, max pairwise |r|) of one fit's standardized axes."""
    C = Z.T @ Z / len(Z)
    ev = np.linalg.eigvalsh(C)
    ev = np.clip(ev, 0.0, None)
    require(ev.sum() > 0, "the axis correlation matrix is degenerate")
    eff = float(ev.sum() ** 2 / (ev ** 2).sum())
    off = np.abs(C - np.eye(len(C)))
    return eff, float(off.max())


# --- the measurement -------------------------------------------------------

def measure():
    arms, budget, schedule, ramp, ortho_w = load_panel()
    load = {arm: load_loadings(arm) for arm in ARMS}

    # ONE population for every pair: attempted, and present in every arm.
    common = set(attempted_keys())
    for df in load.values():
        common &= set(df.index)
    keys = sorted(common)
    require(len(keys) > 1000,
            f"only {len(keys)} attempted problems are shared by every arm — "
            f"too few for a direction correlation to mean anything")
    Z = {arm: standardized(load[arm], keys, arm) for arm in ARMS}

    per_cfg = {}
    for cfg, pat in CONFIGS.items():
        pairs = {(a, b): pair_stats(Z[pat.format(a)], Z[pat.format(b)])
                 for a, b in PAIRS}
        levels = {s: level(arms[pat.format(s)]["metrics"]) for s in SEEDS}
        work = {round(arms[pat.format(s)]["metrics"]["lr_integral_epochs"])
                for s in SEEDS}
        require(len(work) == 1,
                f"{cfg}'s arms delivered different rate integrals {sorted(work)}")
        red = [redundancy(Z[pat.format(s)]) for s in SEEDS]
        per_cfg[cfg] = {
            "pairs": pairs,
            "effective_dim": float(np.mean([e for e, _ in red])),
            "max_within_r": float(np.mean([r for _, r in red])),
            "redundancy_per_seed": red,
            "basis_mean": float(np.mean([p["basis"].mean() for p in pairs.values()])),
            "basis_min": float(np.mean([p["basis"].min() for p in pairs.values()])),
            "subspace_lead": float(np.mean([p["subspace"].max() for p in pairs.values()])),
            "subspace_min": float(np.mean([p["subspace"].min() for p in pairs.values()])),
            "subspace_third": float(np.mean([p["subspace"][2] for p in pairs.values()])),
            "tail_best": float(np.mean([p["tail"][0] for p in pairs.values()])),
            "tail_mean": float(np.mean([p["tail"].mean() for p in pairs.values()])),
            "levels": levels,
            "level": statistics.fmean(levels.values()),
            "level_sd_mnat": 1e3 * statistics.stdev(levels.values()),
            "work": work.pop(),
            "elapsed_s": statistics.median(
                arms[pat.format(s)]["metrics"]["elapsed_s"] for s in SEEDS),
        }

    # PAIRED CONTRASTS, one per seed: the two arms share the initialization and
    # the held-out problems, so their difference is the schedule (or the pull)
    # and nothing else. Positive means the second-named arm predicts worse.
    def paired(cfg_a, cfg_b):
        deltas = [1e3 * (per_cfg[cfg_b]["levels"][s] - per_cfg[cfg_a]["levels"][s])
                  for s in SEEDS]
        return {"mean": statistics.fmean(deltas), "per_seed": deltas,
                "worse": sum(d > 0 for d in deltas)}

    # THE RATING SIDE ALONE: the same paired contrasts with the axis term
    # zeroed at the final state (`stop_bce_no_axes`), which says whether a
    # cost is paid by the rating heads — the trunk is shared, so a pull on the
    # loadings can move what the rating generalizes — or by the axis channel.
    def paired_no_axes(cfg_a, cfg_b):
        deltas = [1e3 * (arms[CONFIGS[cfg_b].format(s)]["metrics"]["stop_bce_no_axes"]
                         - arms[CONFIGS[cfg_a].format(s)]["metrics"]["stop_bce_no_axes"])
                  for s in SEEDS]
        return {"mean": statistics.fmean(deltas), "per_seed": deltas,
                "worse": sum(d > 0 for d in deltas)}

    contrasts = {
        "teaching_rating_side": paired_no_axes("untaught", "taught"),
        "warm": paired("untaught", "untaught_warm"),
        "ortho": paired("untaught", "untaught_ortho"),
        "taught_warm": paired("taught", "taught_warm"),
        "teaching": paired("untaught", "taught"),
    }

    # SAME INITIALIZATION, TWO SCHEDULES: if the basis is decided by where the
    # fit starts, a free fit under the ramp should land near the free fit at
    # the same seed; if it is decided by the path, it should not.
    same_init = {cfg: [pair_stats(Z[CONFIGS[cfg].format(s)],
                                  Z[CONFIGS[WARM_OF[cfg]].format(s)])
                       for s in SEEDS]
                 for cfg in WARM_OF}
    same_init_mean = {cfg: float(np.mean([p["basis"].mean() for p in ps]))
                      for cfg, ps in same_init.items()}

    return {
        "cfg": per_cfg, "contrasts": contrasts,
        "same_init": same_init, "same_init_mean": same_init_mean,
        "keys": len(keys), "budget": budget, "schedule": schedule,
        "ramp": ramp, "ortho_w": ortho_w, "gpu": arms[ARMS[0]]["pins"]["gpu"],
        "split_seed": arms[ARMS[0]]["metrics"]["split_seed"],
        "pca_seed": arms[ARMS[0]]["metrics"]["pca_seed"],
        "stop_rows": arms[ARMS[0]]["metrics"]["stop_rows"],
    }


CFG_NOUN = {
    "untaught": "the free fit (teaching pull off)",
    "untaught_warm": "the free fit under the opening ramp",
    "untaught_ortho": "the free fit under a mutual-orthogonality penalty",
    "taught": "the shipped, taught fit",
    "taught_warm": "the taught fit under the opening ramp",
}


def figures():
    m = measure()
    figs = [
        Figure("axis_space.seeds", len(SEEDS), "d",
               "fits per configuration, differing only in the network's "
               "initialization and minibatch order — the held-aside draw and "
               "the joint-PCA feature basis are pinned across every arm"),
        Figure("axis_space.pairs", len(PAIRS), "d",
               "seed pairs each agreement figure is averaged over"),
        Figure("axis_space.problems_scored", m["keys"], "~1000",
               "problems every agreement figure is measured over: attempted at "
               "least once (first-time attempts) and carried by every arm"),
        Figure("axis_space.tail_n", TAIL_N, "d",
               "extreme problems per axis end the tail-overlap figures count"),
        Figure("axis_space.ramp_epochs", m["ramp"], "d",
               "opening epochs over which the ramp arms scale both rates "
               "linearly up to the schedule, out of the panel's budget"),
        Figure("axis_space.ramp_work_epochs",
               m["cfg"]["untaught"]["work"] - m["cfg"]["untaught_warm"]["work"],
               "d",
               "constant-rate epochs of work the ramp removes from the budget"),
    ]
    for cfg, c in m["cfg"].items():
        noun = CFG_NOUN[cfg]
        figs += [
            Figure(f"axis_space.{cfg}_basis_mean", c["basis_mean"], ".2f",
                   f"{noun}: |correlation| of per-problem loadings between two "
                   f"fits after matching the {K_AXES} axes one-to-one, meaned "
                   f"over the axes and over the seed pairs"),
            Figure(f"axis_space.{cfg}_basis_min", c["basis_min"], ".2f",
                   f"{noun}: the same at each pair's weakest-matched axis"),
            Figure(f"axis_space.{cfg}_subspace_lead", c["subspace_lead"], ".2f",
                   f"{noun}: the strongest canonical correlation between the "
                   f"two fits' {K_AXES}-dimensional spans, averaged over pairs"),
            Figure(f"axis_space.{cfg}_subspace_third", c["subspace_third"], ".2f",
                   f"{noun}: the third-strongest canonical correlation, "
                   f"averaged over pairs — the depth at which two spans still "
                   f"share a direction, read where a free fit's effective "
                   f"dimension sits"),
            Figure(f"axis_space.{cfg}_subspace_min", c["subspace_min"], ".2f",
                   f"{noun}: the weakest canonical correlation — near zero "
                   f"means one fit holds a direction the other's span lacks, "
                   f"or that the span holds fewer than {K_AXES} distinct "
                   f"directions at all (see the effective dimension)"),
            Figure(f"axis_space.{cfg}_effective_dim", c["effective_dim"], ".1f",
                   f"{noun}: effective number of distinct directions among a "
                   f"fit's {K_AXES} axes — the participation ratio of its axis "
                   f"correlation matrix's eigenvalues over attempted problems, "
                   f"averaged over the seeds"),
            Figure(f"axis_space.{cfg}_max_within_r", c["max_within_r"], ".2f",
                   f"{noun}: |correlation| of the most correlated pair of a "
                   f"fit's own axes, averaged over the seeds — near one is a "
                   f"duplicated axis"),
            Figure(f"axis_space.{cfg}_tail_shared_best", c["tail_best"], "%.0f",
                   f"{noun}: percent of the {TAIL_N} most extreme problems at "
                   f"each end shared by the two fits on their BEST-matched "
                   f"axis, both ends averaged, then over pairs"),
            Figure(f"axis_space.{cfg}_tail_shared_mean", c["tail_mean"], "%.0f",
                   f"{noun}: the same over all {K_AXES} matched axes"),
            Figure(f"axis_space.{cfg}_level_sd_mnat", c["level_sd_mnat"], ".1f",
                   f"{noun}: run-to-run sd of the held-out-problem level (mean "
                   f"of the last 2% of checkpoints) across the seeds, "
                   f"millinats, at one pinned draw"),
        ]
    u, w, o, t, tw = (m["cfg"][k] for k in ("untaught", "untaught_warm",
                                            "untaught_ortho",
                                            "taught", "taught_warm"))
    ct = m["contrasts"]
    figs += [
        Figure("axis_space.warm_basis_gain", w["basis_mean"] - u["basis_mean"],
               "+.2f",
               "free fit: mean matched-axis agreement under the ramp minus "
               "without it — what a gentler opening buys in basis stability"),
        Figure("axis_space.warm_subspace_gain",
               w["subspace_min"] - u["subspace_min"], "+.2f",
               "free fit: weakest canonical correlation under the ramp minus "
               "without it"),
        Figure("axis_space.warm_cost_mnat", ct["warm"]["mean"], "+.1f",
               "free fit: held-out-problem level under the ramp minus without "
               "it, PAIRED at each seed (same initialization, same held-out "
               "problems) and averaged; positive means the ramp predicts worse"),
        Figure("axis_space.warm_cost_seeds_worse", ct["warm"]["worse"], "d",
               f"of the {len(SEEDS)} seeds, how many put the ramp's free fit "
               f"on the worse side"),
        Figure("axis_space.ortho_weight", m["ortho_w"], ".2f",
               "weight of the mutual-orthogonality penalty the ortho arms "
               "carry (the squared off-diagonal of the loadings' batch "
               "correlation matrix); production carries none"),
        Figure("axis_space.ortho_effective_dim_gain",
               o["effective_dim"] - u["effective_dim"], "+.1f",
               "free fit: effective number of distinct directions under the "
               "mutual penalty minus without it"),
        Figure("axis_space.ortho_basis_gain", o["basis_mean"] - u["basis_mean"],
               "+.2f",
               "free fit: mean matched-axis agreement under the mutual penalty "
               "minus without it — whether seven DISTINCT free directions "
               "reproduce across seeds any better than duplicated ones"),
        Figure("axis_space.ortho_subspace_gain",
               o["subspace_third"] - u["subspace_third"], "+.2f",
               "free fit: third canonical correlation under the mutual "
               "penalty minus without it"),
        Figure("axis_space.ortho_cost_mnat", ct["ortho"]["mean"], "+.1f",
               "free fit: held-out-problem level under the mutual penalty "
               "minus without it, paired per seed and averaged; positive "
               "means the penalty predicts worse"),
        Figure("axis_space.ortho_cost_seeds_worse", ct["ortho"]["worse"], "d",
               f"of the {len(SEEDS)} seeds, how many put the mutual-penalty "
               f"free fit on the worse side"),
        Figure("axis_space.taught_warm_basis_gain",
               tw["basis_mean"] - t["basis_mean"], "+.2f",
               "taught fit: mean matched-axis agreement under the ramp minus "
               "without it"),
        Figure("axis_space.taught_warm_cost_mnat", ct["taught_warm"]["mean"],
               "+.1f",
               "taught fit: held-out-problem level under the ramp minus "
               "without it, paired per seed and averaged"),
        Figure("axis_space.taught_warm_cost_seeds_worse",
               ct["taught_warm"]["worse"], "d",
               f"of the {len(SEEDS)} seeds, how many put the ramp's taught fit "
               f"on the worse side"),
        Figure("axis_space.teaching_gap", t["basis_mean"] - u["basis_mean"],
               "+.2f",
               "taught minus free mean matched-axis agreement, at pinned draw "
               "and feature basis — what the constant pull buys in basis "
               "stability against initialization alone"),
        Figure("axis_space.teaching_effective_dim_gap",
               t["effective_dim"] - u["effective_dim"], "+.1f",
               "taught minus free effective dimension — how many distinct "
               "directions the pull recovers from a fit that would otherwise "
               "spend its axes on duplicates"),
        Figure("axis_space.teaching_subspace_gap",
               t["subspace_min"] - u["subspace_min"], "+.2f",
               "taught minus free weakest canonical correlation"),
        Figure("axis_space.teaching_cost_mnat", ct["teaching"]["mean"], "+.1f",
               "held-out-problem level of the taught fit minus the free fit, "
               "paired per seed and averaged; positive means teaching "
               "predicts worse"),
        Figure("axis_space.teaching_cost_seeds_worse", ct["teaching"]["worse"],
               "d",
               f"of the {len(SEEDS)} seeds, how many put the taught fit on the "
               f"worse side"),
        Figure("axis_space.teaching_rating_side_cost_mnat",
               ct["teaching_rating_side"]["mean"], "+.1f",
               "the same paired teaching cost with the axis term zeroed at the "
               "final state — what the pull costs the RATING heads' "
               "generalization to unseen problems, millinats"),
        Figure("axis_space.teaching_rating_side_seeds_worse",
               ct["teaching_rating_side"]["worse"], "d",
               f"of the {len(SEEDS)} seeds, how many put the taught fit's "
               f"rating side on the worse side"),
        Figure("axis_space.same_init_basis_mean",
               m["same_init_mean"]["untaught"], ".2f",
               "free fits at the SAME seed under the two schedules: mean "
               "matched-axis agreement — high means the initialization decides "
               "the basis, low means the path does"),
    ]
    return figs


def main():
    m = measure()
    print(f"axis_space panel  (all arms full budget: {m['budget']} epochs, "
          f"{m['schedule']} rate; ramp arms open with a {m['ramp']}-epoch "
          f"ramp; ortho arms carry a mutual penalty at {m['ortho_w']}; "
          f"one machine: {m['gpu']})")
    print(f"  pinned across every arm: held-aside draw (split_seed "
          f"{m['split_seed']}, {m['stop_rows']} rows of held-out problems) and "
          f"feature basis (pca_seed {m['pca_seed']}); the seed moves the "
          f"initialization and batch order only")
    print(f"  agreement over {m['keys']} attempted problems shared by every "
          f"arm; axes matched one-to-one by maximum total |correlation|; "
          f"tails are the {TAIL_N} most extreme problems per end")
    for cfg, c in m["cfg"].items():
        print(f"\n== {cfg}: {CFG_NOUN[cfg]}  (work {c['work']} constant-rate "
              f"epochs, median {c['elapsed_s']:.0f}s)")
        print(f"{'pair':>8}{'basis mean':>12}{'basis min':>11}"
              f"{'cca lead':>10}{'cca min':>9}{'tail best':>11}{'tail mean':>11}"
              f"   per-axis basis (matched) | canonical")
        for (a, b), p in c["pairs"].items():
            basis = " ".join(f"{v:.2f}" for v in p["basis"])
            canon = " ".join(f"{v:.2f}" for v in p["subspace"])
            print(f"{f'{a}/{b}':>8}{p['basis'].mean():>12.2f}"
                  f"{p['basis'].min():>11.2f}{p['subspace'].max():>10.2f}"
                  f"{p['subspace'].min():>9.2f}{p['tail'][0]:>11.2f}"
                  f"{p['tail'].mean():>11.2f}   {basis} | {canon}")
        print(f"{'mean':>8}{c['basis_mean']:>12.2f}{c['basis_min']:>11.2f}"
              f"{c['subspace_lead']:>10.2f}{c['subspace_min']:>9.2f}"
              f"{c['tail_best']:>11.2f}{c['tail_mean']:>11.2f}")
        red = " ".join(f"s{s}={e:.1f}/{r:.2f}"
                       for s, (e, r) in zip(SEEDS, c["redundancy_per_seed"]))
        print(f"  within-fit redundancy, effective dim / max |r| between own "
              f"axes: {red}  (mean {c['effective_dim']:.1f} / "
              f"{c['max_within_r']:.2f})")
        lv = " ".join(f"s{s}={v:.4f}" for s, v in c["levels"].items())
        print(f"  held-out-problem level: {lv}  (mean {c['level']:.4f}, "
              f"sd {c['level_sd_mnat']:.1f} mnat)")
    print("\n== paired contrasts per seed (positive = second arm predicts "
          "worse), millinats")
    for name, label in (("warm", "free fit: ramp minus none"),
                        ("ortho", "free fit: mutual penalty minus none"),
                        ("taught_warm", "taught fit: ramp minus none"),
                        ("teaching", "taught minus free"),
                        ("teaching_rating_side", "  same, axis term zeroed")):
        ct = m["contrasts"][name]
        per = " ".join(f"{d:+.2f}" for d in ct["per_seed"])
        print(f"  {label:<30} {per}   mean {ct['mean']:+.2f}  "
              f"worse on {ct['worse']}/{len(SEEDS)} seeds")
    print("\n== same initialization, two schedules (does the init decide the "
          "basis?)")
    for cfg, ps in m["same_init"].items():
        per = " ".join(f"s{s}={p['basis'].mean():.2f}" for s, p in zip(SEEDS, ps))
        print(f"  {cfg:<14} matched-axis mean {per}   "
              f"overall {m['same_init_mean'][cfg]:.2f}")


if __name__ == "__main__":
    main()
