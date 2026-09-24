#!/usr/bin/env python3
"""Which terms on the axis loadings need to reach the shared trunk (methodology
§1.5, §1.14), read from the `detach` refit panel: the panel's baseline — the
taught fit WITHOUT a mutual penalty, called `shipped` in its manifest and its
figure keys — against the same fit with one term's path into the trunk closed
(the teaching pull, the rating-orthogonality penalty, both), plus that fit
carrying a detached mutual penalty, and a free fit under an anisotropic ridge.
The mutual arm is what ships; the panel is defined against the fit without it,
so its `shipped` arms declare `{}` and cannot be re-run under a baseline that
carries the term without the manifest first pinning `ortho_axes 0`.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` and hard-fails when the panel is missing, short of
budget, internally inconsistent or stale against the attempt log.

WHAT IS MEASURED. Every eight-seed configuration is PAIRED against the shipped
arm at the same seed on held-out problems — same initialization, same held-out
draw, the term's path the only difference — with a t-interval, a paired t-test
and the count of seeds on the better side; the same gain with the axis term
zeroed is the rating's share. A detachment may not lose identity, so each
configuration is also read for agreement with the taught rule ends (the
trainer's own identity report), cross-seed matched-axis agreement, and within-
fit redundancy. The anisotropic-ridge free fit has no paired reading: it is
scored on the free fit's own question, whether its axes reproduce across seeds
and how many distinct directions they span.

  detach.<cfg>_gain_mnat            shipped level minus the arm's, paired per
  detach.<cfg>_gain_ci_mnat         seed on held-out problems; positive = the
  detach.<cfg>_gain_p               arm predicts better; its 95% half-width,
  detach.<cfg>_seeds_better         paired t p, and seeds on the better side
  detach.<cfg>_rating_gain_mnat     the same gain with the axis term zeroed
  detach.<cfg>_taught_agreement     mean agreement of each axis's taught end
                                    with its rule (shipped too)
  detach.<cfg>_basis_mean           cross-seed matched-axis agreement
  detach.<cfg>_effective_dim        distinct directions among the K axes
  detach.<cfg>_max_within_r         the fit's most correlated pair of axes
  detach.aniso_*                    basis_mean / basis_min / effective_dim /
                                    max_within_r / level_sd_mnat for the
                                    anisotropic free fit
  detach.seeds / detach.aniso_seeds / detach.mutual_weight / detach.aniso_ladder
  detach.level_sd_mnat              run-to-run sd pooled over the eight-seed
                                    configurations

    ./.venv/bin/python3 -m ablations.refit_detach_figures
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
from ablations.refit_axis_space_figures import (  # noqa: E402
    EMBED_COLS, attempted_keys, pair_stats, redundancy, standardized)
from ablations.refit_ramp_figures import paired_stats, taught_agreement  # noqa: E402
from ablations.refit_schedule_figures import level  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.model_constants import K_AXES  # noqa: E402

TIER = "refit"

PANEL = "detach"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")
SEEDS = tuple(range(8))
ANISO_SEEDS = tuple(range(4))
# Configuration -> arm pattern, and the knobs each one must carry, read off the
# RECORDS: a record run with a different flag is a different experiment.
CONFIGS = {
    "shipped": ("shipped_s{}", {}),
    "teach_detach": ("teach_detach_s{}", {"teach_detach": True}),
    "ortho_detach": ("ortho_detach_s{}", {"ortho_detach": True}),
    "both_detach": ("both_detach_s{}", {"teach_detach": True, "ortho_detach": True}),
    "mutual": ("mutual_s{}", {"ortho_axes_detach": True}),
}
ANISO = ("aniso_s{}", {"v_ridge_ladder": None})   # any ratio above one
CFG_NOUN = {
    "shipped": "the taught fit without a mutual penalty (the panel's baseline)",
    "teach_detach": "the teaching pull detached from the trunk",
    "ortho_detach": "the rating-orthogonality penalty detached from the trunk",
    "both_detach": "both detached",
    "mutual": "the taught fit plus a detached mutual-orthogonality penalty",
}


def load_panel():
    require(os.path.exists(MANIFEST), f"no {MANIFEST}")
    manifest = json.load(open(MANIFEST))
    want = {pat.format(s) for pat, _ in CONFIGS.values() for s in SEEDS}
    want |= {ANISO[0].format(s) for s in ANISO_SEEDS}
    require(set(manifest["arms"]) == want,
            f"{MANIFEST} declares {sorted(manifest['arms'])}, this producer "
            f"reads {sorted(want)}")
    arms = {}
    for arm, cfg in manifest["arms"].items():
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the detach panel has not finished. Run it with "
                f"`python -m ablations.run_panel detach` (or "
                f"`auxiliary/vast_orchestration.py ablate detach`).")
        rec = json.load(open(path))
        require(rec["config"] == cfg,
                f"{arm} ran {rec['config']!r}, the manifest declares {cfg!r}")
        require(cfg.get("stop_unit") == "problem", f"{arm} holds aside rows")
        arms[arm] = rec
    budget, schedule = panel_regime(arms, mutual_may_vary=True)
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
                f"the arms record {len(vals)} different {field} values")
    rows = {(r["metrics"]["stop_rows"], r["metrics"]["rows"]) for r in arms.values()}
    require(len(rows) == 1, f"the arms held aside different row counts {rows}")

    # Each configuration's knobs, on the records. Everything not named must be
    # off — a shipped arm carrying a detachment would be the wrong baseline.
    knobs = ("teach_detach", "ortho_detach", "ortho_axes_detach")
    for cfg, (pat, expect) in CONFIGS.items():
        for s in SEEDS:
            m = arms[pat.format(s)]["metrics"]
            for k in knobs:
                require(bool(m.get(k, False)) == bool(expect.get(k, False)),
                        f"{pat.format(s)} records {k}={m.get(k)}")
            require((m.get("ortho_axes", 0.0) > 0) == (cfg == "mutual"),
                    f"{pat.format(s)} records a mutual weight of "
                    f"{m.get('ortho_axes')} — this panel is defined against "
                    f"the taught fit WITHOUT the term; under a baseline that "
                    f"carries it, every non-mutual arm needs `ortho_axes 0` "
                    f"declared in the manifest")
            require(float(m.get("v_ridge_ladder", 1.0)) == 1.0,
                    f"{pat.format(s)} carries an anisotropic ridge")
            require(int(m.get("lr_warmup_epochs", 0)) == 0,
                    f"{pat.format(s)} ran a ramp")
    mutual_w = {arms[CONFIGS["mutual"][0].format(s)]["metrics"]["ortho_axes"]
                for s in SEEDS}
    require(len(mutual_w) == 1, f"the mutual arms ran different weights {mutual_w}")
    ladders = set()
    for s in ANISO_SEEDS:
        m = arms[ANISO[0].format(s)]["metrics"]
        require(float(m.get("v_ridge_ladder", 1.0)) > 1.0,
                f"{ANISO[0].format(s)} carries an isotropic ridge")
        require(float(m.get("teach_weight", 0.0)) == 0.0 or
                arms[ANISO[0].format(s)]["config"].get("teach_weight") == 0.0,
                f"{ANISO[0].format(s)} is not a free fit")
        ladders.add(float(m["v_ridge_ladder"]))
    require(len(ladders) == 1, f"the aniso arms ran different ladders {ladders}")

    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute("SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"measured at attempt-log frontier {base['db_max_id']}, the log now "
            f"ends at {now} — re-run the panel")
    return arms, budget, schedule, mutual_w.pop(), ladders.pop(), base["gpu"]


def load_loadings(arm):
    path = os.path.join(PANEL_DIR, arm, "axes", "per_problem.csv")
    require(os.path.exists(path), f"no {path}")
    df = pd.read_csv(path, dtype={"tsumego_id": str})
    require(list(df.columns) == ["tsumego_id", "size"] + EMBED_COLS,
            f"{path} has an unexpected layout")
    df = df.set_index(df["size"].astype(str) + ":" + df["tsumego_id"])
    require(not df.index.has_duplicates, f"{path} carries a duplicate key")
    return df


def measure():
    arms, budget, schedule, mutual_w, ladder, gpu = load_panel()
    names = {cfg: [pat.format(s) for s in SEEDS] for cfg, (pat, _) in CONFIGS.items()}
    names["aniso"] = [ANISO[0].format(s) for s in ANISO_SEEDS]
    loads = {a: load_loadings(a) for group in names.values() for a in group}
    common = set(attempted_keys())
    for df in loads.values():
        common &= set(df.index)
    keys = sorted(common)
    require(len(keys) > 1000, "too few attempted problems shared by every arm")
    Z = {a: standardized(df, keys, a) for a, df in loads.items()}

    def identity(group):
        vals = [pair_stats(Z[a], Z[b])["basis"] for a, b in itertools.combinations(group, 2)]
        red = [redundancy(Z[a]) for a in group]
        return {"basis_mean": float(np.mean([v.mean() for v in vals])),
                "basis_min": float(np.mean([v.min() for v in vals])),
                "effective_dim": float(np.mean([e for e, _ in red])),
                "max_within_r": float(np.mean([r for _, r in red]))}

    base_lv = {s: level(arms[names["shipped"][s]]["metrics"]) for s in SEEDS}
    base_na = {s: arms[names["shipped"][s]]["metrics"]["stop_bce_no_axes"] for s in SEEDS}
    per = {}
    for cfg in CONFIGS:
        group = names[cfg]
        lv = {s: level(arms[group[s]]["metrics"]) for s in SEEDS}
        c = {"levels": lv, "level": statistics.fmean(lv.values()),
             "sd_mnat": 1e3 * statistics.stdev(lv.values()),
             "taught_agreement": statistics.fmean(
                 taught_agreement(arms[a]["metrics"]) for a in group),
             "elapsed_s": statistics.median(arms[a]["metrics"]["elapsed_s"] for a in group)}
        c.update(identity(group))
        if cfg != "shipped":
            gains = [1e3 * (base_lv[s] - lv[s]) for s in SEEDS]
            rgains = [1e3 * (base_na[s] - arms[group[s]]["metrics"]["stop_bce_no_axes"])
                      for s in SEEDS]
            c["gains"] = gains
            c["gain"], c["gain_ci"], c["gain_p"], c["seeds_better"] = paired_stats(gains)
            c["rating_gain"] = paired_stats(rgains)[0]
        per[cfg] = c
    an_lv = [level(arms[a]["metrics"]) for a in names["aniso"]]
    aniso = identity(names["aniso"])
    aniso.update({"level": statistics.fmean(an_lv),
                  "level_sd_mnat": 1e3 * statistics.stdev(an_lv)})

    ss = sum((x - per[c]["level"]) ** 2 for c in CONFIGS for x in per[c]["levels"].values())
    pooled = 1e3 * math.sqrt(ss / (len(CONFIGS) * (len(SEEDS) - 1)))
    return {"per": per, "aniso": aniso, "budget": budget, "schedule": schedule,
            "mutual_w": mutual_w, "ladder": ladder, "gpu": gpu, "keys": len(keys),
            "pooled_sd": pooled}


def figures():
    m = measure()
    figs = [
        Figure("detach.seeds", len(SEEDS), "d",
               "fits per paired configuration, at one pinned draw and basis"),
        Figure("detach.aniso_seeds", len(ANISO_SEEDS), "d",
               "fits of the anisotropic-ridge free fit"),
        Figure("detach.mutual_weight", m["mutual_w"], ".2f",
               "weight of the detached mutual-orthogonality penalty the taught "
               "fit carries in the mutual arms"),
        Figure("detach.aniso_ladder", m["ladder"], ".2f",
               "ratio between consecutive axes' ridge weights in the "
               "anisotropic free fit"),
        Figure("detach.level_sd_mnat", m["pooled_sd"], ".1f",
               "run-to-run sd of the held-out-problem level, pooled over the "
               "eight-seed configurations, millinats"),
        Figure("detach.problems_scored", m["keys"], "~1000",
               "attempted problems the identity figures are measured over"),
    ]
    for cfg, c in m["per"].items():
        noun = CFG_NOUN[cfg]
        figs += [
            Figure(f"detach.{cfg}_taught_agreement", c["taught_agreement"], ".2f",
                   f"{noun}: mean agreement of each axis's taught end with its "
                   f"rule, over the seeds"),
            Figure(f"detach.{cfg}_basis_mean", c["basis_mean"], ".2f",
                   f"{noun}: cross-seed matched-axis |correlation|, averaged "
                   f"over all seed pairs"),
            Figure(f"detach.{cfg}_effective_dim", c["effective_dim"], ".1f",
                   f"{noun}: distinct directions among the {K_AXES} axes"),
            Figure(f"detach.{cfg}_max_within_r", c["max_within_r"], ".2f",
                   f"{noun}: the fit's most correlated pair of axes"),
        ]
        if cfg == "shipped":
            continue
        figs += [
            Figure(f"detach.{cfg}_gain_mnat", c["gain"], "+.1f",
                   f"{noun}: shipped level minus this arm's on held-out "
                   f"problems, paired per seed and averaged, millinats; "
                   f"positive = this arm predicts better"),
            Figure(f"detach.{cfg}_gain_ci_mnat", c["gain_ci"], ".1f",
                   "half-width of that gain's 95% t-interval, millinats"),
            Figure(f"detach.{cfg}_gain_p", c["gain_p"], ".3f",
                   "two-sided paired t-test p for that gain"),
            Figure(f"detach.{cfg}_seeds_better", c["seeds_better"], "d",
                   f"of {len(SEEDS)} seeds, how many this arm beats the shipped "
                   f"fit on"),
            Figure(f"detach.{cfg}_rating_gain_mnat", c["rating_gain"], "+.1f",
                   "the same paired gain with the axis term zeroed — the "
                   "rating's share"),
        ]
    a = m["aniso"]
    figs += [
        Figure("detach.aniso_basis_mean", a["basis_mean"], ".2f",
               "free fit under the anisotropic ridge: cross-seed matched-axis "
               "agreement, averaged over pairs"),
        Figure("detach.aniso_basis_min", a["basis_min"], ".2f",
               "the same at each pair's weakest-matched axis"),
        Figure("detach.aniso_effective_dim", a["effective_dim"], ".1f",
               "free fit under the anisotropic ridge: distinct directions "
               "among the axes"),
        Figure("detach.aniso_max_within_r", a["max_within_r"], ".2f",
               "free fit under the anisotropic ridge: its most correlated pair "
               "of axes"),
        Figure("detach.aniso_level_sd_mnat", a["level_sd_mnat"], ".1f",
               "run-to-run sd of its held-out-problem level, millinats"),
    ]
    return figs


def main():
    m = measure()
    print(f"detach panel  (full budget {m['budget']} epochs, {m['schedule']} "
          f"rate; {len(SEEDS)} seeds per paired configuration at one pinned "
          f"draw and basis; {m['gpu']}; identity over {m['keys']} attempted "
          f"problems; pooled run-to-run sd {m['pooled_sd']:.1f} mnat)\n")
    print(f"{'configuration':<14}{'level':>8}{'sd':>5}{'gain':>7}{'±95%':>6}{'p':>7}"
          f"{'better':>8}{'rating':>8}{'taught':>8}{'basis':>7}{'effdim':>7}{'max|r|':>7}{'s/arm':>7}")
    for cfg, c in m["per"].items():
        g = (f"{c['gain']:>+7.2f}{c['gain_ci']:>6.2f}{c['gain_p']:>7.3f}"
             f"{c['seeds_better']:>5}/{len(SEEDS)}{c['rating_gain']:>+8.2f}"
             if cfg != "shipped" else f"{'—':>7}{'':>6}{'':>7}{'':>8}{'':>8}")
        print(f"{cfg:<14}{c['level']:>8.4f}{c['sd_mnat']:>5.1f}{g}"
              f"{c['taught_agreement']:>8.2f}{c['basis_mean']:>7.2f}"
              f"{c['effective_dim']:>7.1f}{c['max_within_r']:>7.2f}{c['elapsed_s']:>7.0f}")
    print("\nper-seed gains vs shipped (mnat, + = arm better):")
    for cfg, c in m["per"].items():
        if cfg != "shipped":
            print(f"  {cfg:<14}" + " ".join(f"{g:+.2f}" for g in c["gains"]))
    a = m["aniso"]
    print(f"\naniso (free fit, ridge ladder {m['ladder']}): level "
          f"{a['level']:.4f} sd {a['level_sd_mnat']:.1f} | cross-seed basis "
          f"{a['basis_mean']:.2f} / min {a['basis_min']:.2f} | effective dim "
          f"{a['effective_dim']:.1f} | max within |r| {a['max_within_r']:.2f}")


if __name__ == "__main__":
    main()
