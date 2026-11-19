#!/usr/bin/env python3
"""Whether the curve's two levels are the same from one fit to the next
(methodology §1.5), read from the `levels` refit panel: the SHIPPED fit, row-held
and calibrated as production is, at several seeds.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` and hard-fails when the panel is missing, short of
budget, internally inconsistent or stale against the attempt log.

Every record carries three curves over the same held rows (`metrics.calibration`):
the levels the fit TRAINED, the PLAIN four-number calibration, and the SHIPPED
one that averages each row over its player's ability uncertainty. Two sets of
arms read them:

  draw   draw_s0..  every seed moves initialization, feature basis AND the held
                    draw — the reproducibility of what ships, a magnitude.
  init   draw_s0 + init_s1..  the held draw and the basis pinned to seed 0's,
                    initialization alone moving — the fit's own scatter, with
                    every arm scored on the same held rows.

  levels.<set>_seeds
  levels.<set>_<curve>_<floor|lapse>_<mean|sd|min|max>   curve ∈ trained, plain, shipped
  levels.<set>_slope_mult_<mean|sd>         the calibrated slope multiplier
  levels.<set>_kappa_<mean|sd>              the uncertainty multiplier
  levels.<set>_calibration_gain_mnat        held loss as trained minus shipped, mean
  levels.<set>_calibration_gain_min_mnat    its smallest seed
  levels.<set>_uncertainty_gain_mnat        plain minus shipped, mean
  levels.<set>_uncertainty_gain_min_mnat    its smallest seed
  levels.<set>_plain_higher_seeds           seeds where the plain curve reads BOTH levels higher
  levels.<set>_trained_lapse_max            the largest lapse any fit trained
  levels.<set>_top_gap_max_pp               |actual − predicted| failure rate, top tenth, worst seed
  levels.<set>_bottom_gap_max_pp            |actual − predicted| success rate, bottom twentieth, worst seed
  levels.floor_sd_ratio_init                trained floor's sd over the shipped floor's, init set

    ./.venv/bin/python3 -m ablations.refit_levels_figures
"""

import json
import os
import sqlite3
import statistics
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require, panel_regime  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.model_constants import K_AXES  # noqa: E402

TIER = "refit"

PANEL = "levels"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")

# A nested curve cannot score worse than the one it contains, up to the
# optimizer's own tolerance.
LOSS_TOL = 1e-6
SET_NOTE = {
    "draw": "seeds moving initialization, feature basis and the held draw",
    "init": "seeds moving initialization alone, held draw and basis pinned",
}
CURVE_NOTE = {
    "trained": "as the fit trained it",
    "plain": "under the plain held-row calibration",
    "shipped": "under the shipped uncertainty-averaged held-row calibration",
}


def load_panel():
    require(os.path.exists(MANIFEST), f"no {MANIFEST}")
    manifest = json.load(open(MANIFEST))
    arms = {}
    for arm, cfg in manifest["arms"].items():
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the levels panel has not finished. Run it with "
                f"`python -m ablations.run_panel levels`.")
        rec = json.load(open(path))
        require(rec["config"] == cfg,
                f"{arm} ran {rec['config']!r}, the manifest declares {cfg!r} — "
                f"the record is not this arm")
        mt = rec["metrics"]
        require(mt.get("calibrated") is True and isinstance(mt.get("calibration"), dict),
                f"{arm} carries no held-row calibration")
        require(mt["trace"] and mt["trace"][-1]["epoch"] == mt["epochs"],
                f"{arm}'s trace does not reach its budget")
        arms[arm] = rec
    budget, schedule = panel_regime(arms)

    base = next(iter(arms.values()))["pins"]
    for arm, r in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha", "blocks",
                    "pca_dim", "k_axes"):
            require(r["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} differs — the arms are not one panel")
    require(int(base["k_axes"]) == int(K_AXES),
            f"fitted at K={base['k_axes']}, the shipping constant is {K_AXES}")
    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute("SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"measured at attempt-log frontier {base['db_max_id']}, the log now "
            f"ends at {now} — re-run the panel")

    draw = {a: r for a, r in arms.items() if a.startswith("draw_s")}
    init = {a: r for a, r in arms.items() if a.startswith("init_s")}
    require(len(draw) + len(init) == len(arms), "an arm belongs to neither set")
    require("draw_s0" in draw, "the panel carries no draw_s0, the init set's anchor")
    for a, r in draw.items():
        mt, seed = r["metrics"], int(r["config"]["seed"])
        require(mt["split_seed"] == seed and mt["pca_seed"] == seed,
                f"{a} did not move its held draw and basis with its seed")
    init = {"draw_s0": draw["draw_s0"], **init}
    for a, r in init.items():
        mt = r["metrics"]
        require(mt["split_seed"] == 0 and mt["pca_seed"] == 0,
                f"{a} is not pinned to seed 0's held draw and basis")
    held = {r["metrics"]["calibration"]["held_rows"] for r in init.values()}
    require(len(held) == 1, f"the init set scored different held rows {held}")
    for name, s in (("draw", draw), ("init", init)):
        seeds = [int(r["config"]["seed"]) for r in s.values()]
        require(len(set(seeds)) == len(seeds) >= 3,
                f"the {name} set needs three or more distinct seeds, has {sorted(seeds)}")
    return {"draw": draw, "init": init}, budget, schedule, base["gpu"]


def curves(rec):
    """{curve: (floor, lapse)} plus the three held losses, one arm."""
    cal = rec["metrics"]["calibration"]
    lv = {"trained": (cal["as_trained"]["floor"], cal["as_trained"]["lapse"]),
          "plain": (cal["plain_floor"], cal["plain_lapse"]),
          "shipped": (cal["floor"], cal["lapse"])}
    for name, (c, lam) in lv.items():
        require(0.0 <= c < 1.0 and 0.0 <= lam < 0.5 and c + lam < 1.0,
                f"{name} levels are not a response curve's: floor {c}, lapse {lam}")
    require(cal["held_loss"] <= cal["held_loss_plain"] + LOSS_TOL
            and cal["held_loss_plain"] <= cal["held_loss_as_trained"] + LOSS_TOL,
            f"the held losses are not nested: as trained {cal['held_loss_as_trained']}, "
            f"plain {cal['held_loss_plain']}, shipped {cal['held_loss']}")
    return lv, cal


def spread(vals):
    vals = list(vals)
    return {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals),
            "min": min(vals), "max": max(vals)}


def measure():
    sets, budget, schedule, gpu = load_panel()
    out = {"budget": budget, "schedule": schedule, "gpu": gpu, "sets": {}}
    for name, arms in sets.items():
        rows = {}
        for arm, rec in sorted(arms.items(), key=lambda kv: int(kv[1]["config"]["seed"])):
            lv, cal = curves(rec)
            rows[arm] = {
                "seed": int(rec["config"]["seed"]), "levels": lv,
                "slope_mult": cal["slope_mult"], "kappa": cal["kappa"],
                "cal_gain": 1e3 * (cal["held_loss_as_trained"] - cal["held_loss"]),
                "unc_gain": 1e3 * (cal["held_loss_plain"] - cal["held_loss"]),
                "top_gap": 100 * abs(cal["top10_fail"][0] - cal["top10_fail"][1]),
                "bottom_gap": 100 * abs(cal["bottom5_success"][0] - cal["bottom5_success"][1]),
                "held_rows": cal["held_rows"],
            }
        s = {"rows": rows, "n": len(rows)}
        for curve in ("trained", "plain", "shipped"):
            for i, lvl in enumerate(("floor", "lapse")):
                s[f"{curve}_{lvl}"] = spread(r["levels"][curve][i] for r in rows.values())
        for k in ("slope_mult", "kappa", "cal_gain", "unc_gain"):
            s[k] = spread(r[k] for r in rows.values())
        s["plain_higher"] = sum(
            r["levels"]["plain"][0] > r["levels"]["shipped"][0]
            and r["levels"]["plain"][1] > r["levels"]["shipped"][1] for r in rows.values())
        s["top_gap"] = max(r["top_gap"] for r in rows.values())
        s["bottom_gap"] = max(r["bottom_gap"] for r in rows.values())
        out["sets"][name] = s
    i = out["sets"]["init"]
    require(i["shipped_floor"]["sd"] > 0, "the shipped floor does not vary at all")
    out["floor_sd_ratio_init"] = i["trained_floor"]["sd"] / i["shipped_floor"]["sd"]
    return out


def figures():
    m = measure()
    figs = []
    for name, s in m["sets"].items():
        note = SET_NOTE[name]
        figs.append(Figure(f"levels.{name}_seeds", s["n"], "d", f"fits in the {name} set: {note}"))
        for curve in ("trained", "plain", "shipped"):
            for lvl in ("floor", "lapse"):
                sp = s[f"{curve}_{lvl}"]
                for stat, word in (("mean", "mean"), ("sd", "sd"),
                                   ("min", "smallest"), ("max", "largest")):
                    figs.append(Figure(
                        f"levels.{name}_{curve}_{lvl}_{stat}", sp[stat], ".3f",
                        f"{word} {lvl} {CURVE_NOTE[curve]}, over {note}"))
        figs += [
            Figure(f"levels.{name}_slope_mult_mean", s["slope_mult"]["mean"], ".2f",
                   f"mean calibrated slope multiplier, over {note}"),
            Figure(f"levels.{name}_slope_mult_sd", s["slope_mult"]["sd"], ".2f",
                   f"sd of the calibrated slope multiplier, over {note}"),
            Figure(f"levels.{name}_kappa_mean", s["kappa"]["mean"], ".2f",
                   f"mean multiplier on ability uncertainty, over {note}"),
            Figure(f"levels.{name}_kappa_sd", s["kappa"]["sd"], ".2f",
                   f"sd of the multiplier on ability uncertainty, over {note}"),
            Figure(f"levels.{name}_calibration_gain_mnat", s["cal_gain"]["mean"], ".1f",
                   f"held loss as trained minus the shipped calibration's, mean "
                   f"millinats, over {note}"),
            Figure(f"levels.{name}_calibration_gain_min_mnat", s["cal_gain"]["min"], ".1f",
                   "that gain on its smallest seed, millinats"),
            Figure(f"levels.{name}_uncertainty_gain_mnat", s["unc_gain"]["mean"], ".2f",
                   f"plain calibration's held loss minus the uncertainty-averaged "
                   f"one's, mean millinats, over {note}"),
            Figure(f"levels.{name}_uncertainty_gain_min_mnat", s["unc_gain"]["min"], ".2f",
                   "that gain on its smallest seed, millinats"),
            Figure(f"levels.{name}_plain_higher_seeds", s["plain_higher"], "d",
                   f"of {s['n']} seeds, how many the plain calibration reads both "
                   f"levels higher than the shipped one on"),
            Figure(f"levels.{name}_trained_lapse_max", s["trained_lapse"]["max"], ".3f",
                   f"the largest lapse any fit trained, over {note}"),
            Figure(f"levels.{name}_top_gap_max_pp", s["top_gap"], ".1f",
                   "largest gap between actual and predicted failure rate among the "
                   "held attempts highest on the curve, percentage points, worst seed"),
            Figure(f"levels.{name}_bottom_gap_max_pp", s["bottom_gap"], ".1f",
                   "largest gap between actual and predicted success rate among the "
                   "held attempts lowest on the curve, percentage points, worst seed"),
        ]
    figs.append(Figure("levels.floor_sd_ratio_init", m["floor_sd_ratio_init"], ".1f",
                       "seed-to-seed sd of the trained floor over that of the shipped "
                       "floor, initialization alone moving"))
    return figs


def main():
    m = measure()
    print(f"levels panel  (full budget {m['budget']} epochs, {m['schedule']} rate; {m['gpu']})")
    for name, s in m["sets"].items():
        print(f"\n{name} set — {SET_NOTE[name]}")
        print(f"{'arm':<10}{'held':>8}"
              f"{'trained c':>11}{'λ':>7}{'plain c':>9}{'λ':>7}{'shipped c':>11}{'λ':>7}"
              f"{'slope×':>8}{'κ':>6}{'cal gain':>10}{'unc gain':>10}{'top':>6}{'bottom':>8}")
        for arm, r in s["rows"].items():
            lv = r["levels"]
            print(f"{arm:<10}{r['held_rows']:>8}"
                  f"{lv['trained'][0]:>11.3f}{lv['trained'][1]:>7.3f}"
                  f"{lv['plain'][0]:>9.3f}{lv['plain'][1]:>7.3f}"
                  f"{lv['shipped'][0]:>11.3f}{lv['shipped'][1]:>7.3f}"
                  f"{r['slope_mult']:>8.2f}{r['kappa']:>6.2f}"
                  f"{r['cal_gain']:>10.2f}{r['unc_gain']:>10.2f}"
                  f"{r['top_gap']:>6.1f}{r['bottom_gap']:>8.1f}")
        for curve in ("trained", "plain", "shipped"):
            f, l = s[f"{curve}_floor"], s[f"{curve}_lapse"]
            print(f"  {curve:<8} floor {f['mean']:.3f} ± {f['sd']:.3f} [{f['min']:.3f}, {f['max']:.3f}]"
                  f"   lapse {l['mean']:.3f} ± {l['sd']:.3f} [{l['min']:.3f}, {l['max']:.3f}]")
    print(f"\ntrained floor sd / shipped floor sd, initialization alone: "
          f"{m['floor_sd_ratio_init']:.1f}")


if __name__ == "__main__":
    main()
