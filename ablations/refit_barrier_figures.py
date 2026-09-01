#!/usr/bin/env python3
"""The divergence barrier's figures (methodology §1.7), read from the `barrier`
refit panel — the first refit-tier producer.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` (`ablations.run_panel`, locally or via
`vast_orchestration.py ablate`) and hard-fails when the panel is missing, short
of budget, internally inconsistent, or STALE against the current attempt log —
the pins rule `docs/mirt_benchmark.md` states, mechanized. Guards and spans keep
evaluating against the last-measured ledger values, so the panel's cost is paid
only when it is deliberately re-run.

  barrier.off_rating_low / _high   the 0.1th / 99.9th percentile of the derived
                                   80%-clear rating with the barrier removed —
                                   the blow-out §1.7 describes.
  barrier.off_slope_at_floor_pct   share of slopes on the clamp without it.
  barrier.off_stop_cost_mnat       the barrier-off arm's held-out (stop-split)
                                   cost against the shipped arm, in millinats —
                                   PAIRED: both arms hold out the same rows,
                                   drawn from the same seed at the same log.
  barrier.tenx_rating_span         the p0.1..p99.9 rating span at ten times the
                                   barrier weight — the compression failure the
                                   weight is swept between.

    ./.venv/bin/python3 -m ablations.refit_barrier_figures
"""

import json
import os
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
from pipeline.train_mirt import EPOCHS  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402

TIER = "refit"

PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", "barrier")
ARMS = ("shipped", "barrier_off", "barrier_10x")


def load_panel():
    arms = {}
    for arm in ARMS:
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the barrier panel has not run. Run it with "
                f"`python -m ablations.run_panel barrier` (or "
                f"`vast_orchestration.py ablate barrier`) first.")
        arms[arm] = json.load(open(path))

    for arm, r in arms.items():
        require(r["metrics"]["epochs"] == EPOCHS,
                f"{arm} ran {r['metrics']['epochs']} epochs, not the production "
                f"budget of {EPOCHS} — a smoke arm is not quotable")
        require(r["metrics"]["trace"]
                and r["metrics"]["trace"][-1]["epoch"] == EPOCHS,
                f"{arm}'s convergence trace does not reach its budget")

    base = arms["shipped"]["pins"]
    for arm, r in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha",
                    "blocks", "pca_dim", "k_axes"):
            require(r["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} ({r['pins'][pin]!r}) differs from the "
                    f"shipped arm's ({base[pin]!r}) — the arms are not one "
                    f"panel and their comparison is void")

    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute("SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"the panel was measured at attempt-log frontier {base['db_max_id']} "
            f"but the log now ends at {now} — the stop-row draw moved, the "
            f"paired comparison is void. Re-run the panel.")
    return arms


def measure():
    arms = load_panel()
    off, ship, tenx = (arms[a]["metrics"] for a in
                       ("barrier_off", "shipped", "barrier_10x"))
    cost = (off["trace"][-1]["stop_bce"] - ship["trace"][-1]["stop_bce"]) * 1e3
    require(cost > 0,
            f"the barrier-off arm is not worse held-out ({cost:+.2f} mnat) — "
            f"the barrier's stated purpose is refuted, not just its magnitude")
    return {
        "off_low": off["rating_pct"]["0.1"],
        "off_high": off["rating_pct"]["99.9"],
        "off_floor_share": off["slope_at_floor_share"],
        "off_cost_mnat": cost,
        "tenx_span": tenx["rating_pct"]["99.9"] - tenx["rating_pct"]["0.1"],
        "ship_span": ship["rating_pct"]["99.9"] - ship["rating_pct"]["0.1"],
        "gpu": arms["shipped"]["pins"]["gpu"],
    }


def figures():
    m = measure()
    return [
        Figure("barrier.off_rating_low", m["off_low"], ".0f",
               "0.1th percentile of the derived rating with the slope barrier "
               "removed — the low side of the blow-out"),
        Figure("barrier.off_rating_high", m["off_high"], ".0f",
               "…and the 99.9th percentile, the high side"),
        Figure("barrier.off_slope_at_floor_pct", m["off_floor_share"], "%.0f",
               "share of slopes on the clamp with the barrier removed"),
        Figure("barrier.off_stop_cost_mnat", m["off_cost_mnat"], ".0f",
               "held-out (stop-split) cost of removing the barrier, millinats, "
               "paired rows against the shipped arm"),
        Figure("barrier.tenx_rating_span", m["tenx_span"], ".1f",
               "p0.1..p99.9 span of the derived rating at ten times the barrier "
               "weight — the compression the weight is swept against"),
    ]


def main():
    m = measure()
    print(f"barrier panel  (both arms full budget, one machine: {m['gpu']})")
    print(f"{'':>22}{'barrier off':>14}{'shipped':>10}{'10x weight':>12}")
    print(f"{'rating p0.1..p99.9':>22}"
          f"{m['off_low']:>7.0f}..{m['off_high']:<5.0f}"
          f"{m['ship_span']:>9.1f}w {m['tenx_span']:>10.1f}w   (w = span)")
    print(f"{'slopes on the clamp':>22}{m['off_floor_share']:>13.1%}")
    print(f"{'held-out cost':>22}{m['off_cost_mnat']:>10.1f} mnat  (off vs "
          f"shipped, paired stop rows)")


if __name__ == "__main__":
    main()
