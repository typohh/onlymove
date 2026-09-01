#!/usr/bin/env python3
"""Does decaying the learning rate make a run consistent enough to stop
replicating it? Read from the `schedule` refit panel.

THE QUESTION IS NOT THE OBVIOUS ONE. A constant rate keeps taking full-size
steps long after the fit has arrived, so a cosine decay to zero should narrow
the band a run wanders inside at its end almost by construction — that much is
nearly a tautology and settles nothing. What decides whether a comparison needs
more than one seed is the spread of the RESULT between runs: a paired contrast
is read against how far two runs of one configuration land apart, and a run that
sits perfectly still at the wrong place is no more useful than one that wanders.
So this producer reports three things and refuses to collapse them:

  the WITHIN-RUN band   what the decay is expected to shrink,
  the BETWEEN-RUN spread what actually licenses dropping replicates,
  the LEVEL             whether the consistency was bought by fitting worse.

WHY THE ARMS ARE COMPARABLE AT ALL. A seed normally drives both the
initialization and the held-aside draw, so two runs at different seeds differ in
the fit AND in the rows the fit is scored on — their difference is not a
property of the fit. Every arm of this panel pins the draw (`split_seed`) and
varies only the initialization, so the between-run spread below is the fit
moving and nothing else. That also makes the two schedules' levels directly
comparable, being scored on identical rows.

WHAT WOULD LICENSE ONE SEED. Not a small number in isolation: the between-run
spread has to be small against the contrasts the panels are asked to resolve.
`schedule.between_run_range_cosine_mnat` is that spread; compare it to the
differences quoted in methodology §1.7 and §1.8 before concluding anything, and
note that three initializations estimate a spread crudely — the range of three
draws understates the spread of the distribution they came from.

    ./.venv/bin/python3 -m ablations.refit_schedule_figures
"""

import json
import os
import statistics
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
from pipeline.train_mirt import EPOCHS  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402

import sqlite3  # noqa: E402

TIER = "refit"

PANEL = "schedule"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")
SCHEDULES = ("constant", "cosine")
TAIL_FROM_EPOCH = 2 * EPOCHS // 3


def load_panel():
    require(os.path.exists(MANIFEST), f"no {MANIFEST}")
    manifest = json.load(open(MANIFEST))
    arms = {}
    for arm in manifest["arms"]:
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the schedule panel has not finished. Run it with "
                f"`python -m ablations.run_panel schedule` (or "
                f"`vast_orchestration.py ablate schedule`).")
        rec = json.load(open(path))
        require(rec["config"] == manifest["arms"][arm],
                f"{arm} ran {rec['config']!r}, the manifest declares "
                f"{manifest['arms'][arm]!r} — the record is not this arm")
        require(rec["metrics"]["epochs"] == EPOCHS,
                f"{arm} ran {rec['metrics']['epochs']} epochs, not the "
                f"production budget of {EPOCHS} — a smoke arm is not quotable")
        arms[arm] = rec

    base = next(iter(arms.values()))["pins"]
    for arm, rec in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha", "blocks",
                    "pca_dim", "k_axes"):
            require(rec["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} differs from the panel's — the arms are "
                    f"not one panel")

    # THE DRAW MUST BE THE SAME EVERYWHERE. It is the whole basis for reading a
    # between-run difference as the fit moving, so it is checked rather than
    # trusted to the manifest: an arm scored on other rows would make the
    # headline spread a mixture of two things.
    draws = {rec["metrics"].get("split_seed") for rec in arms.values()}
    require(len(draws) == 1 and None not in draws,
            f"the arms hold aside {len(draws)} different draws ({draws}) — this "
            f"panel's whole design is one pinned draw, and without it no "
            f"between-run figure here means what it says")
    rows = {rec["metrics"]["stop_rows"] for rec in arms.values()}
    require(len(rows) == 1,
            f"the arms hold aside different row counts ({rows}) at one draw — "
            f"they are not scored on the same rows")

    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute(
            "SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"measured at attempt-log frontier {base['db_max_id']}, the log now "
            f"ends at {now} — re-run the panel")

    by_sched = {s: [] for s in SCHEDULES}
    for arm, rec in arms.items():
        sched = rec["metrics"].get("lr_schedule")
        require(sched in by_sched,
                f"{arm} records schedule {sched!r}, which this producer does "
                f"not read")
        by_sched[sched].append((arm, rec))
    for s, group in by_sched.items():
        require(len(group) >= 2,
                f"the {s} schedule has {len(group)} arm(s) — a spread between "
                f"runs needs at least two")
    return by_sched


def read(rec):
    trace = rec["metrics"]["trace"]
    tail = [r["stop_bce"] for r in trace if r["epoch"] >= TAIL_FROM_EPOCH]
    require(len(tail) > 1, "an arm carries no tail window")
    return {"final": trace[-1]["stop_bce"],
            "tail_mean": statistics.fmean(tail),
            "within_sd": 1e3 * statistics.stdev(tail),
            "within_range": 1e3 * (max(tail) - min(tail))}


def measure():
    by_sched = load_panel()
    out = {}
    for sched, group in by_sched.items():
        reads = {arm: read(rec) for arm, rec in group}
        finals = [r["final"] for r in reads.values()]
        out[sched] = {
            "arms": sorted(reads),
            "n": len(reads),
            "within_sd": statistics.fmean(r["within_sd"] for r in reads.values()),
            "within_range": statistics.fmean(r["within_range"]
                                             for r in reads.values()),
            "between_range": 1e3 * (max(finals) - min(finals)),
            "level": statistics.fmean(finals),
            "finals": finals,
        }
    out["level_cost"] = 1e3 * (out["cosine"]["level"] - out["constant"]["level"])
    return out


def figures():
    m = measure()
    c, k = m["constant"], m["cosine"]
    return [
        Figure("schedule.within_run_range_constant_mnat", c["within_range"],
               ".1f",
               "the band a run wanders inside at its end under the CONSTANT "
               "rate that ships: peak-to-peak of its held-out score across the "
               "last third of the budget, millinats, averaged over the panel's "
               "initializations"),
        Figure("schedule.within_run_range_cosine_mnat", k["within_range"], ".1f",
               "the same band under a cosine decay to zero. Expected to be much "
               "narrower almost by construction — a rate going to zero stops "
               "moving the parameters — which is why it settles nothing on its "
               "own"),
        Figure("schedule.between_run_range_constant_mnat", c["between_range"],
               ".1f",
               "THE FIGURE THAT DECIDES REPLICATION under the constant rate: "
               "how far the FINAL held-out score lands apart across "
               "initializations scored on identical rows. The draw is pinned "
               "across arms, so this is the fit moving and not the rows"),
        Figure("schedule.between_run_range_cosine_mnat", k["between_range"],
               ".1f",
               "the same spread under the cosine decay. Dropping replicates "
               "needs this to be small against the contrasts a panel is asked "
               "to resolve — a number to compare, not to read alone"),
        Figure("schedule.cosine_level_cost_mnat", m["level_cost"], ".1f",
               "what the cosine schedule costs on the held-aside rows against "
               "the constant one, millinats, positive meaning the decayed run "
               "fits worse. Consistency bought by settling into a worse fit is "
               "not consistency worth having"),
        Figure("schedule.initializations", c["n"], "d",
               "initializations per schedule behind the spreads above. Three "
               "points estimate a spread crudely and the range of three "
               "understates the distribution they came from"),
    ]


def main():
    m = measure()
    print(f"schedule panel — one pinned held-aside draw, "
          f"{m['constant']['n']} initializations per schedule\n")
    print(f"{'schedule':<12}{'within-run sd':>15}{'within-run p2p':>16}"
          f"{'between-run p2p':>17}{'level (final)':>15}")
    for s in SCHEDULES:
        d = m[s]
        print(f"{s:<12}{d['within_sd']:>13.2f}  {d['within_range']:>14.2f}  "
              f"{d['between_range']:>15.2f}  {d['level']:>14.5f}")
    print("\n  within-run  = one run's own wander over its last third (mnat)")
    print("  between-run = how far the FINAL scores of different "
          "initializations land apart, same rows (mnat)")
    print(f"\ncosine costs {m['level_cost']:+.2f} mnat on the level.")
    better = m["cosine"]["between_range"] < m["constant"]["between_range"]
    print(f"between-run spread {'NARROWS' if better else 'does NOT narrow'} "
          f"under the decay: {m['cosine']['between_range']:.2f} against "
          f"{m['constant']['between_range']:.2f} mnat.")
    print("Whether that licenses one seed is a comparison against the "
          "contrasts being\nasked about, not a property of this number.")


if __name__ == "__main__":
    main()
