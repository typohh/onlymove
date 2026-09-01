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

WHY THE ARMS ARE COMPARABLE AT ALL. A seed normally drives the held-aside draw
too, so two runs at different seeds differ in the fit AND in the rows the fit is
scored on — their difference is not a property of the fit. Every arm of this
panel pins the draw (`split_seed`), so the between-run spread below is the fit
moving and the two schedules' levels are scored on identical rows.

WHAT "THE FIT MOVING" INCLUDES, stated because it is broader than it sounds: one
seed re-rolls the randomized joint-PCA FEATURE BASIS, the network's initial
weights and the minibatch order together. A spread between arms differing in the
seed is all three at once. The panel's `pinpca` and `pcaonly` arms separate
them — a large share coming from the basis would point at a variance fix that
has nothing to do with the learning rate.

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
        arms[arm] = rec
    # THIS PANEL CARRIES TWO BUDGETS ON PURPOSE, which is why it does not use
    # the one-budget rule every other panel does: a decayed schedule compared at
    # equal EPOCHS is compared at unequal WORK, and the arms that raise the
    # budget exist precisely to remove that confound. What is required instead
    # is that every arm recorded the integral, so the comparison can be stated
    # in the units that matter.
    for arm, rec in arms.items():
        require(rec["metrics"].get("lr_integral_epochs") is not None,
                f"{arm} carries no lr_integral_epochs — it predates the metric, "
                f"and without it a schedule comparison cannot say whether a "
                f"difference is the schedule or the amount of optimization")

    base = next(iter(arms.values()))["pins"]
    for arm, rec in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha", "blocks",
                    "pca_dim", "k_axes"):
            require(rec["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} differs from the panel's — the arms are "
                    f"not one panel")

    # THE DRAW MUST BE PINNED. It is the whole basis for reading a between-run
    # difference as the fit moving, so it is checked rather than trusted to the
    # manifest. Checked WITHIN a unit: holding out whole problems sets aside a
    # different number of rows than holding out rows does, by construction, so
    # the two units are never compared to each other and never pooled.
    draws = {rec["metrics"].get("split_seed") for rec in arms.values()}
    require(len(draws) == 1 and None not in draws,
            f"the arms hold aside {len(draws)} different draws ({draws}) — this "
            f"panel's whole design is one pinned draw, and without it no "
            f"between-run figure here means what it says")

    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute(
            "SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"measured at attempt-log frontier {base['db_max_id']}, the log now "
            f"ends at {now} — re-run the panel")

    by_unit = {}
    for arm, rec in arms.items():
        sched = rec["metrics"].get("lr_schedule")
        require(sched in SCHEDULES,
                f"{arm} records schedule {sched!r}, which this producer does "
                f"not read")
        unit = rec["config"].get("stop_unit", "row")
        by_unit.setdefault(unit, {s: [] for s in SCHEDULES})[sched].append(
            (arm, rec))
    for unit, by_sched in by_unit.items():
        for s, group in by_sched.items():
            require(len(group) >= 2,
                    f"the {s} schedule has {len(group)} arm(s) on the {unit} "
                    f"unit — a spread between runs needs at least two")
    for unit, by_sched in by_unit.items():
        rows = {rec["metrics"]["stop_rows"]
                for group in by_sched.values() for _, rec in group}
        require(len(rows) == 1,
                f"the {unit}-unit arms hold aside different row counts ({rows}) "
                f"at one pinned draw — they are not scored on the same rows")
    require("problem" in by_unit,
            "no arm holds out whole PROBLEMS. The level cannot be judged "
            "without them: a held-out row shares its users and its problems "
            "with training, so a decay that stops the fit sharpening on seen "
            "material looks costly there whether or not it gave up anything "
            "that generalizes")
    return by_unit


def read(rec):
    trace = rec["metrics"]["trace"]
    tail = [r["stop_bce"] for r in trace if r["epoch"] >= TAIL_FROM_EPOCH]
    require(len(tail) > 1, "an arm carries no tail window")
    return {"final": trace[-1]["stop_bce"],
            "tail_mean": statistics.fmean(tail),
            "within_sd": 1e3 * statistics.stdev(tail),
            "within_range": 1e3 * (max(tail) - min(tail))}


def measure():
    by_unit = load_panel()
    return {unit: measure_unit(by_sched) for unit, by_sched in by_unit.items()}


def measure_unit(by_sched):
    """One holdout unit, grouped by SCHEDULE AND BUDGET.

    Three groups can appear: the constant arms, cosine arms at the constant's
    own budget (matched EPOCHS, and so about half the work), and cosine arms at
    a raised budget (matched or better WORK). They are kept apart because the
    two cosine groups answer different questions and averaging them would answer
    neither."""
    out = {}
    const = by_sched["constant"]
    require(const, "the panel has no constant-schedule arm to compare against")
    const_budget = {rec["metrics"]["epochs"] for _, rec in const}
    require(len(const_budget) == 1,
            f"the constant arms ran at {sorted(const_budget)} epochs — the "
            f"baseline every comparison here is taken against has to be one fit")
    const_budget = const_budget.pop()

    def summarize(group):
        reads = {arm: read(rec) for arm, rec in group}
        finals = [r["final"] for r in reads.values()]
        work = {round(rec["metrics"]["lr_integral_epochs"]) for _, rec in group}
        require(len(work) == 1,
                f"the arms {sorted(reads)} delivered different learning-rate "
                f"integrals {sorted(work)} — they are not one configuration")
        return {"arms": sorted(reads), "n": len(reads),
                "epochs": {rec["metrics"]["epochs"] for _, rec in group}.pop(),
                "work": work.pop(),
                "within_sd": statistics.fmean(r["within_sd"]
                                              for r in reads.values()),
                "within_range": statistics.fmean(r["within_range"]
                                                 for r in reads.values()),
                "between_range": 1e3 * (max(finals) - min(finals)),
                "level": statistics.fmean(finals)}

    out["constant"] = summarize(const)
    for arm, rec in by_sched["cosine"]:
        label = ("cosine_epochs" if rec["metrics"]["epochs"] == const_budget
                 else "cosine_work")
        out.setdefault(label, []).append((arm, rec))
    for label in ("cosine_epochs", "cosine_work"):
        if label in out:
            out[label] = summarize(out[label])
            out[label]["level_cost"] = 1e3 * (out[label]["level"]
                                              - out["constant"]["level"])
            out[label]["work_ratio"] = (out[label]["work"]
                                        / out["constant"]["work"])
    return out


def figures():
    m = measure()
    prob, rowu = m["problem"], m["row"]
    require("cosine_work" in prob,
            "the panel carries no cosine arm at a raised budget on the problem "
            "unit. Without one the level comparison is between a constant run "
            "and a decayed run given about half the optimization, and a loss "
            "cannot be attributed to the schedule")
    work, epochs_matched = prob["cosine_work"], prob.get("cosine_epochs")
    return [
        Figure("schedule.within_run_range_constant_mnat",
               rowu["constant"]["within_range"], ".1f",
               "the band a run wanders inside at its end under the CONSTANT "
               "rate that ships: peak-to-peak of its held-out score over the "
               "last third of the budget, millinats, averaged over runs"),
        Figure("schedule.within_run_range_cosine_mnat",
               rowu["cosine_epochs"]["within_range"], ".1f",
               "the same band under a cosine decay to zero — far narrower "
               "almost by construction, since a rate going to zero stops moving "
               "the parameters, which is why it settles nothing on its own"),
        Figure("schedule.between_run_range_constant_mnat",
               prob["constant"]["between_range"], ".1f",
               "THE FIGURE THAT DECIDES REPLICATION: how far the final "
               "held-out-PROBLEM score lands apart across runs of the constant "
               "schedule scored on identical problems. The draw is pinned, so "
               "it spans what a seed re-rolls — feature basis, initial weights, "
               "batch order — and not the data"),
        Figure("schedule.between_run_range_cosine_mnat",
               work["between_range"], ".1f",
               "the same spread under the cosine decay AT MATCHED-OR-BETTER "
               "WORK. Dropping replicates needs this small against the "
               "contrasts a panel is asked to resolve"),
        Figure("schedule.cosine_level_cost_mnat", work["level_cost"], ".1f",
               "what the cosine schedule costs on held-out PROBLEMS against the "
               "constant one, millinats, positive meaning the decayed fit "
               "generalizes worse — measured with the decayed arm given at "
               "least as much optimization, so a cost here is the schedule and "
               "not a shorter run"),
        Figure("schedule.cosine_work_ratio", work["work_ratio"], ".2f",
               "learning-rate integral of that cosine arm over the constant "
               "arm's, both in constant-rate epochs. At or above one the "
               "decayed arm was not starved; a cosine to zero delivers about "
               "half its epoch count, so equal epochs would put this near 0.5"),
        Figure("schedule.cosine_level_cost_equal_epochs_mnat",
               (epochs_matched or work)["level_cost"], ".1f",
               "the same level comparison against a cosine arm run for the same "
               "EPOCHS rather than the same work — what adopting the schedule "
               "without lengthening the budget would cost, kept because that is "
               "the naive change someone would make"),
        Figure("schedule.initializations", prob["constant"]["n"], "d",
               "runs per configuration behind the spreads above. Three points "
               "estimate a spread crudely, and the range of three understates "
               "the distribution they came from"),
    ]


def main():
    m = measure()
    for unit in ("row", "problem"):
        d = m.get(unit)
        if d is None:
            continue
        head = ("held-out ROWS — the trainer's own convergence signal"
                if unit == "row" else
                "held-out PROBLEMS — the unit the level is judged on")
        print(f"\n=== {head} ===")
        print(f"{'configuration':<16}{'epochs':>8}{'work':>7}{'within p2p':>12}"
              f"{'between p2p':>13}{'level':>11}{'vs constant':>13}")
        for label in ("constant", "cosine_epochs", "cosine_work"):
            g = d.get(label)
            if g is None:
                continue
            cost = ("" if label == "constant"
                    else f"{g['level_cost']:+.2f} mnat")
            print(f"{label:<16}{g['epochs']:>8}{g['work']:>7}"
                  f"{g['within_range']:>12.2f}{g['between_range']:>13.2f}"
                  f"{g['level']:>11.5f}{cost:>13}")
        print("  work = the learning-rate integral in constant-rate epochs; a "
              "cosine to zero delivers\n  about half its epoch count, so equal "
              "epochs is NOT equal optimization.")

    prob = m.get("problem", {})
    if "cosine_work" in prob:
        w = prob["cosine_work"]
        print(f"\nthe decision, on held-out problems with the decay given "
              f"{w['work_ratio']:.2f}x the constant arm's work:")
        print(f"  level              {w['level_cost']:+.2f} mnat")
        print(f"  between-run spread constant "
              f"{prob['constant']['between_range']:.2f} -> cosine "
              f"{w['between_range']:.2f} mnat")
        print("  a decay that narrows one run's own band while leaving the "
              "spread BETWEEN runs\n  where it was has not made replication "
              "unnecessary; it has made each run tidier.")


if __name__ == "__main__":
    main()
