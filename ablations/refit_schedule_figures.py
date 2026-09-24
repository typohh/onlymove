#!/usr/bin/env python3
"""What stops the shipped fit: the learning-rate schedule and the epoch budget.
Read from the `schedule` refit panel.

Production decays the rate to zero over a fixed budget and ships the final epoch.
This producer measures the two things that decision rests on — whether the decay
costs held-out accuracy, and where the budget can sit — against the constant-rate
schedule and budget it replaced. Nothing here is assumed from the trainer's
constants except which budget SHIPS, which is read from `pipeline.train_mirt` so
the panel is required to contain the shipping configuration rather than something
near it.

A RUN'S LEVEL IS THE MEAN OF ITS LAST 2% OF CHECKPOINTS, not its closing third.
What ships is the final epoch, so the level has to describe where a run ENDS: the
decayed tail is flat and the constant tail is one draw from a band the fit is
still wandering inside, and a short terminal window is the fair reading of both.
The closing third is not, and reading it that way is what this producer got wrong
before: on held-out PROBLEMS the constant fit is still descending through its own
closing third, so a third-of-the-budget mean scores it above where its runs
actually end and hands the schedule a win it did not earn. The window is a
fraction of each arm's OWN budget, since the panel runs arms at many budgets.

THE UNIT IS HELD-OUT PROBLEMS. A held-out ROW is one more attempt by a user whose
ability was fitted, on a problem whose parameters were, so it keeps improving
while the fit sharpens on material it has already seen — which is exactly what a
decay to zero stops. Judged there, any decay looks costly whether or not it gave
up anything that generalizes. The row arms are read for one thing only: the band
a run wanders inside at its end, which is the trainer's own convergence signal
and the reason a patience rule cannot be trusted to pick an epoch.

THE BUDGET IS A WINDOW, NOT A POINT. Cosine budgets at or above the shipping one
are walked upward while each stays level with it; the walk stops at the first
budget whose runs land worse than the shipping budget's by more than the spread
already accumulated, and that budget is the OVERFIT one. What the window shows is
a plateau — a range of budgets that agree, which is what makes the cheapest of
them a defensible choice rather than the argmax of a noisy curve.

WHY THE DECAY MAKES OVERFITTING PERMANENT. Under a constant rate a run that has
gone too far keeps moving and can wander back; the decay freezes the fit wherever
it stands as the rate reaches zero, so a budget long enough for the held-out
score to have turned locks in whatever it had reached by then. That is what the
overfit budget's climb figure measures — how far above its own held-out minimum a
run finishes — and it is why the window is bounded from above rather than open.
The learning-rate INTEGRAL, in constant-rate epochs, is what makes the two
schedules' budgets comparable at all: a cosine to zero delivers about half its
epoch count, so equal epochs are not equal optimization in either direction.

WHY THE ARMS ARE COMPARABLE. A seed normally drives the held-aside draw too, so
two runs at different seeds differ in the fit AND in the rows they are scored on.
Every arm here pins the draw (`split_seed`), so a difference between runs is the
fit moving — which spans everything one seed re-rolls: the randomized joint-PCA
feature basis, the initial weights and the minibatch order.

    ./.venv/bin/python3 -m ablations.refit_schedule_figures
"""

import json
import math
import os
import sqlite3
import statistics
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402

TIER = "refit"

PANEL = "schedule"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")
SCHEDULES = ("constant", "cosine")

# The terminal window a run's LEVEL is read over, as a fraction of its own
# checkpoint trace: what ships is the last epoch, so the level describes the end
# of the run rather than its closing stretch.
LEVEL_TAIL_FRAC = 0.02
# Where the "which epoch was it cut at" probe reads, as a fraction of the budget.
EPOCH_DRAW_FRAC = 0.8
# The within-run band the row arms report is peak-to-peak over the closing third
# of the budget: wander, not descent, needs a window wide enough to contain the
# oscillation and short enough to sit past convergence.
BAND_FRAC = 2.0 / 3.0


def load_panel():
    """Every arm's record, grouped `unit -> (schedule, budget) -> [(arm, rec)]`,
    with the panel's identity checks. A record that is not this arm, a pin that
    differs across arms, a moved attempt log or an unpinned draw all fail here:
    the whole panel is a set of paired comparisons and none of them survives any
    of those."""
    require(os.path.exists(MANIFEST), f"no {MANIFEST}")
    manifest = json.load(open(MANIFEST))
    arms = {}
    for arm in manifest["arms"]:
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the schedule panel has not finished. Run it with "
                f"`python -m ablations.run_panel schedule` (or "
                f"`auxiliary/vast_orchestration.py ablate schedule`).")
        rec = json.load(open(path))
        require(rec["config"] == manifest["arms"][arm],
                f"{arm} ran {rec['config']!r}, the manifest declares "
                f"{manifest['arms'][arm]!r} — the record is not this arm")
        arms[arm] = rec
    # THIS PANEL CARRIES MANY BUDGETS ON PURPOSE, which is why it does not use
    # the one-budget rule every other panel does: the budget is one of the two
    # things being measured. What is required instead is that every arm can
    # state the WORK it delivered, so budgets under two schedules can be put on
    # one axis.
    for rec in arms.values():
        rec["metrics"]["lr_integral_epochs"] = lr_integral(rec["metrics"])

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
        key = (sched, int(rec["metrics"]["epochs"]))
        by_unit.setdefault(unit, {}).setdefault(key, []).append((arm, rec))
    for unit, groups in by_unit.items():
        for (sched, budget), group in groups.items():
            require(len(group) >= 2,
                    f"{sched}@{budget} has {len(group)} arm(s) on the {unit} "
                    f"unit — a spread between runs needs at least two, and "
                    f"every figure here is read against one")
        rows = {rec["metrics"]["stop_rows"]
                for group in groups.values() for _, rec in group}
        require(len(rows) == 1,
                f"the {unit}-unit arms hold aside different row counts ({rows}) "
                f"at one pinned draw — they are not scored on the same rows")
    require("problem" in by_unit,
            "no arm holds out whole PROBLEMS. The level cannot be judged "
            "without them: a held-out row shares its users and its problems "
            "with training, so a decay that stops the fit sharpening on seen "
            "material looks costly there whether or not it gave up anything "
            "that generalizes")
    # The panel's `pinpca` and `pcaonly` arms decompose what a seed re-rolls
    # (feature basis against initialization and batch order). They are constant
    # row-unit runs like the other row arms, so they enter the within-run band
    # as further samples of it and nothing else — the level comparison is
    # problem-unit only, and the decomposition is not a schedule question.
    return by_unit


def lr_integral(metrics):
    """The learning-rate integral an arm delivered, in constant-rate epochs.

    Recorded by the trainer, and DERIVED for arms that predate the metric —
    exactly, not approximately: the schedule is a closed form of the budget with
    nothing fitted in it, so recomputing from the recorded schedule and epoch
    count reproduces what the run did. Re-training arms to write down a number
    already implied by two fields they carry would be waste, not rigour.
    """
    recorded = metrics.get("lr_integral_epochs")
    if recorded is not None:
        return float(recorded)
    epochs = int(metrics["epochs"])
    sched = metrics.get("lr_schedule", "constant")
    if sched == "constant":
        return float(epochs)
    require(sched == "cosine", f"cannot derive the integral for schedule "
                              f"{sched!r} — only the closed-form ones")
    return float(sum(0.5 * (1.0 + math.cos(math.pi * ep / max(epochs - 1, 1)))
                     for ep in range(epochs)))


def level(metrics):
    """Where a run ENDED: the mean held-out score over its last checkpoints. See
    the module docstring for why this window and not the closing third."""
    trace = [r["stop_bce"] for r in metrics["trace"]]
    n = max(1, round(LEVEL_TAIL_FRAC * len(trace)))
    require(len(trace) > n,
            f"an arm's trace holds {len(trace)} checkpoints — too few to read a "
            f"terminal level off")
    return statistics.fmean(trace[-n:])


def at_fraction(metrics, frac):
    """The single checkpoint nearest a given fraction of the arm's budget — one
    epoch's score, deliberately not averaged, because the question it answers is
    what a run would have shipped had it been cut there."""
    target = frac * int(metrics["epochs"])
    return min(metrics["trace"], key=lambda r: abs(r["epoch"] - target))


def summarize(group):
    """One configuration: the arms, their levels, and the spread between them."""
    levels = {arm: level(rec["metrics"]) for arm, rec in group}
    work = {round(rec["metrics"]["lr_integral_epochs"]) for _, rec in group}
    require(len(work) == 1,
            f"the arms {sorted(levels)} delivered different learning-rate "
            f"integrals {sorted(work)} — they are not one configuration")
    return {
        "arms": sorted(levels),
        "n": len(levels),
        "epochs": {rec["metrics"]["epochs"] for _, rec in group}.pop(),
        "work": work.pop(),
        "levels": list(levels.values()),
        "level": statistics.fmean(levels.values()),
        "sd_mnat": 1e3 * statistics.stdev(levels.values()),
        "elapsed_s": statistics.median(rec["metrics"]["elapsed_s"]
                                       for _, rec in group),
        "max_climb_mnat": max(
            1e3 * (level(rec["metrics"])
                   - min(r["stop_bce"] for r in rec["metrics"]["trace"]))
            for _, rec in group),
        "epoch_draw_mnat": statistics.fmean(
            1e3 * (at_fraction(rec["metrics"], EPOCH_DRAW_FRAC)["stop_bce"]
                   - level(rec["metrics"]))
            for _, rec in group),
    }


def band(group):
    """The within-run wander band the row arms report: peak-to-peak of the
    held-out score over the closing third of the budget, averaged over runs."""
    out = []
    for _, rec in group:
        metrics = rec["metrics"]
        floor_epoch = BAND_FRAC * int(metrics["epochs"])
        tail = [r["stop_bce"] for r in metrics["trace"]
                if r["epoch"] >= floor_epoch]
        require(len(tail) > 1,
                f"an arm carries no closing-third window past epoch "
                f"{floor_epoch:.0f}")
        out.append(1e3 * (max(tail) - min(tail)))
    return statistics.fmean(out)


def shipped_budget():
    """The budget production ships, from the trainer itself — never restated
    here. The schedule is checked with it: this producer reads a cosine panel,
    so a production move off the decay makes every figure below the wrong
    comparison and has to fail rather than quietly measure the retired one."""
    from pipeline.train_mirt import EPOCHS, LR_SCHEDULE
    require(LR_SCHEDULE == "cosine",
            f"production now ships the {LR_SCHEDULE!r} schedule; this panel "
            f"measures the cosine decay against the constant rate and cannot "
            f"speak for a third one")
    return int(EPOCHS)


def pooled_sd(groups):
    """Run-to-run sd pooled over several configurations: residuals about each
    configuration's OWN mean, so a level difference between budgets never enters
    the spread. df = sum(n - 1)."""
    ss = sum((x - g["level"]) ** 2 for g in groups for x in g["levels"])
    df = sum(g["n"] - 1 for g in groups)
    require(df > 0, "no residual degrees of freedom to pool a spread over")
    return 1e3 * math.sqrt(ss / df), df


def measure():
    by_unit = load_panel()
    prob = {k: summarize(g) for k, g in by_unit["problem"].items()}
    ships = shipped_budget()

    const_budgets = sorted(b for s, b in prob if s == "constant")
    require(const_budgets, "the panel has no constant-schedule arm on the "
                           "problem unit — nothing to compare the decay to")
    baseline_budget = min(const_budgets)
    baseline = prob[("constant", baseline_budget)]

    # The constant schedule run to the SHIPPING budget separates the two things
    # the change moved: without it, a decayed arm that beats the baseline has
    # both a different schedule and a longer budget and the win cannot be
    # attributed to either.
    require(("constant", ships) in prob,
            f"the panel has no constant-schedule arm at {ships} epochs — with "
            f"the schedule and the budget both moving at once, neither can be "
            f"credited for what the shipping arm does")
    baseline_long = prob[("constant", ships)]

    cosine_budgets = sorted(b for s, b in prob if s == "cosine")
    require(ships in cosine_budgets,
            f"the panel does not contain the shipping configuration: production "
            f"trains {ships} cosine epochs and the problem-unit cosine arms run "
            f"{cosine_budgets}")
    shipped = prob[("cosine", ships)]

    # THE WINDOW, grown upward from the shipping budget. Each next budget is
    # admitted while it stays level with the shipping one, judged against the
    # spread the window has accumulated so far — the same yardstick a paired
    # contrast would be read against. The first budget that fails is the
    # OVERFIT one, and the walk stops there rather than skipping it: what is
    # being established is a contiguous plateau, not a set of budgets that
    # happen to pass.
    window = [shipped]
    overfit = None
    for budget in [b for b in cosine_budgets if b > ships]:
        sd, _ = pooled_sd(window)
        if 1e3 * (prob[("cosine", budget)]["level"] - shipped["level"]) > sd:
            overfit = prob[("cosine", budget)]
            break
        window.append(prob[("cosine", budget)])
    require(overfit is not None,
            "no cosine budget in the panel is worse than the shipping one — the "
            "sweep ends at a boundary rather than at a turn, so the window has "
            "no upper bound and the budget cannot be defended as bounded above. "
            "Extend the panel to a longer budget")
    window_sd, window_df = pooled_sd(window)
    window_level = statistics.fmean(x for g in window for x in g["levels"])

    from scipy import stats as scipy_stats
    f_stat = (baseline["sd_mnat"] / window_sd) ** 2
    f_p = float(scipy_stats.f.sf(f_stat, baseline["n"] - 1, window_df))

    rowu = by_unit.get("row", {})
    row_const = [k for k in rowu if k[0] == "constant"]
    require(len(row_const) == 1,
            f"the row unit carries constant arms at {sorted(b for _, b in row_const)} "
            f"epochs — the within-run band is reported for one configuration")
    row_cos = [b for s, b in rowu if s == "cosine"]
    require(row_cos, "the row unit carries no cosine arm — the within-run band "
                     "the decay is supposed to shrink has nothing to shrink from")
    row_cos_budget = ships if ships in row_cos else min(row_cos)

    return {
        "problem": prob,
        "baseline": baseline,
        "baseline_long": baseline_long,
        "baseline_budget": baseline_budget,
        "const_budgets": const_budgets,
        "cosine_budgets": cosine_budgets,
        "ships": ships,
        "shipped": shipped,
        "window": window,
        "window_level": window_level,
        "window_sd": window_sd,
        "window_df": window_df,
        "window_n": sum(g["n"] for g in window),
        "overfit": overfit,
        "sd_ratio": baseline["sd_mnat"] / window_sd,
        "f_stat": f_stat,
        "f_p": f_p,
        "row_band_constant": band(rowu[row_const[0]]),
        "row_band_cosine": band(rowu[("cosine", row_cos_budget)]),
        "row_const_budget": row_const[0][1],
        "row_cos_budget": row_cos_budget,
    }


def figures():
    m = measure()
    base, ship, win, over = (m["baseline"], m["shipped"], m["window"],
                             m["overfit"])
    lo, hi = win[0]["epochs"], win[-1]["epochs"]
    cos_note = ("at the shipping budget's own row-unit arms"
                if m["row_cos_budget"] == m["ships"] else
                f"at the {m['row_cos_budget']}-epoch row-unit arms, the only "
                f"cosine budget the row unit carries")
    return [
        Figure("schedule.constant_initializations", base["n"], "d",
               f"runs of the retired constant schedule at {m['baseline_budget']} "
               f"epochs behind the reference level, all on held-out problems at "
               f"one pinned draw, differing only in the seed"),
        Figure("schedule.constant_level_sd_mnat", base["sd_mnat"], ".1f",
               "run-to-run sd of that constant configuration's final level "
               "(mean of the last 2% of checkpoints) on held-out problems, "
               "millinats"),
        Figure("schedule.constant_epoch_draw_mnat", base["epoch_draw_mnat"],
               ".1f",
               "mean over those runs of (the single held-out score at 80% of "
               "the budget - the run's final level), millinats: how much what "
               "ships depends on where the budget happens to fall"),
        Figure("schedule.constant_long_level_cost_mnat",
               1e3 * (m["baseline_long"]["level"] - base["level"]), "+.1f",
               f"final level of the constant schedule run to {m['ships']} epochs "
               f"minus its own {m['baseline_budget']}-epoch level, held-out "
               f"problems, millinats — positive means the undecayed fit also "
               f"gets worse when given the longer budget"),
        Figure("schedule.shipped_budget", m["ships"], "d",
               "the cosine epoch budget production trains, read from "
               "pipeline.train_mirt and required to be an arm of this panel"),
        Figure("schedule.shipped_level_cost_mnat",
               1e3 * (ship["level"] - base["level"]), "+.1f",
               "final level of the shipping cosine budget minus the constant "
               "baseline's, held-out problems, millinats — negative means the "
               "shipped fit generalizes better"),
        Figure("schedule.window_level_cost_mnat",
               1e3 * (m["window_level"] - base["level"]), "+.1f",
               "the same comparison pooled over every cosine budget in the "
               "plateau window, millinats"),
        Figure("schedule.window_level_sd_mnat", m["window_sd"], ".1f",
               "run-to-run sd inside that window, pooled about each budget's "
               "own mean, millinats — the spread a paired contrast under the "
               "shipped schedule is read against"),
        Figure("schedule.sd_ratio", m["sd_ratio"], ".2f",
               "constant-schedule run-to-run sd over the window's, both on "
               "held-out problems: how much of the fit's wander the decay takes "
               "out of the RESULT rather than out of one run's tail"),
        Figure("schedule.sd_ratio_p", m["f_p"], ".2f",
               f"one-sided F-test p that the constant spread exceeds the "
               f"window's, at {base['n'] - 1} and {m['window_df']} degrees of "
               f"freedom — the honest reading of a variance ratio from this few "
               f"runs"),
        Figure("schedule.window_lo_budget", lo, "d",
               "lowest cosine budget in the plateau window, which is the "
               "shipping budget by construction"),
        Figure("schedule.window_hi_budget", hi, "d",
               "highest cosine budget still level with the shipping one: the "
               "walk upward stops before the first budget that is worse by more "
               "than the window's own spread"),
        Figure("schedule.overfit_budget", over["epochs"], "d",
               "smallest cosine budget whose final level is worse than the "
               "shipping budget's by more than the window's run-to-run sd — the "
               "definition of the window's upper bound, not a separate test"),
        Figure("schedule.overfit_level_cost_mnat",
               1e3 * (over["level"] - ship["level"]), "+.1f",
               "that budget's final level minus the shipping budget's, held-out "
               "problems, millinats"),
        Figure("schedule.overfit_max_climb_mnat", over["max_climb_mnat"], ".1f",
               "over that budget's runs, the largest (final level - the run's "
               "own held-out minimum), millinats: how far past its best point a "
               "run finishes, which the decay to zero makes permanent"),
        Figure("schedule.overfit_work_epochs", over["work"], ".0f",
               "that budget's learning-rate integral in constant-rate epochs — "
               "the optimization it actually delivered, not its epoch count"),
        Figure("schedule.shipped_work_epochs", ship["work"], ".0f",
               "the shipping budget's learning-rate integral in constant-rate "
               "epochs"),
        Figure("schedule.wall_clock_ratio",
               ship["elapsed_s"] / base["elapsed_s"], ".2f",
               "median wall clock of the shipping cosine arms over the constant "
               "baseline's — arms shared a GPU pool, so this tracks the budget "
               "ratio rather than timing a run"),
        Figure("schedule.within_run_range_constant_mnat",
               m["row_band_constant"], ".1f",
               "the band a run wanders inside at its end under the retired "
               "CONSTANT rate: peak-to-peak of its held-out-ROW score over the "
               "last third of the budget, millinats, averaged over runs"),
        Figure("schedule.within_run_range_cosine_mnat", m["row_band_cosine"],
               ".1f",
               f"the same band under the cosine decay to zero, {cos_note} — far "
               f"narrower almost by construction, since a rate going to zero "
               f"stops moving the parameters, which is why it settles nothing "
               f"on its own"),
        Figure("schedule.budgets_swept", len(m["cosine_budgets"]), "d",
               "distinct cosine epoch budgets the panel runs on the held-out "
               "problem unit, each with its own replicated arms — a shorter "
               "cosine run is not a prefix of a longer one, so every point is "
               "its own fit"),
    ]


def main():
    m = measure()
    base, ship, over = m["baseline"], m["shipped"], m["overfit"]
    win_budgets = [g["epochs"] for g in m["window"]]

    print("\n=== held-out PROBLEMS — the unit the level is judged on ===")
    print(f"level = mean held-out score over the last {LEVEL_TAIL_FRAC:.0%} of "
          f"each run's own checkpoints (what ships is the final epoch)")
    print(f"{'schedule':<10}{'budget':>8}{'n':>4}{'work':>7}{'level':>10}"
          f"{'sd':>7}{'vs base':>9}   role")
    for sched in SCHEDULES:
        for budget in sorted(b for s, b in m["problem"] if s == sched):
            g = m["problem"][(sched, budget)]
            role = []
            if g is base:
                role.append("baseline")
            if g is ship:
                role.append("SHIPS")
            if budget in win_budgets and sched == "cosine":
                role.append("window")
            if g is over:
                role.append("overfit")
            print(f"{sched:<10}{budget:>8}{g['n']:>4}{g['work']:>7}"
                  f"{g['level']:>10.5f}{g['sd_mnat']:>7.2f}"
                  f"{1e3 * (g['level'] - base['level']):>+9.2f}   "
                  f"{'/'.join(role)}")
    print("  work = learning-rate integral in constant-rate epochs; a cosine to "
          "zero delivers about\n  half its epoch count, so equal epochs is NOT "
          "equal optimization. sd and 'vs base' are millinats.")

    print(f"\n=== the window: cosine budgets level with the shipping one "
          f"({m['ships']}) ===")
    print(f"{'quantity':<34}{'value':>12}")
    for label, value in (
            (f"window {win_budgets[0]}..{win_budgets[-1]}, runs",
             f"{m['window_n']:d}"),
            ("pooled level", f"{m['window_level']:.5f}"),
            (f"vs constant@{m['baseline_budget']}",
             f"{1e3 * (m['window_level'] - base['level']):+.2f} mnat"),
            ("pooled run-to-run sd",
             f"{m['window_sd']:.2f} mnat (df {m['window_df']})"),
            (f"constant@{m['baseline_budget']} run-to-run sd",
             f"{base['sd_mnat']:.2f} mnat (df {base['n'] - 1})"),
            ("sd ratio constant/window", f"{m['sd_ratio']:.2f}"),
            ("one-sided F-test",
             f"F={m['f_stat']:.2f}, p={m['f_p']:.3f}"),
            ("shipping budget wall clock",
             f"{ship['elapsed_s'] / base['elapsed_s']:.2f}x the baseline"),
    ):
        print(f"{label:<34}{value:>12}")

    print(f"\n=== the upper bound: cosine@{over['epochs']} ===")
    print(f"{'quantity':<34}{'value':>12}")
    for label, value in (
            (f"level vs shipping@{m['ships']}",
             f"{1e3 * (over['level'] - ship['level']):+.2f} mnat"),
            ("work delivered",
             f"{over['work']:.0f} vs {ship['work']:.0f} epochs"),
            ("max climb above own minimum",
             f"{over['max_climb_mnat']:.2f} mnat"),
            (f"constant@{m['ships']} vs constant@{m['baseline_budget']}",
             f"{1e3 * (m['baseline_long']['level'] - base['level']):+.2f} mnat"),
    ):
        print(f"{label:<34}{value:>12}")
    print("  the decay freezes the fit as the rate reaches zero, so a budget "
          "past the turn ships\n  the overfit rather than wandering back out of "
          "it — and the constant schedule overfits\n  its own longer budget too, "
          "so the turn is the budget and not the schedule.")

    print(f"\n=== held-out ROWS — the trainer's own convergence signal ===")
    print(f"{'schedule':<10}{'budget':>8}{'closing-third p2p':>20}")
    print(f"{'constant':<10}{m['row_const_budget']:>8}"
          f"{m['row_band_constant']:>20.2f}")
    print(f"{'cosine':<10}{m['row_cos_budget']:>8}"
          f"{m['row_band_cosine']:>20.2f}")
    print("  peak-to-peak of one run's held-out-row score over the closing "
          "third, millinats, averaged\n  over runs. This is the band a patience "
          "rule would be selecting an epoch out of.")
    print(f"\n  a constant run's own end wanders "
          f"{m['row_band_constant'] / m['row_band_cosine']:.1f}x wider than a "
          f"decayed one's, but what licenses the\n  budget decision is the "
          f"PROBLEM-unit table above: level parity across a window of budgets,\n"
          f"  bounded above by a turn the decay makes permanent.")


if __name__ == "__main__":
    main()
