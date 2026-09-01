#!/usr/bin/env python3
"""What stopping on the held-aside score would cost (methodology §1.8), read off
the convergence traces the panels already wrote.

THE SUBJECT IS THE CONSTANT-RATE FIT — the alternative to the shipped decayed
schedule, not the shipped fit itself. Both quantities below are properties of
what a constant rate's closing window looks like: it takes full-size steps long
after the fit has arrived, which is the wander a patience rule would be choosing
among. A decayed run's closing window is a different shape and neither figure
here describes it. The arms read are therefore the SCHEDULE PANEL's constant-rate
arms, which are the permanent record of that alternative; every other panel's
arms follow whatever the trainer's baseline currently is and would stop being
constant-rate the moment that baseline moves.

§1.8 spends a fixed budget and selects no checkpoint, and the argument against a
patience rule is that it would choose among epochs differing by noise. That is
checkable without training anything: every full-budget arm records its held-out
score every ten epochs, so a patience rule can simply be REPLAYED on those
traces — the same rule, the same numbers it would have seen, the epoch it would
have stopped at.

Two quantities come out, and the second is the one the section turns on.

WHERE IT STOPS. Replayed at several patience settings across several seeds of
the shipped configuration, the chosen epoch moves — and what moves it is mostly
the SETTING rather than the fit. That is the argument stated as a measurement: a
rule whose answer is a property of its own tuning is not reading anything about
convergence.

WHAT IT WOULD BUY, measured in the direction most favourable to the rule. The
selected epoch's score is a running minimum over the very rows the selection
reads, so any advantage it shows over the final epoch is selection on the same
sample — which makes an advantage an overstatement and a DISADVANTAGE the
stronger finding, since the rule cannot even win on its own terms. That is what
`stopping.patience_ever_beats_final` records. No held-out-of-the-holdout
comparison is available here, and inventing one from the same rows would not be
a different measurement.

    ./.venv/bin/python3 -m ablations.refit_stopping_figures
"""

import json
import math
import os
import statistics
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require, panel_regime  # noqa: E402

TIER = "refit"

DEV_RUNS = os.path.join(PROJECT_ROOT, "ablations", "dev_runs")
# The panel that holds the constant-rate arms. Named rather than searched for:
# these arms are what this producer is ABOUT, and a scan over every panel would
# quietly swap in whatever schedule the trainer's baseline happens to run.
SCHEDULE_PANEL = "schedule"

# Patience settings to replay, in CHECKPOINTS. Spanning an order of magnitude
# because the point is how much the answer depends on this number: a single
# setting would report one epoch and say nothing about what chose it.
PATIENCES = (5, 10, 20, 50)

# Config keys that select the DRAW or the initialization rather than anything
# the model is made of. An arm carrying only these is a replicate of the
# constant-rate configuration; one carrying anything else — another budget,
# another feature basis, a decayed rate — is a different fit.
REPLICATE_KEYS = ("seed", "stop_unit", "split_seed")


def constant_arms(unit="row"):
    """The constant-rate arms at one holdout unit, keyed by seed, with the
    (budget, schedule) regime they share.

    The held-aside draw is PINNED across these arms (`split_seed`), so what
    separates two of them is the initialization alone — which is exactly what
    the between-seed spread below is asked to be about.

    THE BUDGET IS READ OFF THE ARMS, never off the trainer's current constant:
    these traces are only ever compared to each other, and a producer pinned to
    the shipped budget would lose the whole record the moment that constant
    moves. What is required instead is that the arms read here share one budget
    and one schedule.

    THE UNIT IS THE WHOLE POINT OF THE ARGUMENT BELOW. The row split shares its
    users and its problems with training — a held-out row is one more attempt by
    a user whose ability was fitted and on a problem whose parameters were — so
    it keeps improving long after generalization to unseen problems has
    arrived. It is a convergence detector and was never the criterion the budget
    was chosen on."""
    panel_dir = os.path.join(DEV_RUNS, SCHEDULE_PANEL)
    require(os.path.isdir(panel_dir),
            f"no {panel_dir} — this producer replays a patience rule on the "
            f"constant-rate arms that panel records, and there are none")
    by_seed, records = {}, {}
    for entry in sorted(os.listdir(panel_dir)):
        if not entry.endswith(".json"):
            continue
        arm = entry[:-len(".json")]
        with open(os.path.join(panel_dir, entry)) as f:
            rec = json.load(f)
        cfg = dict(rec["config"])
        seed = cfg.pop("seed", 0)
        if cfg.pop("stop_unit", "row") != unit:
            continue
        for key in REPLICATE_KEYS:
            cfg.pop(key, None)
        if cfg or rec["metrics"].get("lr_schedule", "constant") != "constant":
            continue
        prior = records.get(seed, (None, None))[0]
        require(prior is None,
                f"{SCHEDULE_PANEL}/{arm} and {SCHEDULE_PANEL}/{prior} are both "
                f"seed {seed} at the {unit!r} unit — one seed is one "
                f"observation, and counting a repeat twice would report it as "
                f"agreement between runs")
        records[seed] = (arm, rec)
    require(records,
            f"no constant-rate arm of {SCHEDULE_PANEL} holds aside "
            f"{unit}s — run that panel's constant arms first")
    budget, schedule = panel_regime({a: r for a, r in records.values()})
    for seed, (arm, rec) in records.items():
        trace = rec["metrics"]["trace"]
        require(trace and trace[-1]["epoch"] == budget,
                f"{SCHEDULE_PANEL}/{arm}'s trace does not reach the "
                f"{budget}-epoch budget its arms share")
        by_seed[seed] = trace
    return by_seed, budget, schedule


def patience_epoch(trace, patience):
    """The epoch a patience rule would select: the running best, abandoned once
    `patience` consecutive checkpoints fail to improve on it. Causal — it only
    ever looks at checkpoints the rule would already have seen."""
    best = float("inf")
    best_epoch = trace[0]["epoch"]
    stale = 0
    for row in trace:
        if row["stop_bce"] < best:
            best, best_epoch, stale = row["stop_bce"], row["epoch"], 0
        else:
            stale += 1
            if stale >= patience:
                return best_epoch, best
    return best_epoch, best


def tail_shape(by_seed, tail_from):
    """(slope per 500 epochs, detrended sd), both in millinats, averaged over
    the runs given. Fitting a line through the closing window separates a fit
    still descending from noise around a settled one — a spread read as noise
    while the run is still improving is not noise."""
    trend, detrended = [], []
    for trace in by_seed.values():
        pts = [(r["epoch"], r["stop_bce"]) for r in trace
               if r["epoch"] >= tail_from]
        x = np.array([e for e, _ in pts], float)
        y = np.array([v for _, v in pts], float)
        slope, intercept = np.polyfit(x, y, 1)
        trend.append(1e3 * slope * 500.0)
        detrended.append(1e3 * float(np.std(y - (intercept + slope * x), ddof=1)))
    return statistics.fmean(trend), statistics.fmean(detrended)


def measure():
    by_seed, budget, schedule = constant_arms()
    require(len(by_seed) >= 2,
            f"only {len(by_seed)} seed(s) of the constant-rate configuration "
            f"carry a full-budget trace under {DEV_RUNS}/{SCHEDULE_PANEL} — the "
            f"whole point is how far the chosen epoch moves between runs, which "
            f"one run cannot show")

    chosen = {}          # (seed, patience) -> epoch
    gains = []
    best_case = (-math.inf, None)   # (advantage, epoch it stopped at)
    for seed, trace in by_seed.items():
        final = trace[-1]["stop_bce"]
        for pat in PATIENCES:
            epoch, best = patience_epoch(trace, pat)
            chosen[(seed, pat)] = epoch
            gains.append(1e3 * (final - best))
            if gains[-1] > best_case[0]:
                best_case = (gains[-1], epoch)

    # How much of the movement is the SETTING and how much is the fit: the same
    # spread taken the two ways round, over the same numbers.
    across_settings = [max(chosen[(s, p)] for p in PATIENCES)
                       - min(chosen[(s, p)] for p in PATIENCES)
                       for s in by_seed]
    across_seeds = [max(chosen[(s, p)] for s in by_seed)
                    - min(chosen[(s, p)] for s in by_seed)
                    for p in PATIENCES]
    # IS THE CLOSING THIRD SETTLED, AND ON WHICH SIGNAL? Read on both units the
    # panels supply. The row split is what the trainer watches and what a
    # patience rule would stop on; whole held-out problems are what a budget
    # should actually be chosen on, since only that unit can tell continued
    # fitting apart from continued generalizing.
    # The window every spread here is read over: the last third of the budget
    # the arms actually ran, not the trainer's current one.
    tail_from = 2 * budget // 3
    trend, detrended = tail_shape(by_seed, tail_from)
    prob, prob_budget, prob_schedule = constant_arms("problem")
    require(prob,
            "no constant-rate arm holds out whole PROBLEMS — without one the "
            "closing trend can only be read on the row split, which shares its "
            "users and problems with training and cannot say whether a longer "
            "budget would generalize better")
    require((prob_budget, prob_schedule) == (budget, schedule),
            f"the problem-unit arms ran {prob_budget} epochs at "
            f"{prob_schedule!r} against the row-unit arms' {budget} at "
            f"{schedule!r} — the two closing trends would be read over "
            f"different windows of different fits, and the comparison between "
            f"the units is the whole reason both are reported")
    prob_trend, prob_detrended = tail_shape(prob, tail_from)

    return {"seeds": len(by_seed), "settings": len(PATIENCES),
            "budget": budget, "schedule": schedule,
            "trend_per_500": trend, "detrended_sd": detrended,
            "prob_seeds": len(prob), "prob_trend_per_500": prob_trend,
            "prob_detrended_sd": prob_detrended,
            "chosen": chosen,
            "spread_settings": statistics.fmean(across_settings),
            "spread_seeds": statistics.fmean(across_seeds),
            "advantage_mnat": statistics.fmean(gains),
            "advantage_best_mnat": max(gains),
            "advantage_best_epoch": best_case[1],
            "ever_beats": 1 if max(gains) > 0 else 0}


def figures():
    m = measure()
    return [
        Figure("stopping.seeds", m["seeds"], "d",
               "seeds of the CONSTANT-RATE configuration whose recorded traces "
               "a patience rule is replayed on. They share one held-aside "
               "draw, so what separates two of them is the initialization "
               "alone"),
        Figure("stopping.patience_spread_settings", m["spread_settings"], "~10",
               "how far the epoch a patience rule selects moves when only the "
               "PATIENCE SETTING changes, averaged over seeds — the rule's "
               "answer as a property of its own tuning"),
        Figure("stopping.patience_spread_seeds", m["spread_seeds"], "~10",
               "how far it moves when only the SEED changes, at a fixed "
               "setting, averaged over the settings — the same spread taken the "
               "other way round, over the same numbers"),
        Figure("stopping.tail_trend_mnat_per_500", m["trend_per_500"], ".1f",
               "the SLOPE of the ROW-held-out score through the last third of "
               "the budget, millinats per 500 epochs, averaged over seeds. This "
               "is the trainer's convergence detector, whose rows share their "
               "users and problems with training — it keeps descending, which "
               "is why it is not what a budget is chosen on"),
        Figure("stopping.tail_detrended_sd_mnat", m["detrended_sd"], ".1f",
               "what is left of that window's spread once the slope is removed: "
               "the run-to-run noise, against which the raw spread overstates "
               "the noise by whatever the trend contributes"),
        Figure("stopping.prob_tail_trend_mnat_per_500",
               m["prob_trend_per_500"], ".1f",
               "the same slope on whole HELD-OUT PROBLEMS — the unit that can "
               "tell continued generalizing from continued fitting, and the "
               "criterion a budget belongs on. Flat here while the row split "
               "still descends is the ordinary shape of a budget that has "
               "arrived"),
        Figure("stopping.prob_tail_detrended_sd_mnat",
               m["prob_detrended_sd"], ".1f",
               "and its detrended noise, on that unit"),
        Figure("stopping.patience_setting_over_seed_ratio",
               m["spread_settings"] / max(m["spread_seeds"], 1e-9), ".1f",
               "the first spread over the second: how many times more the "
               "chosen epoch answers to the rule's own tuning than to the fit "
               "it is reading. Above one means the setting dominates"),
        Figure("stopping.patience_advantage_mnat", m["advantage_mnat"], ".1f",
               "millinats the selected epoch beats the FINAL epoch by on the "
               "held-aside rows, meaned over seeds and settings; SIGNED, so a "
               "negative value means the rule stops early and ships the worse "
               "fit. Flattering to the rule by construction: the selected epoch "
               "is a running minimum over the very rows the selection reads"),
        Figure("stopping.patience_advantage_best_mnat",
               m["advantage_best_mnat"], ".1f",
               "the rule's BEST case over every seed and setting tried — the "
               "most it could claim under its own criterion"),
        Figure("stopping.patience_best_case_epoch",
               float(m["advantage_best_epoch"]), "~10",
               "the epoch the rule stopped at in that best case. Read it beside "
               "the advantage: a rule can only match a budget it is competing "
               "with by declining to stop early, which is the budget's own "
               "answer arrived at expensively"),
        Figure("stopping.patience_ever_beats_final", m["ever_beats"], "d",
               "1 if a patience rule beat the final epoch at any seed and any "
               "setting, 0 if it never did. An indicator for a fig-guard: 0 "
               "says the fixed budget is not merely lower-variance but "
               "better"),
    ]


def main():
    m = measure()
    print(f"patience rules replayed on {m['seeds']} seed(s) of the "
          f"CONSTANT-RATE configuration")
    print(f"  ({m['budget']} epochs, {m['schedule']} rate, held-aside draw "
          f"pinned across the seeds)\n")
    print(f"{'patience (epochs)':>20}" +
          "".join(f"{f'seed {s}':>10}" for s in sorted(
              {s for s, _ in m["chosen"]})))
    for pat in PATIENCES:
        row = "".join(f"{m['chosen'][(s, pat)]:>10}"
                      for s in sorted({s for s, _ in m["chosen"]}))
        print(f"{pat * 10:>20}{row}")
    print(f"\nchosen epoch moves by {m['spread_settings']:.0f} when only the "
          f"setting changes, {m['spread_seeds']:.0f} when only the seed does")
    print(f"selection's advantage over the final epoch: "
          f"{m['advantage_mnat']:+.1f} mnat mean, {m['advantage_best_mnat']:+.1f} "
          f"at best (negative = the rule ships the worse fit; flattering to the "
          f"rule, since it is scored on its own selection rows)")


if __name__ == "__main__":
    main()
