#!/usr/bin/env python3
"""What stopping on the held-aside score would cost (methodology §1.8), read off
the convergence traces the panels already wrote.

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
import os
import statistics
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
from pipeline.train_mirt import EPOCHS  # noqa: E402

TIER = "refit"

DEV_RUNS = os.path.join(PROJECT_ROOT, "ablations", "dev_runs")

# Patience settings to replay, in CHECKPOINTS. Spanning an order of magnitude
# because the point is how much the answer depends on this number: a single
# setting would report one epoch and say nothing about what chose it.
PATIENCES = (5, 10, 20, 50)


def shipped_arms():
    """Full-budget runs of the shipped configuration, keyed by seed.

    The shipped configuration is an empty config or one carrying nothing but a
    seed. Several panels run it, and runs at one seed are bitwise identical, so
    a seed contributes ONE trace however many panels produced it."""
    by_seed = {}
    for panel in sorted(os.listdir(DEV_RUNS)) if os.path.isdir(DEV_RUNS) else []:
        panel_dir = os.path.join(DEV_RUNS, panel)
        if not os.path.isdir(panel_dir):
            continue
        for entry in sorted(os.listdir(panel_dir)):
            if not entry.endswith(".json"):
                continue
            rec = json.load(open(os.path.join(panel_dir, entry)))
            cfg = dict(rec["config"])
            seed = cfg.pop("seed", 0)
            if cfg or rec["metrics"]["epochs"] != EPOCHS:
                continue
            trace = rec["metrics"]["trace"]
            require(trace and trace[-1]["epoch"] == EPOCHS,
                    f"{panel}/{entry[:-5]}'s trace does not reach its budget")
            by_seed.setdefault(seed, trace)
    return by_seed


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


def measure():
    by_seed = shipped_arms()
    require(len(by_seed) >= 2,
            f"only {len(by_seed)} seed(s) of the shipped configuration carry a "
            f"full-budget trace under {DEV_RUNS} — the whole point is how far "
            f"the chosen epoch moves between runs, which one run cannot show")

    chosen = {}          # (seed, patience) -> epoch
    gains = []
    for seed, trace in by_seed.items():
        final = trace[-1]["stop_bce"]
        for pat in PATIENCES:
            epoch, best = patience_epoch(trace, pat)
            chosen[(seed, pat)] = epoch
            gains.append(1e3 * (final - best))

    # How much of the movement is the SETTING and how much is the fit: the same
    # spread taken the two ways round, over the same numbers.
    across_settings = [max(chosen[(s, p)] for p in PATIENCES)
                       - min(chosen[(s, p)] for p in PATIENCES)
                       for s in by_seed]
    across_seeds = [max(chosen[(s, p)] for s in by_seed)
                    - min(chosen[(s, p)] for s in by_seed)
                    for p in PATIENCES]
    return {"seeds": len(by_seed), "settings": len(PATIENCES),
            "chosen": chosen,
            "spread_settings": statistics.fmean(across_settings),
            "spread_seeds": statistics.fmean(across_seeds),
            "advantage_mnat": statistics.fmean(gains),
            "advantage_best_mnat": max(gains),
            "ever_beats": 1 if max(gains) > 0 else 0}


def figures():
    m = measure()
    return [
        Figure("stopping.seeds", m["seeds"], "d",
               "seeds of the shipped configuration whose recorded traces a "
               "patience rule is replayed on; runs of one seed are bitwise "
               "identical, so each seed counts once"),
        Figure("stopping.patience_spread_settings", m["spread_settings"], "~10",
               "how far the epoch a patience rule selects moves when only the "
               "PATIENCE SETTING changes, averaged over seeds — the rule's "
               "answer as a property of its own tuning"),
        Figure("stopping.patience_spread_seeds", m["spread_seeds"], "~10",
               "how far it moves when only the SEED changes, at a fixed "
               "setting, averaged over the settings — the same spread taken the "
               "other way round, over the same numbers"),
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
        Figure("stopping.patience_ever_beats_final", m["ever_beats"], "d",
               "1 if a patience rule beat the final epoch at any seed and any "
               "setting, 0 if it never did. An indicator for a fig-guard: 0 "
               "says the fixed budget is not merely lower-variance but "
               "better"),
    ]


def main():
    m = measure()
    print(f"patience rules replayed on {m['seeds']} seed(s) of the shipped "
          f"configuration\n")
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
