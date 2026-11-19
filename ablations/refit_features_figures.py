#!/usr/bin/env python3
"""The ACTIVATION WIDTH's figures (methodology §1.3), read from the `features`
refit panel: what each narrower width costs held out against the full width
that ships.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` (`ablations.run_panel`, locally or via
`auxiliary/vast_orchestration.py ablate`) and hard-fails when the panel is missing, short
of budget, running a config other than the manifest's, internally inconsistent,
or STALE against the current attempt log — the pins rule `docs/mirt_benchmark.md`
states, mechanized. Guards and spans keep evaluating against the last-measured
ledger values, so the panel's cost is paid only when it is deliberately re-run.

THE QUESTION IS ONE-SIDED. The shipped width is the activation store's own
(`PCA_DIM == ACT_STORE_DIM`: every channel, standardized, unrotated), so there
is no wider arm and nothing here can say the shipped width "wins from both
sides". What the panel measures is what each NARROWER joint-PCA width costs
against it, and whether that cost is monotone in the width.

THE SHIPPED ARM DIFFERS FROM THE NARROWER ONES IN TWO WAYS AT ONCE: it keeps
every channel, and it is not rotated, where a narrower arm is the leading
components of a joint PCA of the same channels. No rotated full-width arm
exists (a width at or above the store's means no rotation), so a cost here is
the cost of the narrower REPRESENTATION as a whole — truncation and rotation
together — and must not be quoted as the cost of dropping channels alone.

THE SCORE IS ON HELD-OUT PROBLEMS, not on the shipped row split. Every arm runs
`stop_unit="problem"`, so a held-aside problem contributes no row to training at
all and its parameters come from the feature trunk alone — the only split that
can score a claim about the FEATURE side. The row split leaves every problem
partly trained, so a representation able to fingerprint individual problems is
rewarded by it rather than caught. A problem-held arm is UNCALIBRATED: its
floor, lapse and slope scale are as trained — one treatment across every arm,
so the paired costs stand, while the levels and the slopes are not the shipped
(row-held, calibrated) fit's and must not be quoted beside it.

THE HEADLINE READING IS THE MEAN OVER THE LAST THIRD OF THE BUDGET, the window
the other refit producers read their settled level over. The final-epoch
reading is computed and printed beside it, as the spread one checkpoint
carries, but it carries no key of its own: a second key for the same contrast
is an invitation to quote the less stable one.

PAIRING AND SEEDS. Within one seed every arm holds aside the same problems, so a
cross-arm delta is paired, and that is the quantity below. The two seeds hold
aside DIFFERENT problems, so no cross-seed delta exists here and none is taken:
a contrast is formed inside a seed and the seeds are averaged afterwards. What
the second seed buys is a second independent estimate of the same paired delta,
and each contrast carries a 0/1 indicator saying whether the seeds agreed in
SIGN. The seed also re-rolls the randomized PCA basis of the narrower arms.

A REFUTED DIRECTION IS REPORTED, NOT RAISED. Should a narrower width beat the
shipped one, the indicators come back 0 and the costs come back signed; a
fig-guard beside the sentence is what fires, because it is then the sentence
and not the number that has to change.

  features.pca<W>_holdout_bce      held-out-PROBLEM cross-entropy at width W
                                   (64, 128, 256, 512), nats per attempt,
                                   last-third mean, averaged over the two seeds.
                                   Descriptive only: the seeds hold out
                                   different problems, so this column is a level
                                   and never a comparison.
  features.pca<W>_cost_mnat        narrower width W (64, 128, 256) against the
                                   shipped 512, millinats, PAIRED inside a seed
                                   and then averaged over the seeds. Signed,
                                   positive = worse than the shipped width.
  features.pca<W>_cost_sign_consistent
                                   1 only when both seeds put that contrast on
                                   the same side of zero, on the headline
                                   reading. For a `<!-- fig-guard: KEY == 1 -->`,
                                   not a span. The final-epoch readings'
                                   agreement is printed beside it as context.
  features.narrower_all_cost       1 only when EVERY narrower width scores worse
                                   held out than the shipped one at EVERY seed.
  features.cost_monotone_in_width  1 only when, at every seed, the held-out
                                   score strictly improves at each step up the
                                   swept widths (64 > 128 > 256 > 512) — the
                                   cost of narrowing grows the further one
                                   narrows. A 0 with `narrower_all_cost` 1 means
                                   every narrower width costs but not in order.
  features.pca<W>_slope_median     median as-trained per-problem slope at width
                                   W, averaged over the seeds.
  features.slope_rises_with_width  1 only when the median slope strictly rises
                                   at each step up the swept widths at every
                                   seed. DESCRIPTIVE: under the previous
                                   baseline a wider-than-shipped arm carrying
                                   the highest slope was read as a memorization
                                   signature against widening; with no wider arm
                                   that argument has nothing to reject, and this
                                   indicator licenses no such sentence by itself.

WHAT THIS PANEL CANNOT SETTLE. These arms stop at the fit: no rank mapping is
re-derived and no motif reading re-counted, so what a width does downstream of
the likelihood has no figure here. Truncation is not separated from rotation,
above. And the panel runs two seeds.

    ./.venv/bin/python3 -m ablations.refit_features_figures
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

from ablations._figures import Figure, require, panel_regime  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.model_constants import (  # noqa: E402
    ACT_STORE_DIM, PCA_DIM, PRODUCTION_BLOCKS)

TIER = "refit"

PANEL = "features"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")

# The swept widths, ascending — the order monotonicity is read along. The last
# is the shipped one and the widest there is: the activation store's own width,
# unrotated (checked against PCA_DIM and ACT_STORE_DIM in `load_panel`; the
# figure keys embed the widths, so they are stated rather than derived).
WIDTHS = (64, 128, 256, 512)
SHIPPED_WIDTH = WIDTHS[-1]
# Every narrower width, each contrasted against the shipped one.
CONTRASTS = WIDTHS[:-1]
# Adjacent pairs, narrower first: the steps monotonicity is read over.
STEPS = tuple(zip(WIDTHS, WIDTHS[1:]))
SEEDS = (0, 1)
# Every arm scores on held-out PROBLEMS: the claim is about the feature side,
# and the shipped row split leaves every problem partly trained.
STOP_UNIT = "problem"


def arm_name(width, seed):
    """The manifest's name for one (width, seed) arm. The shipped width names
    itself, so the baseline every contrast is taken against is obvious in a
    directory listing and in a failure message."""
    return f"{'shipped' if width == SHIPPED_WIDTH else 'w'}{width}_s{seed}"


ARMS = {(w, s): arm_name(w, s) for w in WIDTHS for s in SEEDS}


# --- the panel record ---------------------------------------------------------

def load_panel():
    """Every arm's record, checked for what makes this panel quotable: one
    regime, the manifest's own config, the held-out-problem split,
    pins agreeing across arms EXCEPT the width they sweep, the within-seed
    pairing, and a snapshot still matching the current attempt log."""
    require(os.path.exists(MANIFEST),
            f"no {MANIFEST} — the panel manifest is what defines these arms")
    with open(MANIFEST) as f:
        manifest = json.load(f)
    require(set(manifest["arms"]) == set(ARMS.values()),
            f"{MANIFEST} names arms {sorted(manifest['arms'])}, this producer "
            f"reads {sorted(ARMS.values())} — the manifest and the claims have "
            f"separated")
    require(SHIPPED_WIDTH == int(PCA_DIM) == int(ACT_STORE_DIM),
            f"this panel's baseline is a width of {SHIPPED_WIDTH}, but the tree "
            f"ships PCA_DIM={PCA_DIM} over a store of {ACT_STORE_DIM} channels "
            f"— the panel is one-sided BECAUSE the shipped width is the full, "
            f"unrotated store, and that is no longer what ships")

    arms = {}
    for key, arm in ARMS.items():
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the features panel has not run at every width and "
                f"seed (missing {arm}: width {key[0]}, seed {key[1]}). Run it "
                f"with `python -m ablations.run_panel features` (or "
                f"`auxiliary/vast_orchestration.py ablate features`) first; the resume "
                f"unit is the arm, so arms already recorded are skipped.")
        with open(path) as f:
            arms[key] = json.load(f)

    # ONE PANEL IS ONE REGIME, read off the arms rather than off the trainer's
    # current constants: the arms are only ever compared to each other, so a
    # later change to the shipped budget or schedule must not make a finished
    # panel unquotable.
    budget, schedule = panel_regime(arms)
    for key, r in arms.items():
        arm, (width, seed) = ARMS[key], key
        require(r["metrics"]["trace"]
                and r["metrics"]["trace"][-1]["epoch"] == budget,
                f"{arm}'s convergence trace does not reach the panel's "
                f"{budget}-epoch budget")
        require(all(math.isfinite(float(t["stop_bce"]))
                    for t in r["metrics"]["trace"]),
                f"{arm}'s held-out trace is not finite throughout. A width that "
                f"diverges is a different finding from a width that costs "
                f"likelihood, not a larger version of it, and this producer has "
                f"no key for it")

        # THE ARM RAN THE MANIFEST'S CONFIG. Every figure below is attributed to
        # a named width at a named seed, so a record disagreeing with the
        # manifest entry it is named after — a manifest edited after the run, or
        # an arm invoked by hand into the panel directory — is a well-formed
        # record of a different experiment under this arm's name.
        want, got = manifest["arms"][arm], r["config"]
        for k, v in want.items():
            require(k in got,
                    f"{arm} was run without {k}, which its manifest entry sets "
                    f"to {v!r} — the record is not this arm")
            same = (float(got[k]) == float(v)
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                    else got[k] == v)
            require(same,
                    f"{arm} ran with {k}={got[k]!r}, the manifest says {v!r}")
        extra = [k for k in got if k not in want]
        require(not extra,
                f"{arm} ran with {extra} beyond its manifest entry — the arm "
                f"carries a knob the panel's claims do not account for")

        # Read off the record rather than left to the manifest comparison above,
        # because the whole panel rests on these two: a row-split arm would
        # produce every number below in the same shape from a split that cannot
        # see the feature side at all, and an arm at the wrong seed would be
        # differenced against a baseline holding out other problems.
        require(got.get("stop_unit") == STOP_UNIT,
                f"{arm} was fitted with stop_unit={got.get('stop_unit')!r}, not "
                f"{STOP_UNIT!r} — its held-out score is not over unseen "
                f"problems and says nothing about the feature width")
        require(int(got["seed"]) == seed,
                f"{arm} ran at seed {got['seed']}, not the {seed} its name "
                f"declares")
        require(int(r["pins"]["pca_dim"]) == width,
                f"{arm} was fitted at pca_dim={r['pins']['pca_dim']}, not the "
                f"{width} its name declares — the arm measures a different "
                f"width from the one every figure would attribute to it")

    # THE ONE PIN THIS PANEL SWEEPS is pca_dim, so it is exempt from the
    # cross-arm equality the other panels apply to it, and is checked against
    # each arm's declared width above instead. Everything else has to agree, or
    # these are not the arms of one panel.
    base = arms[(SHIPPED_WIDTH, SEEDS[0])]["pins"]
    for key, r in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha",
                    "blocks", "act_source", "k_axes"):
            require(r["pins"][pin] == base[pin],
                    f"{ARMS[key]}'s pin {pin} ({r['pins'][pin]!r}) differs from "
                    f"the shipped arm's ({base[pin]!r}) — the arms are not one "
                    f"panel and their comparison is void")
    require(int(base["pca_dim"]) == int(PCA_DIM),
            f"the baseline arm was fitted at pca_dim={base['pca_dim']} but this "
            f"tree ships PCA_DIM={PCA_DIM} — the panel's baseline is not the "
            f"shipped width, so no contrast below is against what ships")
    require(base["blocks"] == str(PRODUCTION_BLOCKS),
            f"the panel was fitted on blocks {base['blocks']!r}, this tree "
            f"ships {PRODUCTION_BLOCKS!r} — a width measured on some other "
            f"feature side is not evidence about the shipped one")

    # THE PAIRING, CHECKED RATHER THAN ASSUMED. The held-out draw comes from the
    # seed over the problem list, so arms at one seed on one attempt log hold
    # aside the same problems — which is what makes a millinat delta paired
    # instead of two independent estimates. Row, user and base-rate counts are
    # seed-independent and must match everywhere; the held-aside count is the
    # seed's own and must match within a seed.
    ship = arms[(SHIPPED_WIDTH, SEEDS[0])]["metrics"]
    for key, r in arms.items():
        for field in ("rows", "users", "base_rate", "bank_spec"):
            require(r["metrics"][field] == ship[field],
                    f"{ARMS[key]} fitted {field}={r['metrics'][field]!r} "
                    f"against the baseline's {ship[field]!r} — the arms are not "
                    f"fitted over the same observations")
    for s in SEEDS:
        want = arms[(SHIPPED_WIDTH, s)]["metrics"]["stop_rows"]
        for w in WIDTHS:
            got = arms[(w, s)]["metrics"]["stop_rows"]
            require(got == want,
                    f"{ARMS[(w, s)]} held aside {got} rows against "
                    f"{ARMS[(SHIPPED_WIDTH, s)]}'s {want} — the two do not hold "
                    f"out the same problems and their delta is not paired")

    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute("SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"the panel was measured at attempt-log frontier {base['db_max_id']} "
            f"but the log now ends at {now} — the held-out draw moved, the "
            f"paired comparison is void. Re-run the panel.")
    return arms, budget, schedule


# --- reading a convergence trace ----------------------------------------------

def stop_curve(record):
    """(epochs, held-out BCE) from an arm's convergence trace."""
    trace = record["metrics"]["trace"]
    return ([int(r["epoch"]) for r in trace],
            [float(r["stop_bce"]) for r in trace])


def deltas_mnat(arm_bce, ship_bce, tail):
    """One arm's held-out cost against the shipped-width arm of its OWN seed, in
    millinats, as a mean over the tail checkpoints and at the final checkpoint.
    Positive = worse than the shipped width. Both are differences taken inside
    one seed, where the two arms hold aside the same problems, so both are
    paired."""
    tail_mean = (statistics.fmean(arm_bce[i] for i in tail)
                 - statistics.fmean(ship_bce[i] for i in tail)) * 1e3
    return tail_mean, (arm_bce[-1] - ship_bce[-1]) * 1e3


def same_sign(*series):
    """1 when every seed's reading lands on the same side of zero in every one of
    the given series, 0 otherwise. A reading of exactly zero agrees with nothing,
    so it is never consistent."""
    return int(all(all(v > 0 for v in s) or all(v < 0 for v in s)
                   for s in series))


# --- the measurement ----------------------------------------------------------

def measure():
    arms, budget, schedule = load_panel()
    curves = {key: stop_curve(r) for key, r in arms.items()}
    grid = curves[(SHIPPED_WIDTH, SEEDS[0])][0]
    for key, (ep, _) in curves.items():
        require(ep == grid,
                f"{ARMS[key]}'s trace is scored at different epochs than the "
                f"baseline's — the deltas would not be like for like")
    # The last third of the budget: past the stretch where these fits still
    # improve, so a mean over it reads the level an arm settled at rather than
    # whichever point of its wander the final checkpoint happened to land on.
    # Same window as the other refit producers, so their tail readings mean the
    # same thing.
    tail_from = 2 * budget // 3
    tail = [i for i, e in enumerate(grid) if e >= tail_from]
    require(len(tail) > 1,
            f"the trace carries {len(tail)} checkpoint(s) at or past epoch "
            f"{tail_from} — the settled level and the wander it averages "
            f"over are read across the last third of the budget, and neither "
            f"exists on one point")

    level = {key: statistics.fmean(bce[i] for i in tail)
             for key, (_, bce) in curves.items()}
    final = {key: bce[-1] for key, (_, bce) in curves.items()}

    # CONTRASTS ARE FORMED INSIDE A SEED and averaged afterwards; nothing here
    # ever subtracts across seeds, which hold aside different problems.
    contrasts = {}
    for w in CONTRASTS:
        tails, finals = [], []
        for s in SEEDS:
            _, ship_bce = curves[(SHIPPED_WIDTH, s)]
            _, arm_bce = curves[(w, s)]
            t, f = deltas_mnat(arm_bce, ship_bce, tail)
            tails.append(t)
            finals.append(f)
        contrasts[w] = {
            "tail": tails, "final": finals,
            "tail_mean": statistics.fmean(tails),
            "final_mean": statistics.fmean(finals),
            # On the headline reading, which is the one the figures carry. The
            # final-epoch series gets the same test for the table, where it says
            # whether a one-checkpoint reading would have agreed — context, not
            # a criterion.
            "sign_consistent": same_sign(tails),
            "final_sign_consistent": same_sign(finals),
        }

    # THE ONE-SIDED QUESTION, as two criteria over the whole panel, never an
    # average that one seed could carry: every narrower width is worse than the
    # shipped one at every seed, and the held-out score strictly improves at
    # each step up the widths at every seed.
    all_cost = int(all(level[(w, s)] > level[(SHIPPED_WIDTH, s)]
                       for s in SEEDS for w in CONTRASTS))
    monotone = int(all(level[(lo, s)] > level[(hi, s)]
                       for s in SEEDS for lo, hi in STEPS))

    slope = {}
    for key, r in arms.items():
        pct = r["metrics"]["slope_pct"]
        require("50" in pct,
                f"{ARMS[key]}'s slope_pct carries no median — the trainer wrote "
                f"a different percentile set than this producer reads")
        v = float(pct["50"])
        require(math.isfinite(v) and v > 0.0,
                f"{ARMS[key]}'s median slope is {v!r}, which is not a slope")
        slope[key] = v
    # Descriptive: whether the median slope climbs with the width, held to every
    # step at every seed. These are AS-TRAINED slopes — a problem-held arm is
    # uncalibrated, so no slope multiplier has been folded in.
    top_slope = {s: max(WIDTHS, key=lambda w: slope[(w, s)]) for s in SEEDS}
    slope_rises = int(all(slope[(lo, s)] < slope[(hi, s)]
                          for s in SEEDS for lo, hi in STEPS))

    # The shipped arm's own spread over the window every contrast is read on —
    # printed as the reason the tail reading is the headline one.
    wander = {s: {"sd": statistics.stdev([curves[(SHIPPED_WIDTH, s)][1][i] * 1e3
                                          for i in tail]),
                  "range": (max(curves[(SHIPPED_WIDTH, s)][1][i] for i in tail)
                            - min(curves[(SHIPPED_WIDTH, s)][1][i]
                                  for i in tail)) * 1e3}
              for s in SEEDS}

    return {
        "tail_epochs": (grid[tail[0]], grid[tail[-1]], len(tail)),
        "budget": budget,
        "schedule": schedule,
        "level": level, "final": final, "contrasts": contrasts,
        "holdout_bce": {w: statistics.fmean(level[(w, s)] for s in SEEDS)
                        for w in WIDTHS},
        "all_cost": all_cost, "monotone": monotone,
        "slope": slope,
        "slope_mean": {w: statistics.fmean(slope[(w, s)] for s in SEEDS)
                       for w in WIDTHS},
        "top_slope": top_slope, "slope_rises": slope_rises,
        "wander": wander,
        "stop_rows": {s: arms[(SHIPPED_WIDTH, s)]["metrics"]["stop_rows"]
                      for s in SEEDS},
        "rows": int(arms[(SHIPPED_WIDTH, SEEDS[0])]["metrics"]["rows"]),
        "gpu": arms[(SHIPPED_WIDTH, SEEDS[0])]["pins"]["gpu"],
    }


def figures():
    m = measure()
    over = f"averaged over the panel's {len(SEEDS)} seeds"
    out = []
    for w in WIDTHS:
        out.append(Figure(
            f"features.pca{w}_holdout_bce", m["holdout_bce"][w], ".4f",
            f"held-out-PROBLEM cross-entropy at an activation width of "
            f"{w}{' (the shipped full width, unrotated)' if w == SHIPPED_WIDTH else ' (joint PCA)'}, nats per attempt, meaned over the last third of the budget "
            f"and then {over}. A level, not a comparison: the seeds hold aside "
            f"different problems"))
    for w in CONTRASTS:
        c = m["contrasts"][w]
        out.append(Figure(
            f"features.pca{w}_cost_mnat", c["tail_mean"], ".1f",
            f"held-out-problem cost of a width of {w} against the shipped "
            f"{SHIPPED_WIDTH}, millinats over the last third of the budget, "
            f"signed (positive = worse than the shipped width). PAIRED inside a "
            f"seed — both arms hold aside the same problems — and then {over}. "
            f"The narrower arm is a joint-PCA truncation and the shipped one is "
            f"unrotated, so this is the cost of the narrower representation, "
            f"not of dropping channels alone"))
        out.append(Figure(
            f"features.pca{w}_cost_sign_consistent", c["sign_consistent"], "d",
            f"1 when both seeds put the {w}-against-{SHIPPED_WIDTH} contrast on "
            f"the same side of zero on the last-third reading, 0 when they "
            f"disagree — the two seeds being independent estimates of one paired "
            f"delta. An indicator for a fig-guard beside a sentence naming that "
            f"contrast's direction, not for a span"))
    chain = " > ".join(str(w) for w in WIDTHS)
    out.append(Figure(
        "features.narrower_all_cost", m["all_cost"], "d",
        f"1 when every narrower width ({', '.join(map(str, CONTRASTS))}) scores "
        f"worse held out than the shipped {SHIPPED_WIDTH} at every seed, on the "
        f"last-third reading; 0 when any narrower arm matches or beats it at "
        f"any seed. An indicator for a fig-guard, not a span"))
    out.append(Figure(
        "features.cost_monotone_in_width", m["monotone"], "d",
        f"1 when, at every seed, the held-out-problem cross-entropy strictly "
        f"falls at each step up the swept widths ({chain} in cross-entropy), on "
        f"the last-third reading — narrowing further always costs more; 0 when "
        f"any step at any seed breaks that order. An indicator for a "
        f"fig-guard, not a span"))
    for w in WIDTHS:
        out.append(Figure(
            f"features.pca{w}_slope_median", m["slope_mean"][w], ".2f",
            f"median fitted per-problem slope at a width of {w}, over every "
            f"fitted problem, {over}. AS TRAINED: the arms hold out whole "
            f"problems and are uncalibrated, so this is not on the shipped "
            f"slopes' scale"))
    out.append(Figure(
        "features.slope_rises_with_width", m["slope_rises"], "d",
        f"1 when the median as-trained slope strictly rises at each step up "
        f"the swept widths at every seed, 0 otherwise. Descriptive only: with "
        f"no arm wider than the shipped one it is not evidence about widening"))
    return out


# --- the table ----------------------------------------------------------------

def main():
    m = measure()
    lo, hi, n = m["tail_epochs"]
    print(f"\nfeatures panel — methodology §1.3, the activation width "
          f"(narrower joint-PCA widths against the shipped full width)")
    print(f"{len(WIDTHS)} widths x {len(SEEDS)} seeds {list(SEEDS)}, full budget "
          f"({m['budget']} epochs, {m['schedule']} rate), one machine "
          f"({m['gpu']}),")
    print(f"{m['rows']:,d} attempt rows per arm. Scored on HELD-OUT PROBLEMS "
          f"(stop_unit={STOP_UNIT!r}): a held problem's")
    print(f"parameters come from the feature trunk alone, which is what a width "
          f"is judged on. Held aside: "
          + ", ".join(f"s{s} {m['stop_rows'][s]:,d} rows" for s in SEEDS) + ".")
    print(f"'tail' is the mean over the {n} checkpoints from epoch {lo} to "
          f"{hi}; 'final' is epoch {m['budget']} alone.")

    print(f"\n{'width':>7}"
          + "".join(f"{f'tail s{s}':>11}{f'final s{s}':>11}" for s in SEEDS)
          + "".join(f"{f'slope s{s}':>11}" for s in SEEDS))
    for w in WIDTHS:
        print(f"{w:>7}"
              + "".join(f"{m['level'][(w, s)]:>11.4f}{m['final'][(w, s)]:>11.4f}"
                        for s in SEEDS)
              + "".join(f"{m['slope'][(w, s)]:>11.2f}" for s in SEEDS)
              + ("   <- shipped width" if w == SHIPPED_WIDTH else ""))
    print(f"  held-out cross-entropy in nats per attempt; slope is the median "
          f"over every fitted problem, AS TRAINED")
    print(f"  (problem-held arms are uncalibrated). The shipped width is the "
          f"store's own, unrotated; the others")
    print(f"  are joint-PCA truncations, so a row differs from the shipped one "
          f"by rotation as well as by width.")

    print(f"\nheld-out cost against the shipped width AT THE SAME SEED, "
          f"millinats (positive = worse)")
    head = "".join(f"{'s' + str(s) + ' tail':>11}{'s' + str(s) + ' final':>12}"
                   for s in SEEDS)
    print(f"{'contrast':>12}{head}{'mean tail':>12}{'mean final':>12}"
          f"{'signs':>9}{'(final)':>10}")
    for w in CONTRASTS:
        c = m["contrasts"][w]
        cells = "".join(f"{t:>11.2f}{f:>12.2f}"
                        for t, f in zip(c["tail"], c["final"]))
        print(f"{f'{w} vs {SHIPPED_WIDTH}':>12}{cells}"
              f"{c['tail_mean']:>12.2f}{c['final_mean']:>12.2f}"
              f"{('agree' if c['sign_consistent'] else 'DIFFER'):>9}"
              f"{('agree' if c['final_sign_consistent'] else 'differ'):>10}")
    print(f"  Each seed's columns are PAIRED (the same held-out problems). The "
          f"two seeds hold aside DIFFERENT")
    print(f"  problems, so they are two independent estimates of one delta and "
          f"are averaged, never subtracted.")
    print(f"  'signs' is the figure: both seeds on the same side of zero on the "
          f"tail reading. '(final)' is the")
    print(f"  same test on the final-epoch column and is context only — a "
          f"disagreement there is the wander below,")
    print(f"  which is why the tail reading is the one quoted.")

    print(f"\n  every narrower width worse than {SHIPPED_WIDTH} at every "
          f"seed: {'YES' if m['all_cost'] else 'NO'}")
    print(f"  held-out score strictly better at each step up "
          f"({' -> '.join(map(str, WIDTHS))}) at every seed: "
          f"{'YES' if m['monotone'] else 'NO'}")
    print(f"  highest median slope: "
          + ", ".join(f"seed {s} -> {m['top_slope'][s]}" for s in SEEDS)
          + f"   (rising at every step at every seed: "
          + f"{'YES' if m['slope_rises'] else 'NO'})")

    print(f"\nthe shipped width's own held-out wander over epochs {lo}..{hi} — "
          f"why the tail reading is the headline")
    for s in SEEDS:
        print(f"{'seed ' + str(s):>12}{m['wander'][s]['sd']:>9.2f} mnat sd"
              f"{m['wander'][s]['range']:>10.2f} mnat peak-to-peak")
    print(f"  A contrast read at one checkpoint carries this spread on top of "
          f"itself.")

    print(f"\nOut of this panel's reach: the arms stop at the fit — no rank "
          f"mapping is re-derived and no motif")
    print(f"reading re-counted — and truncation is not separated from "
          f"rotation: no rotated {SHIPPED_WIDTH}-wide arm exists.\n")


if __name__ == "__main__":
    main()
