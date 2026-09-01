#!/usr/bin/env python3
"""Answer conditioning's rating-parity figures (methodology §1.3), read from the
`blocks` refit panel.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` (`ablations.run_panel`, locally or via
`vast_orchestration.py ablate`) and hard-fails when the panel is missing, short
of budget, running a config other than the manifest's, internally inconsistent,
or STALE against the current attempt log — the pins rule `docs/mirt_benchmark.md`
states, mechanized. Guards and spans keep evaluating against the last-measured
ledger values, so the panel's cost is paid only when it is deliberately re-run.

WHAT THE PANEL VARIES. The shipped trunk input is `PRODUCTION_BLOCKS` —
activations, bot pass rates, the board-size flag — and §1.3's reason for it is
per-cell rather than per-problem: the same trunk is fanned out over every legal
cell, so a block measured ABOUT THE CORRECT MOVE hands every alternative cell a
description of the answer. The `all` arms add those blocks back (move
evaluations, ownership swing, game phase) and change nothing else, which is the
arm §1.3's parity claim is stated against. Everything else in this producer
exists to keep that one contrast readable.

A NULL IS ONLY AS GOOD AS THE SPREAD IT IS READ AGAINST, which is why each
configuration runs at three seeds. "Costs nothing" is not a small number on its
own; it is a number small against the fit's own replicate scatter, and the
shipped configuration's held-out trace wanders across the last third of its
budget by more than the contrast being asked about. So the panel measures both
sides of that comparison and the producer emits the indicator that states it.

BOTH STOP UNITS, MEASURED RATHER THAN CHOSEN. Adding columns is a FEATURE-side
change, and the shipped row split leaves every problem partly trained, so a
layout that fingerprints problems the fit has already seen is rewarded by it
rather than caught — the very failure the trunk's feature-side dropout exists to
suppress. The problem-held-out arms set a whole problem aside at a time, so its
parameters come from the feature trunk alone, and that is the reading which
tests the claim on its own terms. The row arms are kept because the shipped
fit's own convergence signal is a row split, and a null holding on BOTH readings
is worth much more than one holding on either.

THE TWO READINGS NEVER MIX. They score different quantities — generalization to
an unseen problem against generalization to an unseen row of a seen problem — so
their millinats are not comparable, and nothing here subtracts one from the
other or pools them. A delta is paired within one seed AND one stop unit; the
replicate spread is measured separately per unit; every figure carries its unit
in its key.

IF THE TWO READINGS DISAGREE. The row reading holding while the problem reading
fails is the memorization signature: the added blocks buy fit on problems the
model has already seen and not generalization to ones it has not — a finding
ABOUT THE CLAIM, not a fault in the panel, and the case the guard is meant to
catch. The reverse — the problem reading holding while the row reading fails —
carries no such reading: it points at the row split's own wander, which is
larger than these contrasts, rather than at either layout. Both indicators are
emitted so a later reader sees which case they are in without re-running
anything.

TWO READINGS OF THE DELTA IN TIME, AND WHY THE LAST THIRD IS THE HEADLINE. A
single epoch's held-out score is one draw from that wander, so a final-epoch
delta inherits it in full and moves between re-runs of the same configuration;
the mean over the last third of the budget averages the wander down while
staying inside the region where both arms have settled. The final-epoch reading
is emitted beside it because it is the number a reader of the trace would
compute first, and the two disagreeing by more than the spread would itself be a
finding.

THE PAIRING AND ITS LIMIT. The seed drives the initialization AND the held-out
draw, so within one seed and one unit the two arms hold aside exactly the same
rows and the delta between them is paired — the record's row, user and stop-row
counts are checked for that identity rather than assumed. Across seeds nothing
is paired: two seeds are two holdout draws, and this producer therefore quotes
no cross-seed delta, only the per-seed deltas, their mean, and the shipped arm's
own between-seed scatter.

Every key exists once per unit, `<unit>_` naming which reading it belongs to.

  blocks.problem_all_delta_mnat        THE HEADLINE of each reading. Mean over
  blocks.row_all_delta_mnat            the three seeds of the within-seed paired
                                       held-out delta, `all` minus shipped, in
                                       millinats, each seed's delta read as the
                                       mean over the last third of the budget.
                                       Signed: positive means the answer-derived
                                       blocks make the held-out fit WORSE.
  blocks.problem_all_delta_final_mnat  the same average read at the final epoch
  blocks.row_all_delta_final_mnat      alone.
  blocks.problem_all_delta_seed0_mnat  the three per-seed last-third deltas each
  blocks.problem_all_delta_seed1_mnat  headline averages — each one paired, none
  blocks.problem_all_delta_seed2_mnat  comparable with another and none
  blocks.row_all_delta_seed0_mnat      comparable across units.
  blocks.row_all_delta_seed1_mnat
  blocks.row_all_delta_seed2_mnat
  blocks.problem_shipped_seed_spread_mnat  the replicate spread each delta is
  blocks.row_shipped_seed_spread_mnat      read against: the between-seed range
                                       (max minus min) of the SHIPPED arm's own
                                       last-third mean held-out score under that
                                       unit, in millinats. One configuration,
                                       three seeds, so it is scatter and nothing
                                       else.
  blocks.problem_rating_parity_holds   1 when the contrast sits inside that
                                       spread, 0 when it does not. THE CLAIM'S
                                       TEST, because the problem-held-out reading
                                       is the one §1.3's feature-side claim is
                                       about — the figure for a
                                       `<!-- fig-guard: blocks.problem_rating_parity_holds == 1 -->`
                                       beside "costs nothing on the rating", not
                                       for a span.
  blocks.row_rating_parity_holds       the same test on the row reading. Not the
                                       claim's test; it is there so the two can
                                       be compared at a glance.

Neither indicator hides a threshold: the rule is stated in each figure's own
description, and the spread it compares against is measured on this panel's
replicates rather than chosen.

WHAT THIS PANEL DOES NOT REACH. §1.3's parity claim has a second half — that
agreement with declared ranks moves in the third decimal — and reaching it needs
a rank re-map per arm (`map_rating_to_rank` against each arm's own ratings, then
scoring the declarations the mapping accepts), which no arm here runs. The
sign-reversal statistics earlier in the same paragraph are a different
measurement again, made by projecting per-cell vectors through each fit rather
than by comparing held-out likelihoods. Neither has a key here, and neither is
a smaller version of what is measured here.

    ./.venv/bin/python3 -m ablations.refit_blocks_figures
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

from ablations._figures import Figure, require, panel_budget  # noqa: E402
from pipeline.train_mirt import EPOCHS  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.model_constants import K_AXES, PRODUCTION_BLOCKS  # noqa: E402

TIER = "refit"

PANEL = "blocks"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")

# The comparison layout, as `utils.rating.features` names it: the shipped blocks
# plus the answer-derived ones. Named here rather than inlined because every
# check below asks whether an arm really ran it.
ALL_BLOCKS = "all"
LAYOUTS = ("shipped", ALL_BLOCKS)
# The stop units, spelled as `train_mirt.run` and `ablation_mirt --stop-unit`
# spell them. Both readings are carried the whole way through this producer;
# nothing is averaged or differenced across them.
UNITS = ("row", "problem")
# Which reading answers §1.3's claim. The claim is about the FEATURE side, and
# only a held-out PROBLEM is scored on the trunk alone.
CLAIM_UNIT = "problem"
# Three replicates of each configuration. Three rather than two because the
# spread is itself one of the quantities being reported, and a range over two
# points is a difference rather than a spread.
SEEDS = (0, 1, 2)
ARMS = tuple(f"{layout}_{unit}_s{seed}"
             for unit in UNITS for layout in LAYOUTS for seed in SEEDS)

# The last third of the budget, the same window `refit_barrier_figures` reads its
# settled level over: past the stretch where these fits still improve, so a mean
# over it reads the level an arm settled at rather than whichever point of its
# wander the final checkpoint happened to land on.
TAIL_FROM_EPOCH = 2 * EPOCHS // 3


def arm_name(layout, unit, seed):
    return f"{layout}_{unit}_s{seed}"


# --- the panel record ---------------------------------------------------------

def load_panel():
    """Every arm's record, checked for what makes this panel quotable: the
    production budget, the manifest's own config, the layout and stop unit each
    arm is named for, pins agreeing across arms except the one this panel
    sweeps, the within-seed within-unit pairing, and a snapshot still matching
    the current attempt log."""
    require(os.path.exists(MANIFEST),
            f"no {MANIFEST} — the panel manifest is what defines these arms")
    with open(MANIFEST) as f:
        manifest = json.load(f)
    require(set(manifest["arms"]) == set(ARMS),
            f"{MANIFEST} names arms {sorted(manifest['arms'])}, this producer "
            f"reads {sorted(ARMS)} — the manifest and the claims have separated")

    arms = {}
    for arm in ARMS:
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the blocks panel has not run, or has not finished. "
                f"Run it with `python -m ablations.run_panel blocks` (or "
                f"`vast_orchestration.py ablate blocks`) first.")
        with open(path) as f:
            arms[arm] = json.load(f)

    budget = panel_budget(arms)
    for arm, r in arms.items():
        require(r["metrics"]["epochs"] == budget,
                f"{arm} ran {r['metrics']['epochs']} epochs against the panel's "
                f"{budget} — one panel is one budget")
        require(r["metrics"]["trace"]
                and r["metrics"]["trace"][-1]["epoch"] == EPOCHS,
                f"{arm}'s convergence trace does not reach its budget")

    # THE ARM RAN THE MANIFEST'S CONFIG. Every figure below is about a named
    # layout, at a named stop unit, at a named seed, so an arm whose record
    # disagrees with the manifest it is named after — a manifest edited after the
    # run, or a hand-invoked arm dropped into the panel directory — is measuring
    # something else under this arm's name.
    for arm, r in arms.items():
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
        extra = [k for k in got if k not in want and k != "epochs"]
        require(not extra,
                f"{arm} ran with {extra} beyond its manifest entry — the arm "
                f"carries a knob the panel's claims do not account for")

    # THE ARM'S NAME CARRIES ITS READING. The name is what pairs an arm with its
    # baseline and what every per-seed, per-unit figure is attributed to, so a
    # manifest naming one seed or unit while setting another would MISLABEL a
    # paired delta rather than break it — and a delta labelled with the wrong
    # stop unit is the one error this producer's whole structure exists to
    # prevent.
    for unit in UNITS:
        for layout in LAYOUTS:
            for seed in SEEDS:
                cfg = arms[arm_name(layout, unit, seed)]["config"]
                require(int(cfg["seed"]) == seed and cfg["stop_unit"] == unit,
                        f"{arm_name(layout, unit, seed)} ran at seed "
                        f"{cfg['seed']} on the {cfg['stop_unit']!r} stop unit, "
                        f"which its name does not carry — the pairing and the "
                        f"per-unit figures below are made by name")

    # THE LAYOUT PIN IS THIS PANEL'S SWEEP, so it is the one pin exempt from the
    # cross-arm equality below — and exempt only because it is checked harder
    # here instead: each arm carries the EFFECTIVE layout it is named for, and
    # the shipped arms carry what the tree ships today, so a change to
    # PRODUCTION_BLOCKS retires the panel rather than silently renaming its
    # baseline.
    require(ALL_BLOCKS != PRODUCTION_BLOCKS,
            f"the comparison layout and the shipped layout are both "
            f"{PRODUCTION_BLOCKS!r} — this panel has no contrast in it")
    for unit in UNITS:
        for seed in SEEDS:
            ship, other = (arm_name("shipped", unit, seed),
                           arm_name(ALL_BLOCKS, unit, seed))
            require(arms[ship]["pins"]["blocks"] == PRODUCTION_BLOCKS,
                    f"{ship} was fitted on {arms[ship]['pins']['blocks']!r}, but "
                    f"this tree ships {PRODUCTION_BLOCKS!r} — its baseline is "
                    f"not the shipped trunk input")
            require(arms[other]["pins"]["blocks"] == ALL_BLOCKS,
                    f"{other} was fitted on {arms[other]['pins']['blocks']!r}, "
                    f"not {ALL_BLOCKS!r} — it does not add the answer-derived "
                    f"blocks")

    base = arms[arm_name("shipped", UNITS[0], SEEDS[0])]["pins"]
    for arm, r in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha", "pca_dim",
                    "k_axes"):
            require(r["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} ({r['pins'][pin]!r}) differs from the "
                    f"shipped arm's ({base[pin]!r}) — the arms are not one "
                    f"panel and their comparison is void")
    require(int(base["k_axes"]) == K_AXES,
            f"the panel was fitted at K={base['k_axes']}, this tree ships "
            f"K_AXES={K_AXES} — the model the parity claim is about has changed")

    # THE PAIRING, CHECKED RATHER THAN ASSUMED, and checked WITHIN one seed and
    # one stop unit. The split is drawn from the seed over the attempt frame (or
    # over the problem list), so the two arms of one seed at one unit hold aside
    # identical rows — which is what makes their millinat delta a paired
    # comparison instead of two independent estimates. Two seeds are two draws
    # and two units are two quantities; neither is ever compared.
    for unit in UNITS:
        for seed in SEEDS:
            ship = arms[arm_name("shipped", unit, seed)]["metrics"]
            other = arms[arm_name(ALL_BLOCKS, unit, seed)]["metrics"]
            for field in ("rows", "users", "stop_rows", "base_rate"):
                require(other[field] == ship[field],
                        f"on the {unit} unit at seed {seed} the two arms fitted "
                        f"{field}={other[field]!r} against {ship[field]!r} — "
                        f"they do not hold out the same rows and their delta is "
                        f"not paired")
            require(other["bank_spec"] == ship["bank_spec"],
                    f"on the {unit} unit at seed {seed} the two arms were taught "
                    f"different rule banks — they are not one fit with one block "
                    f"changed")

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


# --- reading a convergence trace ----------------------------------------------

def stop_curve(record):
    """(epochs, held-out BCE) from an arm's convergence trace."""
    trace = record["metrics"]["trace"]
    return ([int(r["epoch"]) for r in trace],
            [float(r["stop_bce"]) for r in trace])


def tail_window(epochs):
    """Indices of the checkpoints in the last third of the budget."""
    idx = [i for i, e in enumerate(epochs) if e >= TAIL_FROM_EPOCH]
    require(len(idx) > 1,
            f"the trace carries {len(idx)} checkpoint(s) at or past epoch "
            f"{TAIL_FROM_EPOCH} — the settled level is read over the last third "
            f"of the budget, and a mean over one point is a final-epoch reading "
            f"under another name")
    return idx


# --- the measurement ----------------------------------------------------------

def read_unit(curves, unit, tail):
    """One stop unit's whole reading: the two arms' settled levels per seed, the
    paired deltas both ways, the shipped arm's replicate spread, and whether the
    contrast sits inside it. Everything here is within this unit — the caller
    never combines two of these."""
    ship_tail, all_tail, wander = {}, {}, {}
    delta_tail, delta_final = {}, {}
    for seed in SEEDS:
        _, ship_bce = curves[arm_name("shipped", unit, seed)]
        _, all_bce = curves[arm_name(ALL_BLOCKS, unit, seed)]
        ship_tail[seed] = statistics.fmean(ship_bce[i] for i in tail)
        all_tail[seed] = statistics.fmean(all_bce[i] for i in tail)
        delta_tail[seed] = (all_tail[seed] - ship_tail[seed]) * 1e3
        delta_final[seed] = (all_bce[-1] - ship_bce[-1]) * 1e3
        # The wander the tail mean averages down: context for why the headline
        # is read over a window rather than at a point.
        wander[seed] = (max(ship_bce[i] for i in tail)
                        - min(ship_bce[i] for i in tail)) * 1e3
    spread = (max(ship_tail.values()) - min(ship_tail.values())) * 1e3
    headline = statistics.fmean(delta_tail.values())
    return {
        "ship_tail": ship_tail, "all_tail": all_tail, "wander": wander,
        "delta_tail": delta_tail, "delta_final": delta_final,
        "headline": headline,
        "final": statistics.fmean(delta_final.values()),
        "spread": spread,
        "holds": 1 if abs(headline) <= spread else 0,
    }


def measure():
    arms = load_panel()
    curves = {arm: stop_curve(r) for arm, r in arms.items()}
    grid = curves[arm_name("shipped", UNITS[0], SEEDS[0])][0]
    for arm, (ep, bce) in curves.items():
        require(ep == grid,
                f"{arm}'s trace is scored at different epochs than the shipped "
                f"arm's — the deltas would not be like for like")
        # No arm of this panel is expected to blow up: every one is the shipped
        # configuration with one block group present or absent, scored one way or
        # the other. A diverged trace is a different finding from the one
        # measured here, and there is no key for it.
        require(all(math.isfinite(v) for v in bce),
                f"{arm}'s held-out trace is not finite throughout — a diverged "
                f"arm is not a slower version of this panel's finding, and no "
                f"figure here describes it")
    tail = tail_window(grid)
    return {
        "units": {unit: read_unit(curves, unit, tail) for unit in UNITS},
        "tail_from": grid[tail[0]],
        "tail_points": len(tail),
        "gpu": arms[arm_name("shipped", UNITS[0], SEEDS[0])]["pins"]["gpu"],
    }


# --- the figures ---------------------------------------------------------------

# What each unit's reading is, in one clause, reused by every figure that carries
# it so the two can never be described as the same quantity.
UNIT_NOTE = {
    "row": "held-out ROWS of problems the fit has otherwise seen (the shipped "
           "fit's own convergence split)",
    "problem": "held-out PROBLEMS, whose parameters come from the feature trunk "
               "alone (the reading §1.3's feature-side claim is about)",
}


def figures():
    m = measure()
    figs = []
    for unit in UNITS:
        u = m["units"][unit]
        figs.append(Figure(
            f"blocks.{unit}_all_delta_mnat", u["headline"], ".2f",
            f"held-out delta of the answer-derived blocks added back against "
            f"the shipped trunk input, millinats, scored on {UNIT_NOTE[unit]}, "
            f"averaged over three seeds; each seed's delta is PAIRED (same rows "
            f"held aside) and read as the mean over the last third of the "
            f"budget. Positive means the added blocks fit worse. Not comparable "
            f"with the other unit's delta"))
        figs.append(Figure(
            f"blocks.{unit}_all_delta_final_mnat", u["final"], ".2f",
            f"the same three-seed average on {UNIT_NOTE[unit]}, read at the "
            f"final epoch alone, where one checkpoint is one draw from the "
            f"settled trace's own wander"))
        for seed in SEEDS:
            figs.append(Figure(
                f"blocks.{unit}_all_delta_seed{seed}_mnat",
                u["delta_tail"][seed], ".2f",
                f"the within-seed paired delta at seed {seed} on "
                f"{UNIT_NOTE[unit]}, millinats, last third of the budget — one "
                f"of the three that unit's headline averages; deltas from "
                f"different seeds are different holdout draws, and deltas from "
                f"different units are different quantities"))
        figs.append(Figure(
            f"blocks.{unit}_shipped_seed_spread_mnat", u["spread"], ".2f",
            f"replicate spread of the SHIPPED configuration on "
            f"{UNIT_NOTE[unit]}: the range (max minus min) over the three seeds "
            f"of its own last-third mean held-out score, millinats — one "
            f"configuration, three seeds, so it is scatter alone"))
    figs.append(Figure(
        f"blocks.{CLAIM_UNIT}_rating_parity_holds",
        m["units"][CLAIM_UNIT]["holds"], "d",
        f"THE CLAIM'S TEST: 1 when the answer-conditioning contrast on "
        f"{UNIT_NOTE[CLAIM_UNIT]} sits inside that same reading's replicate "
        f"spread, 0 when it does not. The rule, in full and with no threshold "
        f"in it: the absolute value of blocks.{CLAIM_UNIT}_all_delta_mnat is no "
        f"greater than blocks.{CLAIM_UNIT}_shipped_seed_spread_mnat — the "
        f"averaged paired delta against the between-seed range of the shipped "
        f"arm's last-third mean score, both in millinats and both from the same "
        f"unit. An indicator for a fig-guard, not a span"))
    row_unit = [u for u in UNITS if u != CLAIM_UNIT][0]
    figs.append(Figure(
        f"blocks.{row_unit}_rating_parity_holds",
        m["units"][row_unit]["holds"], "d",
        f"the same test applied to {UNIT_NOTE[row_unit]}: 1 when the absolute "
        f"value of blocks.{row_unit}_all_delta_mnat is no greater than "
        f"blocks.{row_unit}_shipped_seed_spread_mnat. NOT the claim's test — it "
        f"is emitted so the two readings can be compared, and this one holding "
        f"while blocks.{CLAIM_UNIT}_rating_parity_holds is 0 is the signature "
        f"of the added blocks buying memorization rather than generalization"))
    return figs


def main():
    m = measure()
    print(f"blocks panel  (all arms full budget, one machine: {m['gpu']})")
    print(f"  the shipped trunk input ({PRODUCTION_BLOCKS!r}) against "
          f"{ALL_BLOCKS!r}, which adds the answer-derived blocks back")
    print(f"  'last third' = the {m['tail_points']} checkpoints from epoch "
          f"{m['tail_from']} to {EPOCHS}")
    print(f"  a delta is PAIRED within one seed AND one stop unit; no "
          f"cross-seed delta and no cross-unit comparison is quotable")
    for unit in UNITS:
        u = m["units"][unit]
        print()
        print(f"  stop unit {unit!r} — {UNIT_NOTE[unit]}"
              f"{'   <- THE CLAIM' if unit == CLAIM_UNIT else ''}")
        print(f"{'seed':>6}{'shipped':>12}{'all':>12}{'delta mnat':>13}"
              f"{'at final':>11}{'ship wander':>13}")
        for seed in SEEDS:
            print(f"{seed:>6}{u['ship_tail'][seed]:>12.5f}"
                  f"{u['all_tail'][seed]:>12.5f}{u['delta_tail'][seed]:>13.2f}"
                  f"{u['delta_final'][seed]:>11.2f}{u['wander'][seed]:>13.1f}")
        print(f"{'mean':>6}{'':>24}{u['headline']:>13.2f}{u['final']:>11.2f}")
        print(f"    replicate spread (range of the shipped arm's own last-third "
              f"means): {u['spread']:.2f} mnat")
        print(f"    contrast |{u['headline']:.2f}| vs spread {u['spread']:.2f} "
              f"-> parity {'HOLDS' if u['holds'] else 'DOES NOT HOLD'}")
    print()
    row_unit = [u for u in UNITS if u != CLAIM_UNIT][0]
    claim, other = m["units"][CLAIM_UNIT], m["units"][row_unit]
    if claim["holds"] and other["holds"]:
        verdict = ("both readings hold — the null stands on the reading the "
                   "claim is about and on the shipped convergence split")
    elif claim["holds"]:
        verdict = (f"the {CLAIM_UNIT} reading holds and the {row_unit} reading "
                   f"does not — that points at the row split's own wander "
                   f"rather than at either layout")
    elif other["holds"]:
        verdict = (f"the {row_unit} reading holds and the {CLAIM_UNIT} reading "
                   f"does not — the MEMORIZATION SIGNATURE: the added blocks "
                   f"buy fit on problems already seen, not generalization")
    else:
        verdict = ("neither reading holds — the added blocks move the rating by "
                   "more than the fit's own seed scatter under both readings")
    print(f"  {verdict}")
    print(f"  'ship wander' is the range of the shipped arm's held-out score "
          f"INSIDE the window, which is what the window mean averages down")
    print()
    print(f"  NOT measured here: §1.3's declared-rank half (it needs a rank "
          f"re-map per arm) and the per-cell sign reversals (a projection "
          f"measurement, not a likelihood one)")


if __name__ == "__main__":
    main()
