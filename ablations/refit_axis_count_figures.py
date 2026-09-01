#!/usr/bin/env python3
"""Selecting the axis COUNT on held-out likelihood (methodology §1.14), read
from the `axis_count` refit panel.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` (`ablations.run_panel`, locally or via
`vast_orchestration.py ablate`) and hard-fails when the panel is missing, short
of budget, running a config other than the manifest's, internally inconsistent,
or STALE against the current attempt log — the pins rule `docs/mirt_benchmark.md`
states, mechanized. Guards and spans keep evaluating against the last-measured
ledger values, so the panel's cost is paid only when it is deliberately re-run.

WHAT THE PANEL VARIES AND WHY THAT IS THE WHOLE OF IT. The shipped count is a
shipping decision — one axis per taught rule — and selecting it on held-out
likelihood instead is the standing alternative. The banks are NESTED by
construction: a fit at k teaches the first k `components` of `axes/labels.json`
and nothing else, which is the trainer's own truncation rather than a second
bank written for this measurement, so between two arms only the count differs.

EIGHT AXES ARE OUT OF REACH. `axes/labels.json` ships seven components and a
count above what it lists halts, there being no rule to teach the extra axis
with. So the panel tests four against six against seven, and §1.14's eight-axis
arm is not a larger version of anything here.

THE DRAWS ARE THE SEEDS. The seed drives the initialization AND the held-aside
split, so WITHIN one seed every count holds aside the same rows and a gap
between two counts is paired — the record's row, user and stop-row counts are
checked for that identity rather than assumed. ACROSS seeds nothing is paired:
two seeds are two holdout draws, which is precisely the spread the claim is read
against, so no cross-seed likelihood delta is emitted here.

TWO READINGS OF EVERY SCORE, AND WHY THE LAST THIRD IS THE HEADLINE. The shipped
configuration's own held-out trace wanders across the last third of a run by
more than the gaps this panel is asked to resolve (`refit_barrier_figures`
measures that wander as `barrier.shipped_wander_sd_mnat`), so a single epoch's
value is not a stable estimate of where an arm sits: a final-epoch gap inherits
the wander in full and moves between re-runs of the same configuration. The mean
over the last third averages it down while staying inside the region where every
arm has settled, and is what the headline keys carry. The final-epoch reading is
emitted beside it because it is what a reader of the trace computes first, and
the two disagreeing by more than the draw spread would itself be a finding.

  axis_count.k4_vs_k6_gap_mnat   THE COMBINED-LIKELIHOOD GAPS between adjacent
  axis_count.k6_vs_k7_gap_mnat   counts, millinats, each PAIRED within a seed
                                 and then averaged over the seeds, read over the
                                 last third of the budget. Signed so that
                                 positive means the SMALLER count fits worse.
  axis_count.k4_vs_k6_gap_final_mnat  the same two gaps at the final epoch
  axis_count.k6_vs_k7_gap_final_mnat  alone.
  axis_count.largest_gap_mnat    the larger of the two headline gaps in absolute
                                 value — the best case for the criterion.
  axis_count.draw_spread_mnat    the SPREAD BETWEEN DRAWS it is read against:
                                 the between-seed range of the shipped count's
                                 own last-third mean held-out score.
  axis_count.gaps_within_draw_spread  1 when the counts are separated by less
                                 than that spread, i.e. when "the likelihood
                                 cannot order the counts" stands as stated.
  axis_count.increment_k4_mnat   THE TRAP: the axis increment per count —
  axis_count.increment_k6_mnat   `stop_bce_no_axes - stop_bce`, what zeroing the
  axis_count.increment_k7_mnat   axis term costs on the panel's held-aside
                                 sample at the final state — millinats,
                                 averaged over seeds.
  axis_count.increment_rises_with_count  1 when that increment is strictly
                                 increasing in the count at EVERY seed, which is
                                 what makes it unusable as the criterion: more
                                 axes absorb more residual whether or not the
                                 model predicts better, so read alone it
                                 recommends the largest count tried.

Both indicators are for a `<!-- fig-guard: KEY == 1 -->` beside the sentence
they license, never for a span, and a 0 on either is a real outcome that fires
its guard rather than something this producer suppresses: the panel is capable
of refuting the claim it measures.

WHAT THIS PANEL DOES NOT REACH. §1.14 states its sweep over four holdout draws
and this panel affords three, so the draw spread here rests on three points;
§1.14's per-draw clause — that the largest gap clears its own uncertainty on
every draw while the next does not — is visible in the printed per-seed table
but carries no key, since with three seeds the per-configuration wander is
estimated once and a per-draw uncertainty test would be reading that one
estimate three times. The eight-axis arm is unreachable outright, above. And
nothing here is a statement about which RULES earn axes: the half-reliability
measurement §1.14 rests that on is a per-user identifiability question, not a
likelihood one, and no arm of this panel measures it.

    ./.venv/bin/python3 -m ablations.refit_axis_count_figures
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
from utils.rating.model_constants import K_AXES  # noqa: E402

TIER = "refit"

PANEL = "axis_count"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")

# The counts the panel sweeps, ascending, and the shipped one among them. Seven
# is the baseline because it is what ships; four and six are reachable because
# the bank truncates, and nothing above seven is reachable at all — labels.json
# lists seven components and the trainer halts rather than teaching an axis it
# has no rule for.
COUNTS = (4, 6, 7)
BASELINE_COUNT = COUNTS[-1]
# Adjacent pairs, smaller first: the gaps a count criterion would read.
ADJACENT = tuple(zip(COUNTS, COUNTS[1:]))
# Three replicates of each count. Three rather than two because the spread
# between draws is itself one of the quantities being reported, and a range over
# two points is a difference rather than a spread.
SEEDS = (0, 1, 2)
# The arm-name stem per count. The baseline says what it is in its own name, so
# a record cannot be mistaken for one of the swept arms.
STEM = {4: "k4", 6: "k6", 7: "shipped_k7"}


# What a held-aside score is scored ON, as the noun the printout and the figure
# notes use. Read off the records rather than assumed: a held-aside ROW is one
# more attempt on a problem the fit has otherwise seen, a held-aside PROBLEM one
# the trunk alone has to place, and a reader cannot tell the two apart from the
# number.
UNIT_NOUN = {"row": "rows", "problem": "problems"}


def panel_unit(arms):
    """The holdout unit every arm of the panel scored on, or a hard failure if
    they differ — a gap averaged over both units is between the units as much
    as between the counts."""
    units = {r["config"].get("stop_unit", "row") for r in arms.values()}
    require(len(units) == 1,
            f"the panel's arms held aside {sorted(units)} — those are different "
            f"quantities and a gap pooled over them is not one gap")
    unit = units.pop()
    require(unit in UNIT_NOUN, f"unknown stop unit {unit!r}")
    return unit


def arm_name(count, seed):
    return f"{STEM[count]}_s{seed}"


ARMS = tuple(arm_name(c, s) for c in COUNTS for s in SEEDS)


# --- the panel record ---------------------------------------------------------

def load_panel():
    """Every arm's record, checked for what makes this panel quotable: the
    production budget, the manifest's own config, the count each arm is named
    for as the fit ACTUALLY ran it, a bank nested inside the baseline's, pins
    agreeing across arms except the count this panel sweeps, the within-seed
    pairing, and a snapshot still matching the current attempt log."""
    require(os.path.exists(MANIFEST),
            f"no {MANIFEST} — the panel manifest is what defines these arms")
    with open(MANIFEST) as f:
        manifest = json.load(f)
    require(set(manifest["arms"]) == set(ARMS),
            f"{MANIFEST} names arms {sorted(manifest['arms'])}, this producer "
            f"reads {sorted(ARMS)} — the manifest and the claims have separated")
    require(BASELINE_COUNT == K_AXES,
            f"this panel's baseline is {BASELINE_COUNT} axes, but the tree ships "
            f"K_AXES={K_AXES} — the count the claim is stated against has "
            f"changed, so the panel is describing a model that is no longer the "
            f"shipped one")

    arms = {}
    for arm in ARMS:
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the axis_count panel has not run, or has not "
                f"finished. Run it with `python -m ablations.run_panel "
                f"axis_count` (or `vast_orchestration.py ablate axis_count`) "
                f"first; the resume unit is the arm, so arms already recorded "
                f"are skipped.")
        with open(path) as f:
            arms[arm] = json.load(f)

    # ONE PANEL IS ONE REGIME, read off the arms rather than off the trainer's
    # current constants: the arms are only ever compared to each other, so a
    # later change to the shipped budget or schedule must not make a finished
    # panel unquotable.
    budget, schedule = panel_regime(arms)
    unit = panel_unit(arms)
    for arm, r in arms.items():
        require(r["metrics"]["trace"]
                and r["metrics"]["trace"][-1]["epoch"] == budget,
                f"{arm}'s convergence trace does not reach the panel's "
                f"{budget}-epoch budget")

    # THE ARM RAN THE MANIFEST'S CONFIG. Every figure below is about a named
    # count at a named seed, so an arm whose record disagrees with the manifest
    # it is named after — a manifest edited after the run, or a hand-invoked arm
    # dropped into the panel directory — is measuring something else under this
    # arm's name.
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

    for count in COUNTS:
        for seed in SEEDS:
            arm = arm_name(count, seed)
            cfg, met = arms[arm]["config"], arms[arm]["metrics"]
            # The arm's NAME is what pairs it with the counts beside it and what
            # every figure is attributed to, so a manifest naming one seed or
            # count while setting another would mislabel a paired gap rather
            # than break it.
            require(int(cfg["seed"]) == seed and int(cfg["k_axes"]) == count,
                    f"{arm} is configured seed={cfg['seed']!r} "
                    f"k_axes={cfg['k_axes']!r}, which its name does not carry — "
                    f"the pairing below is made by name")
            # WHAT THE FIT ACTUALLY CARRIED, not what it was asked for. An arm
            # that silently ran at the production count would make the panel a
            # comparison of one configuration with itself, and every gap below
            # would then be seed noise reported as a count effect.
            require(int(met["k_axes"]) == count,
                    f"{arm} was configured for {count} axes but the fit reports "
                    f"k_axes={met['k_axes']} — the arm did not run the count it "
                    f"is named for, and the panel has no contrast in it")
            require(int(arms[arm]["pins"]["k_axes"]) == count,
                    f"{arm}'s k_axes pin is {arms[arm]['pins']['k_axes']!r}, not "
                    f"{count} — the record's snapshot and its fit disagree")
            # THE BANKS ARE NESTED, which is what makes the count the only thing
            # that differs. The trainer truncates `components`, so a shorter
            # bank must be a PREFIX of the baseline's; anything else means the
            # taught rules changed alongside the count.
            require(len(met["bank_spec"]) == count,
                    f"{arm} was taught {len(met['bank_spec'])} rules at "
                    f"k_axes={count} — the bank and the axis count have "
                    f"separated")
            base_spec = arms[arm_name(BASELINE_COUNT, seed)]["metrics"]["bank_spec"]
            require(list(met["bank_spec"]) == list(base_spec[:count]),
                    f"{arm}'s bank {met['bank_spec']!r} is not the first "
                    f"{count} of the baseline's {base_spec!r} — the banks are "
                    f"not nested, so this panel varies more than the count")
            # The axis increment is read from a metric the trainer records at
            # the final state. An arm fitted before that metric existed carries
            # a well-formed record without it, and there is no way to recover
            # the quantity from the trace.
            require("stop_bce_no_axes" in met,
                    f"{arm}'s record has no `stop_bce_no_axes` — the axis "
                    f"increment is measured by zeroing the axis term at the "
                    f"final state, which the trainer records per fit. This arm "
                    f"predates that metric and cannot be read for it; re-run it")
            require(math.isfinite(float(met["stop_bce_no_axes"])),
                    f"{arm}'s `stop_bce_no_axes` is not finite")

    # THE COUNT PIN IS THIS PANEL'S SWEEP, so `k_axes` is the one pin exempt
    # from the cross-arm equality below — and exempt only because it is checked
    # harder above instead, against the arm's name, its config and the count the
    # fit reports. Everything else has to agree for the arms to be one panel.
    base = arms[arm_name(BASELINE_COUNT, SEEDS[0])]["pins"]
    for arm, r in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha", "blocks",
                    "pca_dim"):
            require(r["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} ({r['pins'][pin]!r}) differs from the "
                    f"shipped arm's ({base[pin]!r}) — the arms are not one "
                    f"panel and their comparison is void")

    # THE PAIRING, CHECKED RATHER THAN ASSUMED, and checked WITHIN a seed only.
    # The stop split is drawn from the seed over the attempt frame, so the arms
    # of one seed hold aside identical rows — which is what makes a millinat gap
    # between two counts a paired comparison instead of two independent
    # estimates. Two different seeds are two different draws and are never
    # subtracted.
    for seed in SEEDS:
        ship = arms[arm_name(BASELINE_COUNT, seed)]["metrics"]
        for count in COUNTS:
            other = arms[arm_name(count, seed)]["metrics"]
            for field in ("rows", "users", "stop_rows", "base_rate"):
                require(other[field] == ship[field],
                        f"at seed {seed} the {count}-axis arm fitted {field}="
                        f"{other[field]!r} against the shipped count's "
                        f"{ship[field]!r} — they do not hold out the same rows "
                        f"and their gap is not paired")

    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute("SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"the panel was measured at attempt-log frontier {base['db_max_id']} "
            f"but the log now ends at {now} — the stop-row draw moved, the "
            f"paired comparison is void. Re-run the panel.")
    return arms, budget, schedule, unit


# --- reading a convergence trace ----------------------------------------------

def stop_curve(record):
    """(epochs, held-out BCE) from an arm's convergence trace."""
    trace = record["metrics"]["trace"]
    return ([int(r["epoch"]) for r in trace],
            [float(r["stop_bce"]) for r in trace])


def tail_window(epochs, tail_from):
    """Indices of the checkpoints in the last third of the budget."""
    idx = [i for i, e in enumerate(epochs) if e >= tail_from]
    require(len(idx) > 1,
            f"the trace carries {len(idx)} checkpoint(s) at or past epoch "
            f"{tail_from} — the settled level is read over the last third "
            f"of the budget, and a mean over one point is a final-epoch reading "
            f"under another name")
    return idx


# --- the measurement ----------------------------------------------------------

def measure():
    arms, budget, schedule, unit = load_panel()
    curves = {arm: stop_curve(r) for arm, r in arms.items()}
    grid = curves[arm_name(BASELINE_COUNT, SEEDS[0])][0]
    for arm, (ep, bce) in curves.items():
        require(ep == grid,
                f"{arm}'s trace is scored at different epochs than the shipped "
                f"arm's — the gaps would not be like for like")
        # No arm here is expected to blow up: every one is the shipped
        # configuration with a shorter or the full taught bank. A diverged trace
        # is a different finding, and no figure below describes it.
        require(all(math.isfinite(v) for v in bce),
                f"{arm}'s held-out trace is not finite throughout — a diverged "
                f"arm is not a slower version of this panel's finding")
    # The last third of the budget, the same window `refit_barrier_figures` and
    # `refit_blocks_figures` read their settled level over: past the stretch
    # where these fits still improve, so a mean over it reads the level an arm
    # settled at rather than whichever point of its wander the final checkpoint
    # sampled.
    idx = tail_window(grid, 2 * budget // 3)

    # The combined held-out score per count per seed, both readings. This is the
    # figure the count decision would turn on: the whole model's likelihood on
    # the held-aside rows, axes included.
    last = {c: {s: statistics.fmean(curves[arm_name(c, s)][1][i] for i in idx)
                for s in SEEDS} for c in COUNTS}
    final = {c: {s: curves[arm_name(c, s)][1][-1] for s in SEEDS}
             for c in COUNTS}

    # A gap is taken INSIDE a seed, smaller count minus larger, so positive
    # means the smaller count fits worse; the seeds are averaged afterwards.
    def gaps(scores):
        out = {}
        for lo, hi in ADJACENT:
            per_seed = {s: (scores[lo][s] - scores[hi][s]) * 1e3 for s in SEEDS}
            out[(lo, hi)] = {"per_seed": per_seed,
                             "mean": statistics.fmean(per_seed.values())}
        return out

    gap_last, gap_final = gaps(last), gaps(final)
    largest = max(abs(g["mean"]) for g in gap_last.values())

    # THE SPREAD BETWEEN DRAWS: one count's own score across the three seeds.
    # The shipped count is the one used, being the panel's baseline; the others
    # are printed beside it so a reader can see the choice is not doing work.
    spread = {c: (max(last[c].values()) - min(last[c].values())) * 1e3
              for c in COUNTS}
    draw_spread = spread[BASELINE_COUNT]

    # THE AXIS INCREMENT: what zeroing the axis term costs on the same held-out
    # rows at the final state. Positive means the axis channel is contributing.
    inc = {c: {s: (arms[arm_name(c, s)]["metrics"]["stop_bce_no_axes"]
                   - curves[arm_name(c, s)][1][-1]) * 1e3 for s in SEEDS}
           for c in COUNTS}
    inc_mean = {c: statistics.fmean(inc[c].values()) for c in COUNTS}
    rises = int(all(all(inc[lo][s] < inc[hi][s] for lo, hi in ADJACENT)
                    for s in SEEDS))

    # The wander the last-third mean averages down, per shipped replicate:
    # context for why the headline is read over a window rather than at a point.
    wander = {s: (max(curves[arm_name(BASELINE_COUNT, s)][1][i] for i in idx)
                  - min(curves[arm_name(BASELINE_COUNT, s)][1][i] for i in idx))
              * 1e3 for s in SEEDS}
    return {
        "last": last, "final": final,
        "gap_last": gap_last, "gap_final": gap_final,
        "largest": largest,
        "spread": spread, "draw_spread": draw_spread,
        "cannot_order": 1 if largest <= draw_spread else 0,
        "inc": inc, "inc_mean": inc_mean, "rises": rises,
        "wander": wander,
        "window_from": grid[idx[0]], "window_points": len(idx),
        "budget": budget, "schedule": schedule,
        "unit_noun": UNIT_NOUN[unit],
        "gpu": arms[arm_name(BASELINE_COUNT, SEEDS[0])]["pins"]["gpu"],
    }


def figures():
    m = measure()
    figs = []
    for lo, hi in ADJACENT:
        figs.append(Figure(
            f"axis_count.k{lo}_vs_k{hi}_gap_mnat", m["gap_last"][(lo, hi)]["mean"],
            ".2f",
            f"combined held-out (stop-split) score at {lo} axes minus the same "
            f"at {hi}, millinats, averaged over the panel's three seeds; each "
            f"seed's gap is PAIRED (same held-aside {m['unit_noun']}) and read "
            f"as the mean "
            f"over the last third of the budget. Positive means {lo} axes fit "
            f"worse than {hi}"))
    for lo, hi in ADJACENT:
        figs.append(Figure(
            f"axis_count.k{lo}_vs_k{hi}_gap_final_mnat",
            m["gap_final"][(lo, hi)]["mean"], ".2f",
            f"the same {lo}-against-{hi} gap read at the final epoch alone, "
            f"where one checkpoint is one draw from the settled trace's own "
            f"wander"))
    figs.append(Figure(
        "axis_count.largest_gap_mnat", m["largest"], ".2f",
        "the larger of the two adjacent-count gaps in absolute value, "
        "millinats, last third of the budget — the best case a held-out "
        "likelihood criterion could make for ordering the counts"))
    figs.append(Figure(
        "axis_count.draw_spread_mnat", m["draw_spread"], ".2f",
        "the spread between DRAWS the gaps are read against: the range (max "
        "minus min) over the three seeds of the shipped count's own last-third "
        "mean held-out score, millinats. One count, three holdout draws, so it "
        "is scatter alone"))
    figs.append(Figure(
        "axis_count.gaps_within_draw_spread", m["cannot_order"], "d",
        "1 when the counts are separated by less than the spread between draws, "
        "0 when they are not. The rule, in full and with no threshold in it: "
        "axis_count.largest_gap_mnat is no greater than "
        "axis_count.draw_spread_mnat — the larger averaged paired gap against "
        "the between-seed range of the shipped count's last-third mean score, "
        "both in millinats. A 1 is the claim that held-out likelihood cannot "
        "order the counts, standing as stated. An indicator for a fig-guard, "
        "not a span"))
    for c in COUNTS:
        figs.append(Figure(
            f"axis_count.increment_k{c}_mnat", m["inc_mean"][c], ".2f",
            f"the axis increment at {c} axes: what zeroing the axis term costs "
            f"on the same held-aside {m['unit_noun']} at the final state "
            f"(`stop_bce_no_axes - stop_bce`), millinats, averaged over the "
            f"three seeds. Positive means the axis channel is contributing. It "
            f"is a diagnostic beside the combined score and never the criterion "
            f"for choosing the count"))
    figs.append(Figure(
        "axis_count.increment_rises_with_count", m["rises"], "d",
        "1 when the axis increment is strictly larger at every higher count at "
        "EVERY seed, 0 when any seed breaks that order. A 1 is the trap: more "
        "axes mechanically absorb more residual whether or not the model "
        "predicts better, so the increment read alone recommends the largest "
        "count tried. An indicator for a fig-guard, not a span"))
    return figs


def main():
    m = measure()
    print(f"axis_count panel  ({len(COUNTS)} counts x {len(SEEDS)} seeds "
          f"{list(SEEDS)}, full budget ({m['budget']} epochs, {m['schedule']} "
          f"rate), one machine: {m['gpu']})")
    print(f"  the shipped count is {BASELINE_COUNT}; the banks are nested, so a "
          f"shorter arm teaches the first k rules of axes/labels.json and "
          f"nothing else")
    print(f"  eight axes are unreachable: a count above the number of "
          f"components axes/labels.json lists halts, there being no rule to "
          f"teach the extra axis with")
    print(f"  held-out = the stop split; 'last third' = the "
          f"{m['window_points']} checkpoints from epoch {m['window_from']} to "
          f"{m['budget']}")
    print(f"  a gap is PAIRED within a seed (same held-aside "
          f"{m['unit_noun']}); no "
          f"cross-seed delta is quotable")

    print("\ncombined held-out score per count and seed (lower is better)")
    head = "".join(f"{f'seed {s} last':>15}{f'seed {s} final':>15}"
                   for s in SEEDS)
    print(f"{'count':>12}{head}{'draw spread':>14}")
    for c in COUNTS:
        cells = "".join(f"{m['last'][c][s]:>15.5f}{m['final'][c][s]:>15.5f}"
                        for s in SEEDS)
        print(f"{c:>12}{cells}{m['spread'][c]:>14.2f}")
    print("  'draw spread' is that count's own between-seed range of the last "
          "reading, in millinats — three holdout draws of one configuration.")

    print("\nadjacent-count gaps, millinats (positive = the smaller count fits "
          "worse)")
    head = "".join(f"{f'seed {s}':>11}" for s in SEEDS)
    print(f"{'gap':>12}{head}{'mean':>11}{'mean final':>13}")
    for lo, hi in ADJACENT:
        g, gf = m["gap_last"][(lo, hi)], m["gap_final"][(lo, hi)]
        cells = "".join(f"{g['per_seed'][s]:>11.2f}" for s in SEEDS)
        print(f"{f'{lo} vs {hi}':>12}{cells}{g['mean']:>11.2f}"
              f"{gf['mean']:>13.2f}")
    verdict = ("CANNOT ORDER THE COUNTS" if m["cannot_order"]
               else "SEPARATES THE COUNTS")
    print(f"  largest |gap| {m['largest']:.2f} vs the shipped count's draw "
          f"spread {m['draw_spread']:.2f} -> held-out likelihood {verdict}: the "
          f"counts are "
          f"{'closer together than' if m['cannot_order'] else 'further apart than'}"
          f" one configuration's own scatter across draws")

    print("\nthe axis increment: what zeroing the axis term costs at the final "
          "state, millinats")
    head = "".join(f"{f'seed {s}':>11}" for s in SEEDS)
    print(f"{'count':>12}{head}{'mean':>11}")
    for c in COUNTS:
        cells = "".join(f"{m['inc'][c][s]:>11.2f}" for s in SEEDS)
        print(f"{c:>12}{cells}{m['inc_mean'][c]:>11.2f}")
    print(f"  strictly rising with the count at every seed: "
          f"{'YES' if m['rises'] else 'NO'} — where it rises, the increment "
          f"read alone recommends the largest count tried, which is why the "
          f"combined score above is the criterion and this is not")

    print(f"\n  the shipped count's own wander INSIDE the window, per seed "
          f"(millinats, what the window mean averages down): "
          + ", ".join(f"s{s} {m['wander'][s]:.1f}" for s in SEEDS))
    print("\n  NOT measured here: the eight-axis arm (no rule to teach it "
          "with), §1.14's fourth holdout draw, and the per-rule "
          "half-reliability question, which is per-user identifiability rather "
          "than a likelihood.")


if __name__ == "__main__":
    main()
