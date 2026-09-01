#!/usr/bin/env python3
"""The joint-PCA ACTIVATION WIDTH's figures (methodology §1.3), read from the
`features` refit panel: what a width costs held out, and the slope signature a
too-wide one carries.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` (`ablations.run_panel`, locally or via
`vast_orchestration.py ablate`) and hard-fails when the panel is missing, short
of budget, running a config other than the manifest's, internally inconsistent,
or STALE against the current attempt log — the pins rule `docs/mirt_benchmark.md`
states, mechanized. Guards and spans keep evaluating against the last-measured
ledger values, so the panel's cost is paid only when it is deliberately re-run.

THE SCORE IS ON HELD-OUT PROBLEMS, not on the shipped row split. Every arm runs
`stop_unit="problem"`, so a held-aside problem contributes no row to training at
all and its parameters come from the feature trunk alone — the only split that
can score a claim about the FEATURE side. The row split that ships is a
convergence detector: it leaves every problem partly trained, so a width wide
enough to fingerprint individual problems is rewarded by it rather than caught.

THE HEADLINE READING IS THE MEAN OVER THE LAST THIRD OF THE BUDGET. The shipped
configuration's own held-out trace wanders by more than ten millinats across the
last third of its run, which is wider than the contrasts this panel is asked
about, so a single epoch's value is not a stable estimate of where an arm sits.
The final-epoch reading is computed and printed beside it, as the spread one
checkpoint carries, but it carries no key of its own: a second key for the same
contrast is an invitation to quote the less stable one.

PAIRING AND SEEDS. Within one seed every arm holds aside the same problems, so a
cross-arm delta is paired, and that is the quantity below. The two seeds hold
aside DIFFERENT problems, so no cross-seed delta exists here and none is taken:
a contrast is formed inside a seed and the seeds are averaged afterwards. What
the second seed buys is a second independent estimate of the same paired delta,
and each contrast carries a 0/1 indicator saying whether the seeds agreed in
SIGN. That agreement stands in for §1.3's problem-clustered paired bootstrap,
which this panel does not compute.

A REFUTED DIRECTION IS REPORTED, NOT RAISED. Should a swept width beat the
shipped one, the indicators come back 0 and the costs come back signed; the
fig-guards beside §1.3's sentences are what fire, because it is then the
sentence and not the number that has to change.

  features.pca<W>_holdout_bce      held-out-PROBLEM cross-entropy at width W,
                                   nats per attempt, last-third mean, averaged
                                   over the two seeds. Descriptive only: the
                                   seeds hold out different problems, so this
                                   column is a level and never a comparison.
  features.pca<W>_cost_mnat        width W against the shipped 128, millinats,
                                   PAIRED inside a seed and then averaged over
                                   the seeds. Signed, positive = worse than 128.
  features.pca<W>_cost_sign_consistent
                                   1 only when both seeds put that contrast on
                                   the same side of zero, on the headline
                                   reading. For a `<!-- fig-guard: KEY == 1 -->`,
                                   not a span. Held to the headline reading
                                   alone — unlike the barrier panel's, where the
                                   final epoch carries a key of its own — because
                                   an indicator this module's own figures depend
                                   on cannot rest on a reading it declines to
                                   quote. The final-epoch readings' agreement is
                                   printed beside it as context.
  features.pca128_wins_both_sides  1 only when 128 beats BOTH its neighbours in
                                   the swept set (64 below, 256 above) at every
                                   seed — §1.3's "wins from both sides".
  features.pca<W>_slope_median     median fitted per-problem slope at width W,
                                   averaged over the seeds.
  features.pca256_highest_slope    1 only when 256 carries the highest median
                                   slope of any arm at every seed — the
                                   memorization signature §1.3 names.

WHAT THIS PANEL CANNOT SETTLE. §1.3's sweep was end to end — each width
re-rated, re-mapped, re-fit and re-axed — and this panel stops at the fit: no
rank mapping is re-derived and no motif count is re-counted, so the claim that 64
collapses the motif count to its floor has no figure here and must not be read
off one. The panel also runs two seeds against the document's three, and does not
run the widths 8 and 32 at all; the claim's two ends are represented by 16 and
256.

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

from ablations._figures import Figure, require, panel_budget  # noqa: E402
from pipeline.train_mirt import EPOCHS  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.model_constants import PCA_DIM, PRODUCTION_BLOCKS  # noqa: E402

TIER = "refit"

PANEL = "features"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")

# The swept widths, ascending — the order "wins from both sides" is read along.
# 8 and 32 are deliberately absent: the claim's two ends are represented by the
# widths at either end of this list.
WIDTHS = (16, 64, 128, 256)
SHIPPED_WIDTH = 128
CONTRASTS = tuple(w for w in WIDTHS if w != SHIPPED_WIDTH)
# 128's neighbours IN THE SWEPT SET — the pair "from both sides" is about here.
NEIGHBOURS = tuple(w for w in (64, 256) if w in WIDTHS)
# The width whose median slope §1.3 calls the memorization signature.
MEMORIZING_WIDTH = 256
SEEDS = (0, 1)
# Every arm scores on held-out PROBLEMS: the claim is about the feature side,
# and the shipped row split leaves every problem partly trained.
STOP_UNIT = "problem"

# The last third of the budget: past the stretch where these fits still improve,
# so a mean over it reads the level an arm settled at rather than whichever point
# of its wander the final checkpoint happened to land on. Same window as the
# other refit producers, so their tail readings mean the same thing.
TAIL_FROM_EPOCH = 2 * EPOCHS // 3


def arm_name(width, seed):
    """The manifest's name for one (width, seed) arm. The shipped width names
    itself, so the baseline every contrast is taken against is obvious in a
    directory listing and in a failure message."""
    return f"{'shipped' if width == SHIPPED_WIDTH else 'w'}{width}_s{seed}"


ARMS = {(w, s): arm_name(w, s) for w in WIDTHS for s in SEEDS}


# --- the panel record ---------------------------------------------------------

def load_panel():
    """Every arm's record, checked for what makes this panel quotable: the
    production budget, the manifest's own config, the held-out-problem split,
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

    arms = {}
    for key, arm in ARMS.items():
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the features panel has not run at every width and "
                f"seed (missing {arm}: width {key[0]}, seed {key[1]}). Run it "
                f"with `python -m ablations.run_panel features` (or "
                f"`vast_orchestration.py ablate features`) first; the resume "
                f"unit is the arm, so arms already recorded are skipped.")
        with open(path) as f:
            arms[key] = json.load(f)

    budget = panel_budget(arms)
    for key, r in arms.items():
        arm, (width, seed) = ARMS[key], key
        require(r["metrics"]["epochs"] == budget,
                f"{arm} ran {r['metrics']['epochs']} epochs against the panel's "
                f"{budget} — one panel is one budget")
        require(r["metrics"]["trace"]
                and r["metrics"]["trace"][-1]["epoch"] == EPOCHS,
                f"{arm}'s convergence trace does not reach its budget")
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
                    "blocks", "k_axes"):
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
    return arms


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
    arms = load_panel()
    curves = {key: stop_curve(r) for key, r in arms.items()}
    grid = curves[(SHIPPED_WIDTH, SEEDS[0])][0]
    for key, (ep, _) in curves.items():
        require(ep == grid,
                f"{ARMS[key]}'s trace is scored at different epochs than the "
                f"baseline's — the deltas would not be like for like")
    tail = [i for i, e in enumerate(grid) if e >= TAIL_FROM_EPOCH]
    require(len(tail) > 1,
            f"the trace carries {len(tail)} checkpoint(s) at or past epoch "
            f"{TAIL_FROM_EPOCH} — the settled level and the wander it averages "
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

    # "WINS FROM BOTH SIDES": the shipped width beats the width below it AND the
    # width above it, at every seed — one criterion over the whole panel, never
    # an average that one seed could carry.
    wins_both = int(all(level[(SHIPPED_WIDTH, s)] < level[(w, s)]
                        for s in SEEDS for w in NEIGHBOURS))

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
    # The memorization signature is a per-seed maximum held to every seed: a
    # mean over seeds could be carried by one arm alone.
    top_slope = {s: max(WIDTHS, key=lambda w: slope[(w, s)]) for s in SEEDS}
    highest_slope = int(all(top_slope[s] == MEMORIZING_WIDTH for s in SEEDS))

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
        "level": level, "final": final, "contrasts": contrasts,
        "holdout_bce": {w: statistics.fmean(level[(w, s)] for s in SEEDS)
                        for w in WIDTHS},
        "wins_both": wins_both,
        "slope": slope,
        "slope_mean": {w: statistics.fmean(slope[(w, s)] for s in SEEDS)
                       for w in WIDTHS},
        "top_slope": top_slope, "highest_slope": highest_slope,
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
            f"held-out-PROBLEM cross-entropy at a joint-PCA activation width of "
            f"{w}, nats per attempt, meaned over the last third of the budget "
            f"and then {over}. A level, not a comparison: the seeds hold aside "
            f"different problems"))
    for w in CONTRASTS:
        c = m["contrasts"][w]
        out.append(Figure(
            f"features.pca{w}_cost_mnat", c["tail_mean"], ".1f",
            f"held-out-problem cost of a width of {w} against the shipped "
            f"{SHIPPED_WIDTH}, millinats over the last third of the budget, "
            f"signed (positive = worse than the shipped width). PAIRED inside a "
            f"seed — both arms hold aside the same problems — and then {over}"))
        out.append(Figure(
            f"features.pca{w}_cost_sign_consistent", c["sign_consistent"], "d",
            f"1 when both seeds put the {w}-against-{SHIPPED_WIDTH} contrast on "
            f"the same side of zero on the last-third reading, 0 when they "
            f"disagree — the two seeds being independent estimates of one paired "
            f"delta. An indicator for a fig-guard beside a sentence naming that "
            f"contrast's direction, not for a span"))
    out.append(Figure(
        "features.pca128_wins_both_sides", m["wins_both"], "d",
        f"1 when the shipped width scores better held out than BOTH its "
        f"neighbours in the swept set ({NEIGHBOURS[0]} below, {NEIGHBOURS[1]} "
        f"above) at every seed, on the last-third reading — §1.3's 'wins from "
        f"both sides' as one indicator"))
    for w in WIDTHS:
        out.append(Figure(
            f"features.pca{w}_slope_median", m["slope_mean"][w], ".2f",
            f"median fitted per-problem slope at a width of {w}, over every "
            f"fitted problem, {over}"))
    out.append(Figure(
        f"features.pca{MEMORIZING_WIDTH}_highest_slope", m["highest_slope"], "d",
        f"1 when the {MEMORIZING_WIDTH}-wide arm carries the highest median "
        f"slope of any arm at every seed — the memorization signature §1.3 "
        f"names, as an indicator"))
    return out


# --- the table ----------------------------------------------------------------

def main():
    m = measure()
    lo, hi, n = m["tail_epochs"]
    print(f"\nfeatures panel — methodology §1.3, the joint-PCA activation width")
    print(f"{len(WIDTHS)} widths x {len(SEEDS)} seeds {list(SEEDS)}, full budget "
          f"({EPOCHS} epochs), one machine ({m['gpu']}),")
    print(f"{m['rows']:,d} attempt rows per arm. Scored on HELD-OUT PROBLEMS "
          f"(stop_unit={STOP_UNIT!r}): a held problem's")
    print(f"parameters come from the feature trunk alone, which is what a width "
          f"is judged on. Held aside: "
          + ", ".join(f"s{s} {m['stop_rows'][s]:,d} rows" for s in SEEDS) + ".")
    print(f"'tail' is the mean over the {n} checkpoints from epoch {lo} to "
          f"{hi}; 'final' is epoch {EPOCHS} alone.")

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
          f"over every fitted problem.")

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

    print(f"\n  {SHIPPED_WIDTH} beats both its swept neighbours "
          f"({NEIGHBOURS[0]} and {NEIGHBOURS[1]}) at every seed: "
          f"{'YES' if m['wins_both'] else 'NO'}")
    print(f"  highest median slope: "
          + ", ".join(f"seed {s} -> {m['top_slope'][s]}" for s in SEEDS)
          + f"   ({MEMORIZING_WIDTH} highest at every seed: "
          + f"{'YES' if m['highest_slope'] else 'NO'})")

    print(f"\nthe shipped width's own held-out wander over epochs {lo}..{hi} — "
          f"why the tail reading is the headline")
    for s in SEEDS:
        print(f"{'seed ' + str(s):>12}{m['wander'][s]['sd']:>9.2f} mnat sd"
              f"{m['wander'][s]['range']:>10.2f} mnat peak-to-peak")
    print(f"  A contrast read at one checkpoint carries this spread on top of "
          f"itself.")

    print(f"\nOut of this panel's reach: §1.3's sweep was end to end and these "
          f"arms stop at the fit — no rank")
    print(f"mapping is re-derived and no motif count re-counted, so 64's motif "
          f"collapse is not measured here.")
    print(f"Widths 8 and 32 are not run; the claim's ends are represented by "
          f"{WIDTHS[0]} and {WIDTHS[-1]}.\n")


if __name__ == "__main__":
    main()
