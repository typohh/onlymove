#!/usr/bin/env python3
"""The global slope multiplier §1.10 declines to apply, read off the sweep every
fit already records.

§1.10's load-bearing sentence is that the best multiplier is close enough to the
shipped one that applying it buys nothing reliable: the optimum sits a little
below one, unity costs a fraction of a millinat against it, and that cost is
smaller than the spread between two training seeds. This producer measures those
three quantities and emits an indicator for the comparison between the last two.

NOT A PANEL OF ITS OWN. `pipeline.train_mirt.run()` records
`metrics["slope_scale_sweep"]` on every fit — a held-out score at each multiplier
over a fixed grid, rescoring the finished model rather than refitting it — so the
measurement rides on arms that already exist. This producer therefore trains
nothing and adds no arms; it reads every arm under `dev_runs/` whose recorded
configuration IS the shipped one and reads the sweep out of it.

THIS IS A CHANGE OF ESTIMAND FROM WHAT §1.10 SAYS, and nothing here may be
presented as a reproduction of that measurement. §1.10 says the sweep was scored
"against held-out users". The recorded sweep is scored against held-aside ROWS:
the user is in the fit and their ability comes from their OTHER attempts, and the
shipped held-aside unit is the row, so a stop row's problem is usually one
training saw as well. What is held out is the row, not the person and not the
position. That is arguably the more relevant quantity for this system — it is
exactly the position the client is in when it predicts one attempt from a history
it has already fitted an ability on — but it is a DIFFERENT quantity, and a
sentence in §1.10 that names held-out users cannot be backed by these figures
without being rewritten to say what was actually held out.

Two consequences of the same substitution, both worth stating because §1.10 leans
on them. The section's seen/unseen SPLIT is out of reach here: the recorded sweep
is pooled over all held-aside rows and carries no partition, so the argument that
pooling never-trained problems drags the optimum down cannot be re-measured from
these records. And the compression the multiplier would correct is a property of
how well the fit generalizes to POSITIONS, which a row split does not probe; a
row-split optimum can sit closer to one than a problem-split optimum would for
that reason alone.

WHICH ARMS ARE READ. Any arm under `dev_runs/<panel>/` whose recorded config is
empty or carries nothing but `seed` — those are independent runs of one
configuration at different seeds, which is exactly what the seed-spread half of
the claim needs. A config carrying anything else is a different experiment, a
budget override included, which is what keeps a throughput-calibration panel's
short arms out. An arm that restates a production default explicitly (a
`stop_unit` of the shipped value, say) is also excluded: the config is the
record's own statement of what it is, and admitting values that merely EQUAL a
default would need a defaults table here that could silently drift from
`train_mirt.run`'s signature.

ONE SEED IS ONE OBSERVATION however many panels ran it. Several panels carry an
arm of the shipped configuration at the same seed, and those runs are bitwise
identical (`refit_determinism_figures` is the standing check), so they are
deduplicated by seed rather than averaged as though they were independent. The
deduplication check earns its keep twice: the pins do NOT cover the training
hyperparameters, and the panels were run at different revisions, so two arms
labelled shipped at different `git` values could in principle be different
configurations — identical sweeps at one seed across two panels is the evidence
that nothing they turn on moved between those revisions, and a disagreement is
raised rather than averaged over.

CHECK THE BOUNDARY INDICATOR BEFORE QUOTING ANYTHING ELSE. A sweep whose minimum
sits on an end of the grid did not contain the optimum, so the optimum is only
"at least this far from one" and the cost of unity is a LOWER BOUND on the true
correction rather than a measurement of it. The fix is to widen the grid in
`pipeline/train_mirt.py` and re-run the arms — never to quote the boundary point
as the optimum.

A REFUTED CLAIM IS A MEASUREMENT, NOT A BROKEN INPUT. If unity turns out to cost
MORE than the seed spread, that is the finding, and the sentence resting on it is
what needs rewriting — so it is emitted as `slopescale.unity_inside_seed_spread`
rather than raised. Hard failure is kept for inputs: a missing sweep, a short
budget, a grid that differs across arms, pins that disagree, a duplicate pair
that is not a duplicate, or a panel gone stale against the attempt log.

  slopescale.seeds                  independent seeds behind everything below.
  slopescale.optimum_mult           the multiplier each arm's own sweep is lowest
  slopescale.optimum_mult_min       at — averaged over the arms, and the per-seed
  slopescale.optimum_mult_max       span it was averaged over.
  slopescale.unity_cost_mnat        what the SHIPPED multiplier costs against
                                    that optimum, millinats, averaged over arms.
  slopescale.seed_spread_mnat       the spread across arms of the held-out score
                                    at the shipped multiplier, millinats — the
                                    noise the cost above is compared against.
  slopescale.unity_inside_seed_spread  1 when that cost is smaller than that
                                    spread. Backs a fig-guard, not a span.
  slopescale.optimum_interior       1 when NO arm's optimum sits on an end of the
                                    swept grid. Read this one first.

    ./.venv/bin/python3 -m ablations.refit_slope_scale_figures
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
from utils.rating.model_constants import SEED  # noqa: E402

TIER = "refit"

DEV_RUNS = os.path.join(PROJECT_ROOT, "ablations", "dev_runs")

# The config keys an arm may carry and still BE the shipped configuration. The
# seed selects the initialization and the held-aside draw and turns nothing the
# model is made of, which is what makes two seeds two runs of one configuration
# rather than two configurations.
SHIPPED_CONFIG_KEYS = frozenset({"seed"})

# The pins that have to agree for these arms to be comparable at all. `git` is
# deliberately absent: the panels supplying these arms were run at different
# revisions, and requiring one revision across them would reject the whole set —
# what stands in its place is the bitwise duplicate check in `dedupe_by_seed`.
COMPARABLE_PINS = ("db_max_id", "gpu", "torch", "labels_sha", "proxies_sha",
                   "blocks", "pca_dim", "k_axes")

# The multiplier a fit actually ships, read from the arm's own written constants
# rather than restated here, so a fit that ever shipped something other than an
# identity multiplier would be measured at what it shipped instead of being
# scored against a number this file asserts.
SHIPPED_MULT_FILE = os.path.join("ratings", "rating_constants.json")


# --- the arms --------------------------------------------------------------

def load_shipped_arms():
    """Every arm under dev_runs whose config is the shipped one, as a list of
    dicts, plus the (budget, schedule) regime they all ran under.

    THE REGIME IS READ OFF THE ARMS, not off the trainer's current constants:
    these records are only ever compared to each other, so an arm measured under
    a retired baseline keeps its meaning and a producer pinned to the shipped
    budget would make every one of them unquotable the moment that constant
    moves. What is required instead is that the arms READ HERE share one regime,
    since the seed spread this producer prices is a spread between runs of one
    configuration and two regimes are two configurations. A mixed set raises
    rather than being narrowed to one regime: dropping arms quietly would leave
    the count of observations depending on what happened to be lying on disk.

    A SHORT SMOKE ARM never becomes a candidate by a different route: an arm run
    with `--epochs` (which is how `run_panel --smoke-epochs` overrides a whole
    panel) records the override in its CONFIG, so it is not this configuration
    at all — that is what keeps a throughput-calibration panel out."""
    require(os.path.isdir(DEV_RUNS),
            f"no {DEV_RUNS} — this producer reads the sweep out of panels that "
            f"have already run, and there are none")
    arms = []
    for panel in sorted(os.listdir(DEV_RUNS)):
        panel_dir = os.path.join(DEV_RUNS, panel)
        if not os.path.isdir(panel_dir):
            continue
        for entry in sorted(os.listdir(panel_dir)):
            if not entry.endswith(".json"):
                continue
            arm = entry[:-len(".json")]
            with open(os.path.join(panel_dir, entry)) as f:
                record = json.load(f)
            cfg = record.get("config", {})
            if set(cfg) - SHIPPED_CONFIG_KEYS:
                continue
            metrics = record["metrics"]
            arms.append({
                "panel": panel, "arm": arm, "name": f"{panel}/{arm}",
                "seed": int(cfg.get("seed", SEED)),
                "record": record, "metrics": metrics,
                "dir": os.path.join(panel_dir, arm),
            })
    require(arms,
            f"no arm under {DEV_RUNS} carries the shipped configuration (an "
            f"empty config, or one holding nothing but `seed`) — run a panel "
            f"that includes such an arm first")
    budget, schedule = panel_regime({a["name"]: a["record"] for a in arms})
    return arms, budget, schedule


def with_sweep(arms):
    """The arms that actually carry `slope_scale_sweep`, requiring at least two.

    An arm trained before the metric existed does not have it, and averaging
    over whichever arms happen to carry it would report an arbitrary subset as
    though it were the panel."""
    carry = [a for a in arms if a["metrics"].get("slope_scale_sweep")]
    have = {a["name"] for a in carry}
    missing = [a["name"] for a in arms if a["name"] not in have]
    require(len(carry) >= 2,
            f"{len(carry)} of {len(arms)} shipped-configuration arms carry "
            f"`slope_scale_sweep`; at least two are needed, one per seed, "
            f"because the seed spread this producer compares against does not "
            f"exist from a single fit. Without it: "
            f"{', '.join(missing) or '(none)'}. An arm trained before "
            f"`pipeline.train_mirt` recorded the sweep does not have it and "
            f"cannot acquire it without being re-run")
    return carry


def check_pins(arms):
    """The arms describe one snapshot on one machine, and that snapshot is still
    the current one."""
    base = arms[0]["record"]["pins"]
    for a in arms:
        pins = a["record"]["pins"]
        for pin in COMPARABLE_PINS:
            require(pin in pins and pin in base,
                    f"{a['name']} records no {pin} pin — a record that cannot "
                    f"say what it was measured against is not comparable")
            require(pins[pin] == base[pin],
                    f"{a['name']}'s pin {pin} ({pins[pin]!r}) differs from "
                    f"{arms[0]['name']}'s ({base[pin]!r}) — the arms are not "
                    f"runs of one configuration on one snapshot and their "
                    f"spread would measure the difference instead")
    for a in arms:
        m = a["metrics"]
        for field in ("rows", "users"):
            require(m[field] == arms[0]["metrics"][field],
                    f"{a['name']} fitted {field}={m[field]!r} against "
                    f"{arms[0]['name']}'s {arms[0]['metrics'][field]!r} — they "
                    f"did not see the same attempt log")
    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute(
            "SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"the arms were measured at attempt-log frontier "
            f"{base['db_max_id']} but the log now ends at {now} — the "
            f"held-aside draw moved, so these sweeps describe a snapshot that "
            f"no longer exists. Re-run the arms.")
    return base


def shipped_multiplier(arm):
    """What multiplier the arm's own run shipped, from the constants it wrote."""
    path = os.path.join(arm["dir"], SHIPPED_MULT_FILE)
    require(os.path.exists(path),
            f"{path} is missing — the shipped multiplier is read from the arm's "
            f"own written constants rather than asserted here, so an arm whose "
            f"artifacts were cleaned up cannot supply it")
    with open(path) as f:
        value = json.load(f).get("slope_scale")
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value),
            f"{path} has no numeric slope_scale")
    return float(value)


# --- the sweep -------------------------------------------------------------

def sweep_of(arm):
    """(multipliers, held-out BCE) from an arm's recorded sweep, checked into a
    shape the optimum and the boundary test can be read off."""
    rows = arm["metrics"]["slope_scale_sweep"]
    require(len(rows) >= 3,
            f"{arm['name']}'s sweep has {len(rows)} point(s) — an interior "
            f"minimum needs a grid with two ends and something between them")
    mults = [float(r["mult"]) for r in rows]
    bce = [float(r["stop_bce"]) for r in rows]
    require(all(a < b for a, b in zip(mults, mults[1:])),
            f"{arm['name']}'s sweep multipliers are not strictly increasing — "
            f"the ends of the grid are what the boundary test names, and an "
            f"unordered grid has none")
    require(all(math.isfinite(v) for v in bce),
            f"{arm['name']}'s sweep carries a non-finite held-out score")
    return mults, bce


def optimum(mults, bce):
    """(index, multiplier) of the sweep's lowest point. Ties break toward the
    smaller multiplier so the reading is a function of the record alone."""
    i = min(range(len(bce)), key=lambda j: (bce[j], mults[j]))
    return i, mults[i]


def index_of(mults, target):
    """Where a multiplier sits on the grid. The grid is written in hundredths,
    so the match is on the value rather than on a reconstructed index."""
    hits = [i for i, m in enumerate(mults) if abs(m - target) < 1e-9]
    require(len(hits) == 1,
            f"the sweep grid {mults[0]}..{mults[-1]} scores the multiplier "
            f"{target} {len(hits)} times — the cost of what ships cannot be "
            f"read off a grid that never scores it, or scores it twice")
    return hits[0]


def dedupe_by_seed(arms, sweeps):
    """One observation per seed. Duplicates must agree value for value: they are
    the same fit run twice, so a disagreement means the two records are not the
    same run and averaging them would hide it."""
    kept, groups = [], {}
    for a in arms:
        groups.setdefault(a["seed"], []).append(a)
    for seed in sorted(groups):
        members = groups[seed]
        first = members[0]
        for other in members[1:]:
            require(sweeps[other["name"]] == sweeps[first["name"]],
                    f"{other['name']} and {first['name']} both ran the shipped "
                    f"configuration at seed {seed} but recorded different "
                    f"sweeps — two runs of one configuration on one snapshot "
                    f"are bitwise identical, so something these arms turn on "
                    f"differs between the revisions they were run at, and they "
                    f"are not the same configuration")
        first["duplicates"] = [o["name"] for o in members[1:]]
        kept.append(first)
    require(len(kept) >= 2,
            f"every sweep-carrying arm ran at seed {kept[0]['seed']} "
            f"({', '.join([kept[0]['name']] + kept[0]['duplicates'])}) — they "
            f"are one fit recorded several times, and the spread between "
            f"training seeds cannot be read off a single seed")
    return kept


# --- the measurement -------------------------------------------------------

def measure():
    arms, budget, schedule = load_shipped_arms()
    arms = with_sweep(arms)
    pins = check_pins(arms)

    sweeps = {a["name"]: sweep_of(a) for a in arms}
    grid = sweeps[arms[0]["name"]][0]
    for a in arms[1:]:
        require(sweeps[a["name"]][0] == grid,
                f"{a['name']}'s sweep grid differs from {arms[0]['name']}'s — "
                f"the optima would not be comparable and the boundary test "
                f"would mean different things on different arms")

    shipped_mults = {a["name"]: shipped_multiplier(a) for a in arms}
    shipped = shipped_mults[arms[0]["name"]]
    for a in arms[1:]:
        require(shipped_mults[a["name"]] == shipped,
                f"{a['name']} shipped a slope multiplier of "
                f"{shipped_mults[a['name']]} against {arms[0]['name']}'s "
                f"{shipped} — the arms do not agree on what ships, so there is "
                f"no single multiplier to price")
    unity_i = index_of(grid, shipped)

    kept = dedupe_by_seed(arms, sweeps)

    per_arm = []
    for a in kept:
        mults, bce = sweeps[a["name"]]
        opt_i, opt_mult = optimum(mults, bce)
        per_arm.append({
            "name": a["name"], "seed": a["seed"],
            "duplicates": a["duplicates"],
            "mults": mults, "bce": bce,
            "opt_i": opt_i, "opt_mult": opt_mult, "opt_bce": bce[opt_i],
            "unity_bce": bce[unity_i],
            # BOTH SPREADS ARE PEAK-TO-PEAK RANGES OF THE SAME HELD-OUT SCORE,
            # in millinats, so the comparison below is like for like: this one
            # ranges over two scored multipliers within one arm, and the seed
            # spread ranges over the arms at one multiplier.
            "unity_cost_mnat": (bce[unity_i] - bce[opt_i]) * 1e3,
            "at_boundary": opt_i in (0, len(mults) - 1),
        })

    unity_cost = statistics.fmean(p["unity_cost_mnat"] for p in per_arm)
    unity_scores = [p["unity_bce"] for p in per_arm]
    seed_spread = (max(unity_scores) - min(unity_scores)) * 1e3
    return {
        "per_arm": per_arm,
        "grid": grid,
        "shipped_mult": shipped,
        "opt_mean": statistics.fmean(p["opt_mult"] for p in per_arm),
        "opt_min": min(p["opt_mult"] for p in per_arm),
        "opt_max": max(p["opt_mult"] for p in per_arm),
        "unity_cost": unity_cost,
        "seed_spread": seed_spread,
        "inside": int(unity_cost < seed_spread),
        "interior": int(not any(p["at_boundary"] for p in per_arm)),
        "pins": pins,
        "budget": budget,
        "schedule": schedule,
    }


ESTIMAND = ("scored on HELD-ASIDE ROWS, not held-out users: the user is in the "
            "fit and their ability comes from their other attempts, which is "
            "the position the client is in when it predicts an attempt from a "
            "history — a different quantity from §1.10's stated one")


def figures():
    m = measure()
    n = len(m["per_arm"])
    over = f"averaged over {n} independent seeds of the shipped configuration"
    return [
        Figure("slopescale.seeds", n, "d",
               "independent training seeds of the SHIPPED configuration whose "
               "recorded slope-multiplier sweep backs every figure below. One "
               "seed counts once however many panels ran it, those runs being "
               "bitwise identical"),
        Figure("slopescale.optimum_mult", m["opt_mean"], ".2f",
               f"the global slope multiplier at which the held-out score is "
               f"lowest, {over}. The multiplier pivots at the shipped rating, "
               f"so the 80%-clear anchor is held and only the steepness moves; "
               f"{ESTIMAND}"),
        Figure("slopescale.optimum_mult_min", m["opt_min"], ".2f",
               "the lowest per-seed optimum behind that average — the span the "
               "optimum itself moves over between fits differing only in the "
               "seed"),
        Figure("slopescale.optimum_mult_max", m["opt_max"], ".2f",
               "and the highest per-seed optimum"),
        Figure("slopescale.unity_cost_mnat", m["unity_cost"], ".2f",
               f"what the SHIPPED multiplier of {m['shipped_mult']:g} costs "
               f"against each arm's own optimum, in millinats of held-out "
               f"score, {over}. Non-negative by construction, the shipped "
               f"multiplier being one of the swept points; {ESTIMAND}"),
        Figure("slopescale.seed_spread_mnat", m["seed_spread"], ".2f",
               "the spread between training seeds of the held-out score AT the "
               "shipped multiplier, in millinats — highest arm minus lowest. "
               "Defined the same way as the cost above, both being a "
               "peak-to-peak range of the same held-out score in the same "
               "units: that one ranges over two multipliers within one arm, "
               "this one over the arms at one multiplier. Each seed selects "
               "its own held-aside rows as well as its own initialization, "
               "which is what 'between two training seeds' means here"),
        Figure("slopescale.unity_inside_seed_spread", m["inside"], "d",
               "1 when slopescale.unity_cost_mnat is strictly smaller than "
               "slopescale.seed_spread_mnat, 0 otherwise — §1.10's claim that "
               "the correction is smaller than the noise on the quantity it "
               "corrects, as a strict comparison of those two ledger values. "
               "An indicator for a `fig-guard: "
               "slopescale.unity_inside_seed_spread == 1`, never for a span"),
        Figure("slopescale.optimum_interior", m["interior"], "d",
               "1 when no arm's sweep is lowest at an END of the swept grid, 0 "
               "when any is. READ THIS BEFORE QUOTING THE OTHERS: a minimum on "
               "a boundary means the grid did not contain the optimum, and "
               "every figure above is then a lower bound on the correction "
               "rather than a measurement of it. The fix is to widen the grid "
               "in pipeline/train_mirt.py and re-run the arms, never to quote "
               "the boundary point"),
    ]


def main():
    m = measure()
    grid = m["grid"]
    step = grid[1] - grid[0]
    pins = m["pins"]
    print(f"slope-multiplier sweep, read off {len(m['per_arm'])} independent "
          f"seed(s) of the SHIPPED configuration")
    print(f"  grid {grid[0]:.2f}..{grid[-1]:.2f} in steps of {step:.2f} "
          f"({len(grid)} points), shipped multiplier {m['shipped_mult']:g}")
    print(f"  one machine: {pins['gpu']}, attempt log to id {pins['db_max_id']}")
    print(f"  all arms at one regime: {m['budget']} epochs, {m['schedule']} "
          f"rate")
    print("  the multiplier pivots at the shipped rating: the 80%-clear anchor")
    print("  is held and only the steepness around it moves.")
    print("  ESTIMAND: held-aside ROWS, not held-out users — each row's user is")
    print("  in the fit and their ability comes from their other attempts. This")
    print("  is not §1.10's stated measurement and must not be quoted as one.")

    print("\nper seed (score = held-out BCE on the held-aside rows)")
    print(f"{'arm':>22}{'seed':>6}{'optimum':>9}{'score@opt':>12}"
          f"{'score@ship':>12}{'ship cost':>11}{'':>4}")
    for p in m["per_arm"]:
        flag = "  END" if p["at_boundary"] else ""
        print(f"{p['name']:>22}{p['seed']:>6}{p['opt_mult']:>9.2f}"
              f"{p['opt_bce']:>12.6f}{p['unity_bce']:>12.6f}"
              f"{p['unity_cost_mnat']:>10.2f}m{flag}")
        if p["duplicates"]:
            print(f"{'':>22}  identical records not counted again: "
                  f"{', '.join(p['duplicates'])}")
    print("  'ship cost' is score@ship - score@opt in millinats; 'END' marks an")
    print("  optimum on a boundary of the grid, which would make it a bound.")

    print("\nsweep shape: millinats above each arm's OWN minimum")
    head = "".join(f"{'s' + str(p['seed']):>9}" for p in m["per_arm"])
    print(f"{'mult':>6}{head}")
    for i, mult in enumerate(grid):
        cells = "".join(f"{(p['bce'][i] - p['opt_bce']) * 1e3:>9.2f}"
                        for p in m["per_arm"])
        mark = "  <- shipped" if abs(mult - m["shipped_mult"]) < 1e-9 else ""
        print(f"{mult:>6.2f}{cells}{mark}")

    print(f"\noptimum multiplier      {m['opt_mean']:.2f}   "
          f"(per seed {m['opt_min']:.2f}..{m['opt_max']:.2f})")
    print(f"cost of the shipped one {m['unity_cost']:.2f} mnat  "
          f"(mean over seeds of score@ship - score@opt)")
    print(f"spread between seeds    {m['seed_spread']:.2f} mnat  "
          f"(max - min of score@ship across the seeds)")
    print(f"  both are peak-to-peak ranges of the same held-out score, in the "
          f"same units.")
    print(f"cost < spread: {'YES' if m['inside'] else 'NO'}  — §1.10's claim "
          f"that the correction is smaller than the noise on what it corrects")
    print(f"every optimum interior: {'YES' if m['interior'] else 'NO'}"
          + ("" if m["interior"] else
             "  — the grid did not contain the optimum; widen it in "
             "pipeline/train_mirt.py and re-run rather than quoting the end"))


if __name__ == "__main__":
    main()
