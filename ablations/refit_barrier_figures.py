#!/usr/bin/env python3
"""The divergence barrier's figures (methodology §1.7), read from the `barrier`
refit panel — the first refit-tier producer.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` (`ablations.run_panel`, locally or via
`vast_orchestration.py ablate`) and hard-fails when the panel is missing, short
of budget, running a config other than the manifest's, internally inconsistent,
or STALE against the current attempt log — the pins rule `docs/mirt_benchmark.md`
states, mechanized. Guards and spans keep evaluating against the last-measured
ledger values, so the panel's cost is paid only when it is deliberately re-run.

EVERY CONFIGURATION RUNS AT TWO SEEDS, and this producer is built around that.
The shipped configuration's own held-out trace wanders across the last third of
its budget by more than the contrasts this panel is asked about, so a single arm
per configuration resolves none of them — `barrier.shipped_wander_sd_mnat` is
that resolution floor, measured here rather than asserted. The seed drives the
initialization AND the held-out draw, so WITHIN one seed every arm holds aside
the same stop rows and a cross-arm delta is paired, while the two seeds are two
independent paired estimates of the same contrast. Nothing below ever subtracts
across seeds: a contrast is taken inside a seed, and the seeds are averaged
afterwards with their SIGN agreement reported separately.

EVERY CONTRAST IS READ TWICE. The FINAL epoch is the fit that would actually
ship; the mean over the last third is the level the arm settled at, which the
final checkpoint only samples at one point of the wander. Neither reading is a
summary of the other, so both carry a key and the tail one is the headline.

BOTH VARIANTS ARE SCORED HELD-OUT, the ten-times arm included. §1.7 chooses the
weight by held-out sweep and names ten times as the failure direction, so where
that arm actually lands is part of the claim: a document cannot rest on a sweep
whose losing arm was never scored, whichever way it falls.

A REFUTED CLAIM IS A MEASUREMENT, NOT A BROKEN INPUT. If removing the barrier
turns out to cost nothing held-out, that is the finding this panel exists to
make and the sentence quoting it is what needs rewriting — so it is emitted as
`barrier.off_stop_cost_worse` rather than raised. Hard failure is kept for
inputs: a record missing, short of budget, mismatched against the manifest,
unpaired inside its seed, or stale against the attempt log.

  barrier.off_rating_low / _high   the 0.1th / 99.9th percentile of the derived
                                   80%-clear rating with the barrier removed —
                                   the blow-out §1.7 describes.
  barrier.off_slope_at_floor_pct   share of slopes on the clamp without it.
  barrier.off_stop_tail_cost_mnat  the barrier-off arm's held-out (stop-split)
                                   cost against the shipped arm of its own seed,
                                   over the last third of the budget, in
                                   millinats — the headline cost.
  barrier.off_stop_cost_mnat       the same cost read at the final epoch.
  barrier.off_stop_cost_worse      1 when removing the barrier is worse
                                   held-out at every seed on both readings — the
                                   direction §1.7's claim rests on, as an
                                   indicator rather than an abort.
  barrier.off_stop_cost_sign_consistent
                                   1 only when every seed agrees in sign on BOTH
                                   readings — an indicator meant for a
                                   `<!-- fig-guard: KEY == 1 -->` beside a
                                   sentence that names the direction of that
                                   cost, never for a span.
  barrier.tenx_rating_span         the p0.1..p99.9 rating span at ten times the
                                   barrier weight — the compression failure the
                                   weight is swept between.
  barrier.tenx_stop_tail_cost_mnat  the ten-times arm's held-out cost against
  barrier.tenx_stop_cost_mnat      the shipped arm, the same two readings — the
                                   losing end of the sweep, scored.
  barrier.tenx_cost_sign_consistent  the same 0/1 seed agreement for that pair.
  barrier.shipped_wander_sd_mnat   the shipped arm's own held-out spread across
  barrier.shipped_wander_range_mnat  the last third of ITS OWN run — the floor
                                   every contrast above is read against.

    ./.venv/bin/python3 -m ablations.refit_barrier_figures
"""

import json
import math
import csv
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
from utils.rating.fit_rank_curves import CALIBRATION_P  # noqa: E402
from utils.rating.model_constants import SEED, SIZES  # noqa: E402

TIER = "refit"

PANEL = "barrier"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")
# One record per CONFIGURATION per seed. The manifest names a replicate by
# suffixing the configuration it repeats, which is what lets every delta be
# matched to the baseline of its own seed.
CONFIGS = ("shipped", "barrier_off", "barrier_10x")
BASELINE = CONFIGS[0]


def replicate_suffixes(manifest):
    """The seed suffixes the manifest actually declares, in seed order.

    Read rather than hardcoded: a panel gains a replicate whenever two seeds
    fail to settle something, and a producer pinned to a fixed list would abort
    on the panel getting BETTER. Every configuration must carry every suffix —
    an unbalanced panel would silently average different configurations over
    different seeds."""
    suffixes = sorted({arm[len(BASELINE):] for arm in manifest["arms"]
                       if arm.startswith(BASELINE)})
    require(suffixes, f"{MANIFEST} declares no {BASELINE} arm to key the "
                      f"replicates off")
    want = {c + suf for suf in suffixes for c in CONFIGS}
    require(set(manifest["arms"]) == want,
            f"{MANIFEST} declares {sorted(manifest['arms'])}, which is not "
            f"every one of {sorted(CONFIGS)} at every one of {suffixes} — an "
            f"unbalanced panel would average different configurations over "
            f"different seeds")
    return tuple(suffixes)

# The last third of the budget: past the stretch where these fits still improve,
# so a mean over it reads the level an arm settled at rather than whichever
# point of its wander the final checkpoint happened to land on.
TAIL_FROM_EPOCH = 2 * EPOCHS // 3


# --- the panel -------------------------------------------------------------

def load_panel():
    require(os.path.exists(MANIFEST),
            f"no {MANIFEST} — the panel manifest is what the arm records are "
            f"checked against, so it cannot be read without it")
    manifest = json.load(open(MANIFEST))
    suffixes = replicate_suffixes(manifest)
    arm_names = tuple(c + suf for suf in suffixes for c in CONFIGS)

    arms = {}
    for arm in arm_names:
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the barrier panel has not run at every seed. Run "
                f"it with `python -m ablations.run_panel barrier` (or "
                f"`vast_orchestration.py ablate barrier`) first; the resume "
                f"unit is the arm, so the arms already recorded are skipped.")
        arms[arm] = json.load(open(path))

    budget = panel_budget(arms)
    for arm, r in arms.items():
        require(r["metrics"]["epochs"] == budget,
                f"{arm} ran {r['metrics']['epochs']} epochs against the panel's "
                f"{budget} — one panel is one budget")
        require(r["metrics"]["trace"]
                and r["metrics"]["trace"][-1]["epoch"] == EPOCHS,
                f"{arm}'s convergence trace does not reach its budget")
        # The record has to be the arm this producer thinks it is: an arm run by
        # hand with a different flag is a well-formed record of a different
        # experiment, and every figure below would be attributed to the wrong
        # configuration.
        require(r["config"] == manifest["arms"][arm],
                f"{arm} was run with {r['config']!r}, but the manifest declares "
                f"{manifest['arms'][arm]!r} — the record is not this arm")
        require(all(math.isfinite(float(t["stop_bce"]))
                    for t in r["metrics"]["trace"]),
                f"{arm}'s held-out trace is not finite throughout. This panel "
                f"carries no divergence machinery: a blown-up arm is a "
                f"different finding from the ones below, not a larger version "
                f"of them")

    # The SEED is what the replicates vary, and no pin carries it (see
    # `ablation_mirt.snapshot_pins`), so this loop spans every arm of both
    # replicates unchanged — everything it names has to hold across the whole
    # panel for any of it to be one panel.
    base = arms[BASELINE]["pins"]
    for arm, r in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha",
                    "blocks", "pca_dim", "k_axes"):
            require(r["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} ({r['pins'][pin]!r}) differs from the "
                    f"shipped arm's ({base[pin]!r}) — the arms are not one "
                    f"panel and their comparison is void")

    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute("SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"the panel was measured at attempt-log frontier {base['db_max_id']} "
            f"but the log now ends at {now} — the stop-row draw moved, the "
            f"paired comparison is void. Re-run the panel.")
    return arms, seed_groups(arms, suffixes)


def seed_groups(arms, suffixes):
    """The panel regrouped into replicates: one entry per seed, holding that
    seed's arm name for every configuration.

    A group is ONE held-out draw. The seed selects the stop rows, so a delta
    between two arms of one group is paired; two groups are independent
    estimates of that delta under different rows and different inits, which is
    why they are only ever averaged and never subtracted."""
    groups = []
    for suffix in suffixes:
        members = {cfg: cfg + suffix for cfg in CONFIGS}
        seeds = {int(arms[a]["config"].get("seed", SEED))
                 for a in members.values()}
        require(len(seeds) == 1,
                f"the arms {sorted(members.values())} were run at seeds "
                f"{sorted(seeds)} — a replicate is one seed across every "
                f"configuration, and a delta taken inside it would not be "
                f"paired")
        groups.append({"seed": seeds.pop(), "arms": members})
    require(len({g["seed"] for g in groups}) == len(groups),
            f"every replicate ran at seed {groups[0]['seed']} — they are the "
            f"same fit repeated, and their agreement measures nothing")

    # The pairing itself, checked rather than assumed: one seed at one attempt
    # log gives one stop split, and equal row / user / stop-row counts are what
    # that identity looks like from the record.
    for g in groups:
        ship = arms[g["arms"][BASELINE]]["metrics"]
        for arm in g["arms"].values():
            for field in ("rows", "users", "stop_rows", "base_rate"):
                require(arms[arm]["metrics"][field] == ship[field],
                        f"{arm} fitted {field}="
                        f"{arms[arm]['metrics'][field]!r} against seed "
                        f"{g['seed']}'s shipped arm's {ship[field]!r} — they do "
                        f"not hold out the same rows, so the delta between them "
                        f"is not paired")
    return groups


# --- reading a convergence trace -------------------------------------------

def stop_curve(record):
    """(epochs, held-out BCE) from an arm's convergence trace."""
    trace = record["metrics"]["trace"]
    return ([int(r["epoch"]) for r in trace],
            [float(r["stop_bce"]) for r in trace])


def deltas_mnat(arm_bce, ship_bce, tail):
    """One arm's held-out cost against the shipped arm of its OWN seed, in
    millinats, at the final checkpoint and as a mean over the tail checkpoints.
    Positive = worse than shipped. Both are differences taken inside one seed,
    where the two arms hold aside the same rows, so both are paired."""
    final = (arm_bce[-1] - ship_bce[-1]) * 1e3
    tail_mean = (statistics.fmean(arm_bce[i] for i in tail)
                 - statistics.fmean(ship_bce[i] for i in tail)) * 1e3
    return final, tail_mean


def same_sign(*series):
    """1 when every seed's reading lands on the same side of zero in each of the
    given series, 0 otherwise. A reading of exactly zero agrees with nothing, so
    it is never consistent."""
    return int(all(all(v > 0 for v in s) or all(v < 0 for v in s)
                   for s in series))


def floors_above_calibration(arm):
    """Problems whose fitted floor reaches the calibration probability, counted
    from the arm's own written floors. Read from the artifact and not from the
    metrics block: this is a tail count, and a distribution's percentiles do not
    carry it."""
    n = 0
    for size in SIZES:
        path = os.path.join(PANEL_DIR, arm, "ratings", str(size),
                            "tsumego_floors.csv")
        require(os.path.exists(path),
                f"{path} is missing — the arm's written floors are what the "
                f"undefined-rating count is read from, and an arm whose "
                f"artifacts were cleaned up cannot supply it")
        with open(path) as f:
            n += sum(1 for row in csv.DictReader(f)
                     if float(row["floor"]) >= CALIBRATION_P)
    return n


# --- the measurement -------------------------------------------------------

def measure():
    arms, groups = load_panel()
    curves = {arm: stop_curve(r) for arm, r in arms.items()}
    grid = curves[BASELINE][0]
    for arm, (ep, _) in curves.items():
        require(ep == grid,
                f"{arm}'s trace is scored at different epochs than the shipped "
                f"arm's — the deltas would not be like for like")
    tail = [i for i, e in enumerate(grid) if e >= TAIL_FROM_EPOCH]
    require(len(tail) > 1,
            f"the trace carries {len(tail)} checkpoint(s) at or past epoch "
            f"{TAIL_FROM_EPOCH} — the settled level and the wander around it "
            f"are read over the last third of the budget, and neither exists "
            f"on a single point")

    # Per-fit properties: read on every arm, then averaged over the replicates.
    def per_fit(cfg, pick):
        return [pick(arms[g["arms"][cfg]]["metrics"]) for g in groups]

    dist = {}
    for cfg in CONFIGS:
        d = {"low": per_fit(cfg, lambda m: m["rating_pct"]["0.1"]),
             "high": per_fit(cfg, lambda m: m["rating_pct"]["99.9"]),
             "floor_share": per_fit(cfg, lambda m: m["slope_at_floor_share"])}
        d["span"] = [hi - lo for lo, hi in zip(d["low"], d["high"])]
        # HOW MANY PROBLEMS END WITH NO DEFINED RATING. A floor at or above the
        # calibration probability leaves the 80%-clear point with no referent,
        # and the count is read from the arm's own written floors rather than
        # interpolated out of percentiles, because at this rarity a percentile
        # says nothing about the count. It is the one consequence of a heavier
        # barrier that the held-out likelihood cannot show, since the affected
        # problems ship a sentinel rather than a wrong number.
        d["no_rating"] = [floors_above_calibration(g["arms"][cfg])
                          for g in groups]
        dist[cfg] = d

    # Contrasts: taken inside a seed, one reading per seed, in seed order.
    contrasts = {}
    for cfg in CONFIGS:
        if cfg == BASELINE:
            continue
        finals, tails = [], []
        for g in groups:
            _, ship_bce = curves[g["arms"][BASELINE]]
            _, arm_bce = curves[g["arms"][cfg]]
            final, tail_mean = deltas_mnat(arm_bce, ship_bce, tail)
            finals.append(final)
            tails.append(tail_mean)
        contrasts[cfg] = {
            "final": finals, "tail": tails,
            "final_mean": statistics.fmean(finals),
            "tail_mean": statistics.fmean(tails),
            "sign_consistent": same_sign(finals, tails),
        }

    # The resolution floor: what the SHIPPED configuration does on its own
    # across the same window every contrast above is read over.
    wander = []
    for g in groups:
        _, ship_bce = curves[g["arms"][BASELINE]]
        tv = [ship_bce[i] * 1e3 for i in tail]
        wander.append({"seed": g["seed"], "sd": statistics.stdev(tv),
                       "range": max(tv) - min(tv)})

    off = contrasts["barrier_off"]
    require(off["final_mean"] > 0 and off["tail_mean"] > 0,
            f"the barrier-off arm is not worse held-out "
            f"({off['final_mean']:+.2f} mnat at the final epoch, "
            f"{off['tail_mean']:+.2f} over the last third, meaned over "
            f"{len(groups)} seeds) — the barrier's stated purpose is refuted, "
            f"not just its magnitude")
    return {
        "seeds": [g["seed"] for g in groups],
        "tail_epochs": (grid[tail[0]], grid[tail[-1]], len(tail)),
        "dist": dist,
        "mean_dist": {cfg: {k: statistics.fmean(v) for k, v in d.items()}
                      for cfg, d in dist.items()},
        "contrasts": contrasts,
        "wander": wander,
        # HOW REPRODUCIBLE THE BLOW-OUT'S SIZE IS, as distinct from whether it
        # happens. The barrier-off arm's high percentile is the loudest number
        # this panel produces and it is also the least stable: without this a
        # reader re-running the panel would read a several-fold difference as a
        # discrepancy to chase rather than as the quantity behaving the way an
        # unbounded tail behaves.
        "off_high_seed_ratio": (max(dist["barrier_off"]["high"])
                                / max(min(dist["barrier_off"]["high"]), 1e-9)),
        "wander_sd": statistics.fmean(w["sd"] for w in wander),
        "wander_range": statistics.fmean(w["range"] for w in wander),
        "gpu": arms[BASELINE]["pins"]["gpu"],
    }


def figures():
    m = measure()
    off_dist = m["mean_dist"]["barrier_off"]
    off = m["contrasts"]["barrier_off"]
    over = f"averaged over the panel's {len(m['seeds'])} seeds"
    return [
        Figure("barrier.off_rating_low", off_dist["low"], ".0f",
               "0.1th percentile of the derived rating with the slope barrier "
               f"removed — the low side of the blow-out. A per-fit property, "
               f"{over}"),
        Figure("barrier.off_rating_high", off_dist["high"], ".0f",
               f"…and the 99.9th percentile, the high side; {over}"),
        Figure("barrier.off_slope_at_floor_pct", off_dist["floor_share"],
               "%.0f",
               f"share of slopes on the clamp with the barrier removed, {over}"),
        Figure("barrier.off_stop_tail_cost_mnat", off["tail_mean"], ".0f",
               "held-out (stop-split) cost of removing the barrier over the "
               "last third of the budget, millinats — PAIRED against the "
               "shipped arm of its own seed, then meaned over the seeds. The "
               "headline reading: it is the level the arm settled at, which a "
               "final checkpoint alone samples only once"),
        Figure("barrier.off_stop_cost_mnat", off["final_mean"], ".0f",
               "the same paired held-out cost read at the FINAL epoch — the fit "
               f"that would actually ship, {over}"),
        Figure("barrier.off_stop_cost_sign_consistent", off["sign_consistent"],
               "d",
               "1 when every seed puts the barrier-off cost on the same side of "
               "zero on BOTH readings, 0 when the seeds disagree. An indicator "
               "for a `fig-guard: barrier.off_stop_cost_sign_consistent == 1` "
               "beside any sentence naming the direction of that cost, never "
               "for a span"),
        Figure("barrier.tenx_rating_span", m["mean_dist"]["barrier_10x"]["span"],
               ".1f",
               "p0.1..p99.9 span of the derived rating at ten times the barrier "
               "weight — the compression the weight is swept against. A per-fit "
               f"property, {over}"),
        Figure("barrier.tenx_stop_tail_cost_mnat",
               m["contrasts"]["barrier_10x"]["tail_mean"], ".0f",
               "held-out cost of TEN TIMES the barrier weight over the last "
               "third of the budget, millinats, paired within a seed and then "
               f"meaned, {over}. §1.7 chooses the weight by held-out sweep and "
               f"names ten times as the failure direction, so where that arm "
               f"lands is part of the claim: negative means the heavier barrier "
               f"fits BETTER than what ships"),
        Figure("barrier.tenx_stop_cost_mnat",
               m["contrasts"]["barrier_10x"]["final_mean"], ".0f",
               f"the same cost read at the FINAL epoch, {over}"),
        Figure("barrier.tenx_cost_sign_consistent",
               m["contrasts"]["barrier_10x"]["sign_consistent"], "d",
               "1 when every seed puts the ten-times cost on the same side of "
               "zero on both readings. An indicator for a fig-guard, not a span"),
        Figure("barrier.shipped_rating_span", m["mean_dist"]["shipped"]["span"],
               ".1f",
               "p0.1..p99.9 span of the derived rating as SHIPPED, the same "
               f"per-fit property the two arms above are compared against, "
               f"{over} — without it neither blow-out nor compression has a "
               f"reference"),
        Figure("barrier.shipped_no_rating_min",
               float(min(m["dist"]["shipped"]["no_rating"])), "d",
               "fewest problems left with no defined rating — a floor at or "
               "above the calibration probability — across the panel's runs of "
               "the SHIPPED configuration"),
        Figure("barrier.shipped_no_rating_max",
               float(max(m["dist"]["shipped"]["no_rating"])), "d",
               "and the most, in a run differing only in the seed. The pair is "
               "the figure: this count is a property of a particular fit and "
               "not of the model, so it is re-read after a refit rather than "
               "carried"),
        Figure("barrier.tenx_no_rating_sign_consistent",
               1 if len({a > b for a, b in zip(
                   m["dist"]["barrier_10x"]["no_rating"],
                   m["dist"]["shipped"]["no_rating"])}) == 1 else 0, "d",
               "1 when the heavier barrier moves the undefined-rating count the "
               "same way at every seed, 0 when it does not. A 0 says this count "
               "does not establish a cost for the heavier weight — the count is "
               "unstable under BOTH weights, so a difference between two single "
               "runs of them is not a difference between the weights"),
        Figure("barrier.tenx_no_rating_mean",
               statistics.fmean(m["dist"]["barrier_10x"]["no_rating"]), "~10",
               "the same count at ten times the barrier weight, seed-averaged "
               "— the one cost of a heavier barrier the held-out likelihood "
               "cannot show, since the problems it touches ship a sentinel "
               "rather than a wrong number"),
        Figure("barrier.off_rating_high_seed_ratio", m["off_high_seed_ratio"],
               ".0f",
               "how many times the barrier-off arm's 99.9th rating percentile "
               "moved between the panel's seeds — the blow-out HAPPENS every "
               "time, while its size is not itself a reproducible quantity, "
               "which is what an unbounded tail does"),
        Figure("barrier.shipped_wander_sd_mnat", m["wander_sd"], ".1f",
               "standard deviation of the SHIPPED arm's own held-out score "
               "across the last third of its budget, millinats, averaged over "
               "the seeds — the panel's resolution floor. No contrast smaller "
               "than this is resolved by one arm per configuration, which is "
               "why every configuration here is replicated"),
        Figure("barrier.shipped_wander_range_mnat", m["wander_range"], ".1f",
               "the same wander as a peak-to-peak range, averaged over the "
               "seeds — how far apart two honest single-arm readings of one "
               "configuration can land"),
    ]


def main():
    m = measure()
    lo, hi, n = m["tail_epochs"]
    seeds = m["seeds"]
    print(f"barrier panel  ({len(CONFIGS)} configurations x {len(seeds)} seeds "
          f"{seeds}, full budget, one machine: {m['gpu']})")
    print("  a contrast is taken WITHIN a seed (same held-out rows, paired) "
          "and the seeds are averaged after;")
    print(f"  'tail' is the mean over the {n} checkpoints from epoch {lo} to "
          f"{hi}, 'final' is epoch {EPOCHS}.")

    print("\nper-fit properties (each arm on its own, no contrast)")
    print(f"{'arm':>14}{'seed':>6}{'rating p0.1..p99.9':>27}{'span':>9}"
          f"{'slopes on clamp':>18}")
    for cfg in CONFIGS:
        d = m["dist"][cfg]
        for i, seed in enumerate(seeds):
            print(f"{cfg:>14}{seed:>6}"
                  f"{d['low'][i]:>18.1f}..{d['high'][i]:<7.1f}"
                  f"{d['span'][i]:>9.1f}{d['floor_share'][i]:>18.1%}")

    print("\nheld-out cost against the shipped arm OF THE SAME SEED, millinats "
          "(positive = worse)")
    head = "".join(f"{f's{s} final':>12}{f's{s} tail':>12}" for s in seeds)
    print(f"{'arm':>14}{head}{'mean final':>13}{'mean tail':>12}{'signs':>9}")
    for cfg, c in m["contrasts"].items():
        cells = "".join(f"{f:>12.2f}{t:>12.2f}"
                        for f, t in zip(c["final"], c["tail"]))
        print(f"{cfg:>14}{cells}{c['final_mean']:>13.2f}{c['tail_mean']:>12.2f}"
              f"{('agree' if c['sign_consistent'] else 'DIFFER'):>9}")
    print("  'signs' agree only where every seed lands on the same side of zero "
          "on both readings.")
    print("  Both rows are figures: the 10x arm's fit is part of the "
          "sweep the weight was chosen by.")

    print(f"\nthe panel's resolution floor: the shipped arm's own held-out "
          f"spread over epochs {lo}..{hi}")
    for w in m["wander"]:
        print(f"{'seed ' + str(w['seed']):>14}{w['sd']:>10.2f} mnat sd"
              f"{w['range']:>11.2f} mnat peak-to-peak")
    print(f"{'averaged':>14}{m['wander_sd']:>10.2f} mnat sd"
          f"{m['wander_range']:>11.2f} mnat peak-to-peak")
    print("  No contrast smaller than this is resolved by one arm per "
          "configuration.")


if __name__ == "__main__":
    main()
