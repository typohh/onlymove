#!/usr/bin/env python3
"""The learning-rate split's figures (methodology §1.8, repeated in §3.5's
constants table), read from the `lr` refit panel.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` (`ablations.run_panel`, locally or via
`vast_orchestration.py ablate`) and hard-fails when the panel is missing, short
of budget, internally inconsistent, or STALE against the current attempt log —
the pins rule `docs/mirt_benchmark.md` states, mechanized. Guards and spans keep
evaluating against the last-measured ledger values, so the panel's cost is paid
only when it is deliberately re-run.

WHAT THE PANEL VARIES. Production trains the shared network at LR_NET and the
free per-user fields (the abilities, the axis positions, the lapse) at LR_FIELD,
a ratio of about thirty-fold. The arms move one rate at a time: `ratio1x` pulls
the field rate down ONTO the network's, which is the single-shared-rate
configuration §1.8 says the split exists to avoid; `ratio10x` sits at the bottom
of the saturating region; `net_slow` keeps the shipped field rate and thirds the
network's, which is the "two values a factor of three apart" comparison.

EVERY CONFIGURATION RUNS AT TWO SEEDS. The shipped configuration's own held-out
trace wanders across the last third of its budget by more than any of the
contrasts here (`barrier.shipped_wander_sd_mnat` measures that floor on the
sibling panel), so one arm per configuration resolves none of them. The seed
drives the initialization AND the held-out draw, so WITHIN one seed every arm
holds aside the same stop rows and a cross-arm delta is paired, while the two
seeds are two independent paired estimates of the same contrast. Nothing below
subtracts across seeds: a contrast is taken inside a seed, and the seeds are
averaged afterwards with their SIGN agreement reported separately.

EVERY CONTRAST IS READ TWICE. The FINAL epoch is the fit that would actually
ship; the mean over the last third is the level the arm settled at, which the
final checkpoint only samples at one point of the wander. Neither reading is a
summary of the other, so both carry a key and the tail one is the headline.

DIVERGENCE IS A MEASUREMENT HERE, NOT AN INPUT ERROR. The shared-rate arm is
expected to blow up — that IS the claim — so a non-finite or grossly risen
held-out trace is read out as `lr.shared_rate_diverged` rather than raised. Only
the SHIPPED arm is required to be finite throughout: it is the baseline every
paired delta is taken against, so a diverging baseline voids its seed.

  lr.shared_rate_diverged      1 when the single-shared-rate arm diverged AT ANY
                               seed, 0 when it did so at none — the claim is
                               that one shared rate CAN blow up, so a single
                               seed that does it is the finding. A 1/0
                               indicator, meant for a
                               `<!-- fig-guard: lr.shared_rate_diverged == 1 -->`
                               beside §1.8's "diverges late and in evaluation
                               mode" rather than for a span.
  lr.shared_rate_diverge_epoch the EARLIEST epoch any seed first did so (0 =
                               none did) — §1.8's "late" as a number.
  lr.shared_rate_tail_cost_mnat  its held-out (stop-split) cost against the
                               shipped arm of its own seed over the last third
                               of the budget, meaned over the seeds — the
                               headline cost.
  lr.shared_rate_cost_mnat     the same cost at the last checkpoint both traces
                               are finite, which for an arm that stayed finite
                               is the final epoch.
  lr.ratio10x_tail_cost_mnat   the same two readings at a ten-fold ratio —
  lr.ratio10x_cost_mnat        §1.8's "below ten-fold the fit loses several
                               millinats".
  lr.net_slow_tail_cost_mnat   the same two for the three-times-slower network
  lr.net_slow_cost_mnat        at the shipped field rate; the claim is a TIE, so
                               these are the figures meant to be small, and they
                               are signed (positive = the slower net is worse).
  lr.shared_rate_cost_sign_consistent  1 only when every seed agrees in sign on
  lr.ratio10x_cost_sign_consistent     BOTH readings of that contrast, 0 when
  lr.net_slow_cost_sign_consistent     they disagree. Indicators meant for a
                               `<!-- fig-guard: KEY == 1 -->` beside a sentence
                               that names the direction of a cost, never for a
                               span.
  lr.shipped_converge_epoch    where each of those two arms reaches its own
  lr.net_slow_converge_epoch   plateau, defined below, averaged over the seeds.
  lr.net_slow_converge_ratio   the second over the first — §1.8's "reaches it in
                               half the epochs", as the factor it actually is.

Two definitions, both stated here because both are choices:

CONVERGENCE. An arm's plateau is the band it settles into, so the bar is the
arm's OWN spread across the tail window rather than a fixed width:
`converge_epoch` is the FIRST checkpoint from which the score never again rises
above the arm's final value by more than that spread. Sustained rather than
first-touch, so a transient dip on the way down cannot claim the plateau, and
per-arm rather than shared, because these configurations differ by an order of
magnitude in how widely they wander — under a fixed bar the arm that wanders
more would read as the slower to converge, which is not what the epochs are
asked. It is a per-fit property, read on each seed's arm and averaged.

DIVERGENCE. Non-finite at any checkpoint, or risen more than
DIVERGENCE_RISE_MNAT millinats above the arm's OWN RUNNING BEST — running rather
than final, so the test is causal and the epoch it reports is the epoch a watcher
would have seen it at. The threshold is twenty times the width of the settled
band, which is what separates a blow-up from the wandering §1.8 describes.

    ./.venv/bin/python3 -m ablations.refit_lr_figures
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
from pipeline.train_mirt import EPOCHS, LR_FIELD, LR_NET  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.model_constants import SEED  # noqa: E402

TIER = "refit"

PANEL = "lr"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")
# One record per CONFIGURATION per seed. The manifest names a replicate by
# suffixing the configuration it repeats, which is what lets every delta be
# matched to the baseline of its own seed.
CONFIGS = ("shipped", "ratio1x", "ratio10x", "net_slow")
BASELINE = CONFIGS[0]
REPLICATE_SUFFIXES = ("", "_s1")
ARMS = tuple(c + s for s in REPLICATE_SUFFIXES for c in CONFIGS)
# The ledger key each variant's costs are published under. `ratio1x` is named
# for what it IS in §1.8 — the single shared rate — rather than for its ratio.
KEY_STEM = {"ratio1x": "shared_rate", "ratio10x": "ratio10x",
            "net_slow": "net_slow"}

# An arm has reached its plateau once it stays within its own closing spread of
# its own final value — a per-arm bar, since that spread differs by an order of
# magnitude across the configurations here (see converge_epoch).
#
# Divergence is the one absolute bar: a rise this far above an arm's own running
# best is past anything the widest settled band here reaches, so it separates a
# blow-up from wandering without being tuned to a particular arm.
DIVERGENCE_RISE_MNAT = 20.0
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
    require(set(manifest["arms"]) == set(ARMS),
            f"{MANIFEST} declares arms {sorted(manifest['arms'])}, this "
            f"producer reads {sorted(ARMS)}")

    arms = {}
    for arm in ARMS:
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the lr panel has not run at every seed. Run it "
                f"with `python -m ablations.run_panel lr` (or "
                f"`vast_orchestration.py ablate lr`) first; the resume unit is "
                f"the arm, so the arms already recorded are skipped.")
        arms[arm] = json.load(open(path))

    for arm, r in arms.items():
        require(r["metrics"]["epochs"] == EPOCHS,
                f"{arm} ran {r['metrics']['epochs']} epochs, not the production "
                f"budget of {EPOCHS} — a smoke arm is not quotable")
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

    groups = seed_groups(arms)

    # What each arm's rates actually WERE, and the ratios the claims are about.
    # Stated here rather than only in the manifest, because a producer that
    # cannot say which ratio it measured cannot be checked. Every replicate is
    # checked too: a seed is the only thing a replicate may change.
    rates = {arm: (float(r["config"].get("lr_net", LR_NET)),
                   float(r["config"].get("lr_field", LR_FIELD)))
             for arm, r in arms.items()}
    for g in groups:
        for cfg, want in (("ratio1x", 1.0), ("ratio10x", 10.0)):
            arm = g["arms"][cfg]
            net, field = rates[arm]
            require(math.isclose(field / net, want, rel_tol=1e-9),
                    f"{arm} runs at a field/net ratio of {field / net:.4g}, not "
                    f"{want:g} — the arm does not measure the point it is named "
                    f"for")
        arm = g["arms"]["net_slow"]
        require(math.isclose(rates[arm][0], LR_NET / 3.0, rel_tol=1e-9)
                and rates[arm][1] == LR_FIELD,
                f"{arm} runs at {rates[arm]}, not the shipped field rate with a "
                f"third of the shipped net rate")
    return arms, rates, groups


def seed_groups(arms):
    """The panel regrouped into replicates: one entry per seed, holding that
    seed's arm name for every configuration.

    A group is ONE held-out draw. The seed selects the stop rows, so a delta
    between two arms of one group is paired; two groups are independent
    estimates of that delta under different rows and different inits, which is
    why they are only ever averaged and never subtracted."""
    groups = []
    for suffix in REPLICATE_SUFFIXES:
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


def diverge_epoch(epochs, bce):
    """The first checkpoint at which the held-out score is non-finite or stands
    DIVERGENCE_RISE_MNAT millinats above the arm's own RUNNING best; 0 = never.
    Running rather than final, so the test only ever looks backwards."""
    best = math.inf
    for e, v in zip(epochs, bce):
        if not math.isfinite(v):
            return e
        if v > best + DIVERGENCE_RISE_MNAT / 1e3:
            return e
        best = min(best, v)
    return 0


def converge_epoch(epochs, bce, tail_from):
    """The first checkpoint from which the score never again rises above the
    arm's own final value by more than the arm's OWN closing spread.

    The bar is per-arm on purpose. A fixed one measures nothing when an arm's
    settled band is several times wider than it: the epoch it returns is where
    that arm's last excursion ended, not where it stopped improving, and an arm
    that merely wanders more then looks slower to converge. What the epochs are
    asked to compare is how quickly each arm reaches ITS OWN plateau, so each
    arm's plateau is defined by the band it actually settles into."""
    require(all(math.isfinite(v) for v in bce),
            "converge_epoch on a trace that is not finite throughout — an arm "
            "that diverged has no plateau to reach")
    tail = [v for e, v in zip(epochs, bce) if e >= tail_from]
    require(len(tail) > 1,
            "an arm's closing spread needs more than one checkpoint in the tail "
            "window — the bar is that spread, so it cannot be taken from one")
    bar = bce[-1] + statistics.stdev(tail)
    i = len(epochs) - 1
    while i >= 0 and bce[i] <= bar:
        i -= 1
    return epochs[i + 1]


def paired_cost_mnat(arm_bce, ship_bce, epochs):
    """An arm's held-out cost against the shipped arm of its own seed, in
    millinats, at the LAST checkpoint both traces are finite. For an arm that
    stayed finite that is the final epoch; for one that blew up past
    representability it is the last epoch the comparison still exists at, which
    is reported alongside."""
    ok = [i for i in range(len(epochs))
          if math.isfinite(arm_bce[i]) and math.isfinite(ship_bce[i])]
    require(ok, "no checkpoint has both arms finite — nothing is comparable")
    i = ok[-1]
    return (arm_bce[i] - ship_bce[i]) * 1e3, epochs[i]


def paired_tail_cost_mnat(arm, arm_bce, ship_bce, tail):
    """The same cost on the MEAN over the last third of the budget — the level
    the two arms settled at rather than the point of the wander their final
    checkpoints happened to land on.

    Restricted to the tail checkpoints where BOTH traces are finite, so an arm
    that blew up past representability still yields whatever of its tail exists;
    how many checkpoints that left is reported alongside."""
    ok = [i for i in tail
          if math.isfinite(arm_bce[i]) and math.isfinite(ship_bce[i])]
    require(ok,
            f"{arm} has no checkpoint in the last third of the budget where "
            f"both it and its seed's shipped arm are finite — it left "
            f"representability before the settled window and never came back, "
            f"so there is no level for it to have settled at. Its divergence is "
            f"still the finding: read it off the final-checkpoint cost, which "
            f"reports the epoch it was taken at.")
    return (statistics.fmean(arm_bce[i] for i in ok)
            - statistics.fmean(ship_bce[i] for i in ok)) * 1e3, len(ok)


def same_sign(*series):
    """1 when every seed's reading lands on the same side of zero in each of the
    given series, 0 otherwise. A reading of exactly zero agrees with nothing, so
    it is never consistent."""
    return int(all(all(v > 0 for v in s) or all(v < 0 for v in s)
                   for s in series))


# --- the measurement -------------------------------------------------------

def measure():
    arms, rates, groups = load_panel()
    curves = {arm: stop_curve(r) for arm, r in arms.items()}
    grid = curves[BASELINE][0]
    for arm, (ep, _) in curves.items():
        require(ep == grid,
                f"{arm}'s trace is scored at different epochs than the shipped "
                f"arm's — the deltas would not be like for like")
    tail = [i for i, e in enumerate(grid) if e >= TAIL_FROM_EPOCH]
    require(len(tail) > 1,
            f"the trace carries {len(tail)} checkpoint(s) at or past epoch "
            f"{TAIL_FROM_EPOCH} — the settled level is read over the last third "
            f"of the budget, and it does not exist on a single point")

    for g in groups:
        require(all(math.isfinite(v) for v in curves[g["arms"][BASELINE]][1]),
                f"the shipped arm's held-out trace at seed {g['seed']} is not "
                f"finite throughout — the baseline every paired delta of that "
                f"seed is taken against diverged, so nothing in this panel is "
                f"quotable. Re-run it.")
        require(all(math.isfinite(v) for v in curves[g["arms"]["net_slow"]][1]),
                f"the net_slow arm diverged at seed {g['seed']}. §1.8's claim "
                f"about it is a TIE reached more slowly, so a diverged trace is "
                f"not a smaller version of that finding — it is a different "
                f"one, and this producer has no key for it.")

    # Divergence, per seed. One shared rate blowing up at EITHER seed is the
    # claim being reproduced, so the indicator is an OR over the seeds and the
    # epoch is the earliest a watcher would have seen it at.
    diverge = {g["seed"]: diverge_epoch(grid, curves[g["arms"]["ratio1x"]][1])
               for g in groups}
    hits = sorted(e for e in diverge.values() if e)

    # Contrasts: taken inside a seed, one reading per seed, in seed order.
    contrasts = {}
    for cfg in CONFIGS:
        if cfg == BASELINE:
            continue
        finals, tails, at, tail_n = [], [], [], []
        for g in groups:
            ship_bce = curves[g["arms"][BASELINE]][1]
            arm_bce = curves[g["arms"][cfg]][1]
            final, final_at = paired_cost_mnat(arm_bce, ship_bce, grid)
            tail_mean, n_used = paired_tail_cost_mnat(
                g["arms"][cfg], arm_bce, ship_bce, tail)
            finals.append(final)
            tails.append(tail_mean)
            at.append(final_at)
            tail_n.append(n_used)
        contrasts[cfg] = {
            "final": finals, "tail": tails, "at": at, "tail_n": tail_n,
            "final_mean": statistics.fmean(finals),
            "tail_mean": statistics.fmean(tails),
            "sign_consistent": same_sign(finals, tails),
        }

    # Convergence is a per-fit property, read on each seed's arm and averaged;
    # the ratio is taken between those averages rather than per seed, so one
    # arm's late plateau cannot dominate through a division.
    conv = {cfg: [converge_epoch(grid, curves[g["arms"][cfg]][1], TAIL_FROM_EPOCH)
                  for g in groups]
            for cfg in (BASELINE, "net_slow")}
    ship_conv = statistics.fmean(conv[BASELINE])
    slow_conv = statistics.fmean(conv["net_slow"])
    return {
        "rates": rates,
        "seeds": [g["seed"] for g in groups],
        "groups": groups,
        "tail_epochs": (grid[tail[0]], grid[tail[-1]], len(tail)),
        "diverged": 1 if hits else 0,
        "diverge_epoch": hits[0] if hits else 0,
        "diverge_per_seed": diverge,
        "contrasts": contrasts,
        "conv": conv,
        "ship_conv": ship_conv,
        "slow_conv": slow_conv,
        "slow_conv_ratio": slow_conv / ship_conv,
        "ship_final": [curves[g["arms"][BASELINE]][1][-1] for g in groups],
        "gpu": arms[BASELINE]["pins"]["gpu"],
    }


def figures():
    m = measure()
    seeds = len(m["seeds"])
    over = f"averaged over the panel's {seeds} seeds"
    paired = ("PAIRED against the shipped arm of its own seed — one seed is one "
              "held-out draw, so the arms inside it hold aside the same stop "
              f"rows — then meaned over the {seeds} seeds")
    out = [
        Figure("lr.shared_rate_diverged", m["diverged"], "d",
               "1 when the single-shared-rate arm (field rate pulled onto the "
               "network's) diverged at ANY seed, 0 when it did so at none — "
               f"non-finite, or {DIVERGENCE_RISE_MNAT:g} millinats above its "
               f"own running best held-out score. The claim is that one shared "
               f"rate CAN blow up, so one seed doing it is the finding. An "
               f"indicator for a fig-guard, not a span"),
        Figure("lr.shared_rate_diverge_epoch", m["diverge_epoch"], "d",
               "the earliest epoch at which any seed's single-shared-rate arm "
               "first met that test; 0 means no seed did"),
    ]
    for cfg in CONFIGS:
        if cfg == BASELINE:
            continue
        c, stem = m["contrasts"][cfg], KEY_STEM[cfg]
        what, fmt = {
            "ratio1x": ("of one shared rate against the shipped split", ".0f"),
            "ratio10x": ("at a ten-fold field/net ratio against the shipped "
                         "~33-fold — the bottom of the saturating region",
                         ".1f"),
            "net_slow": ("of a three-times-slower network at the shipped field "
                         "rate; signed, positive means the slower network is "
                         "the worse fit. The claim is a tie", ".2f"),
        }[cfg]
        out += [
            Figure(f"lr.{stem}_tail_cost_mnat", c["tail_mean"], fmt,
                   f"held-out (stop-split) cost {what}, millinats, over the "
                   f"last third of the budget — {paired}. The headline "
                   f"reading: it is the level the arm settled at, which a final "
                   f"checkpoint alone samples only once"),
            Figure(f"lr.{stem}_cost_mnat", c["final_mean"], fmt,
                   f"the same cost {what}, read at the last checkpoint both "
                   f"traces are finite — the final epoch for an arm that stayed "
                   f"finite, and the fit that would actually ship; {paired}"),
            Figure(f"lr.{stem}_cost_sign_consistent", c["sign_consistent"], "d",
                   f"1 when every seed puts this cost on the same side of zero "
                   f"on BOTH readings, 0 when the seeds disagree. An indicator "
                   f"for a `fig-guard: lr.{stem}_cost_sign_consistent == 1` "
                   f"beside any sentence naming the direction of that cost, "
                   f"never for a span"),
        ]
    out += [
        Figure("lr.shipped_converge_epoch", m["ship_conv"], ".0f",
               "first epoch from which the shipped arm's held-out score never "
               f"again rises above its own final value by more than its own "
               f"closing spread — the band it settles into, each arm measured "
               f"against the band it actually reaches. A per-fit property, "
               f"{over}"),
        Figure("lr.net_slow_converge_epoch", m["slow_conv"], ".0f",
               f"the same epoch for the three-times-slower network, {over}"),
        Figure("lr.net_slow_converge_ratio", m["slow_conv_ratio"], ".1f",
               "how many times longer the slower network takes to reach its own "
               "plateau, at the same held-out margin — the second averaged "
               "epoch over the first, so above 1 means the slower network "
               "plateaus later"),
    ]
    return out


def main():
    m = measure()
    lo, hi, n = m["tail_epochs"]
    seeds = m["seeds"]
    print(f"lr panel  ({len(CONFIGS)} configurations x {len(seeds)} seeds "
          f"{seeds}, full budget, one machine: {m['gpu']})")
    print("  a contrast is taken WITHIN a seed (same held-out rows, paired) "
          "and the seeds are averaged after;")
    print(f"  'tail' is the mean over the {n} checkpoints from epoch {lo} to "
          f"{hi}, 'final' the last epoch both traces are finite.")
    print(f"  convergence = first epoch from which the held-out score never "
          f"again rises above its own final value by more than that arm's own "
          f"closing spread")
    print(f"  divergence  = non-finite, or {DIVERGENCE_RISE_MNAT:g} mnat above "
          f"the arm's own running best")
    print()

    head = "".join(f"{f's{s} final':>12}{f's{s} tail':>12}" for s in seeds)
    print(f"{'arm':>10}{'lr_net':>10}{'lr_field':>10}{'ratio':>8}{head}"
          f"{'mean final':>13}{'mean tail':>12}{'signs':>9}{'converged':>11}")
    for cfg in CONFIGS:
        net, field = m["rates"][m["groups"][0]["arms"][cfg]]
        c = m["contrasts"].get(cfg)
        cells = ("".join(f"{f:>12.2f}{t:>12.2f}"
                         for f, t in zip(c["final"], c["tail"]))
                 if c else " " * (24 * len(seeds)))
        summary = (f"{c['final_mean']:>13.2f}{c['tail_mean']:>12.2f}"
                   f"{('agree' if c['sign_consistent'] else 'DIFFER'):>9}"
                   if c else f"{'baseline':>34}")
        # Convergence is scored only for the two arms §1.8's "half the epochs"
        # claim compares; the ratio arms are about the fit reached, not the
        # epoch it was reached at.
        conv = m["conv"].get(cfg)
        conv_cell = "/".join(str(e) for e in conv) if conv else "not scored"
        print(f"{cfg:>10}{net:>10.1e}{field:>10.1e}{field / net:>7.0f}x"
              f"{cells}{summary}{conv_cell:>11}")
    print()
    print("  cost is against the shipped arm of the SAME seed on the same "
          "held-out rows; positive = worse")
    print("  (shipped final "
          + ", ".join(f"s{s} {v:.4f}" for s, v in zip(seeds, m["ship_final"]))
          + "); 'converged' lists the per-seed epochs")
    print("  'signs' agree only where every seed lands on the same side of zero "
          "on both readings.")
    if m["diverged"]:
        per_seed = ", ".join(f"s{s} {'epoch ' + str(e) if e else 'no'}"
                             for s, e in m["diverge_per_seed"].items())
        print(f"  the shared-rate arm DIVERGED ({per_seed}); the earliest is "
              f"epoch {m['diverge_epoch']}, and each seed's 'final' cost above "
              f"is read at "
              + ", ".join(f"s{s} epoch {e}" for s, e in
                          zip(seeds, m["contrasts"]["ratio1x"]["at"])))
        print(f"  its 'tail' cost is read over "
              + ", ".join(f"s{s} {k}" for s, k in
                          zip(seeds, m["contrasts"]["ratio1x"]["tail_n"]))
              + f" of the {n} tail checkpoints — the ones where both traces are "
                f"still finite")
    else:
        print("  the shared-rate arm did NOT diverge at any seed — §1.8's "
              "stated reason for the split is not reproduced on this panel")
    print(f"  the slower network reaches its plateau at "
          f"{m['slow_conv_ratio']:.1f}x the shipped arm's epoch (above 1 = "
          f"later), on the seed-averaged epochs")


if __name__ == "__main__":
    main()
