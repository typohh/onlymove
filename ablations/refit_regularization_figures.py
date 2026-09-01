#!/usr/bin/env python3
"""Regularization's figures (methodology §1.5), read from the `regularization`
refit panel: what each of the two dropout sites is worth held out, how far their
removals compound, and whether weight decay stands in for either.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` (`ablations.run_panel`, locally or via
`vast_orchestration.py ablate`) and hard-fails when the panel is missing, short
of budget, running a config other than the manifest's, internally inconsistent,
or STALE against the current attempt log — the pins rule `docs/mirt_benchmark.md`
states, mechanized. Guards and spans keep evaluating against the last-measured
ledger values, so the panel's cost is paid only when it is deliberately re-run.

THE STOP UNIT IS THE CRUX, NOT A DETAIL. Every arm holds out whole PROBLEMS.
Memorization means fitting problems the model has already seen at the expense of
ones it has not, so the shipped row split — which leaves every problem partly
trained — is the one split that cannot see it: an arm that fingerprints its own
training problems is rewarded by that split rather than caught by it. A held-out
problem contributes no row to training at all and its parameters come from the
feature trunk alone, which is the only reading on which a claim about
regularization means anything. Nothing here is scored on the row split, and a
record fitted on one is refused rather than reported.

THE HEADLINE READING IS THE MEAN OVER THE LAST THIRD OF THE BUDGET. The shipped
configuration's own held-out score wanders across that last third by more than
some of the contrasts this panel is asked about, so a single checkpoint is not a
stable estimate of where an arm sits; `regularization.shipped_wander_range_mnat`
is that resolution floor, measured on this panel's own replicates and printed
beside every contrast. The final-epoch reading carries a key of its own beside
the headline, because it is the number a reader of the trace computes first and
the two disagreeing by more than the wander would itself be a finding — but the
indicators are defined on the headline reading alone.

PAIRING AND SEEDS. The seed drives the held-out draw as well as the
initialization, so within one seed every arm holds aside the same problems and a
cross-arm delta is paired; the record's row, user, stop-row and base-rate counts
are checked for that identity rather than assumed. Across seeds nothing is
paired — two seeds are two holdout draws — so no cross-seed delta is taken
anywhere: a contrast is formed inside a seed and the seeds are averaged
afterwards. What the second seed buys is a second independent estimate of the
same paired delta, and each contrast carries a 0/1 indicator saying whether the
two agreed in SIGN.

A REFUTED DIRECTION IS REPORTED, NOT RAISED. Should the excess over additivity
come back negative, or should a decay arm beat the both-off arm, the indicators
come back 0 and the magnitudes come back signed. That is a measurement, and what
would then need rewriting is §1.5's sentence rather than this producer — which is
what the fig-guards beside those sentences exist to force.

  regularization.<config>_cost_mnat        each configuration's held-out-PROBLEM
                                           cost against the shipped fit,
                                           millinats, meaned over the last third
                                           of the budget. PAIRED inside a seed,
                                           then averaged over the two seeds.
                                           Signed: positive = worse than the
                                           shipped fit. The configs are
                                           input_off (feature-side column dropout
                                           removed), trunk_off (the trunk's
                                           dropout removed), both_off (both),
                                           decay (weight decay added, dropout
                                           intact) and decay_both_off (weight
                                           decay standing in for both dropouts).
  regularization.<config>_cost_final_mnat  the same average read at the final
                                           epoch alone, where one checkpoint is
                                           one draw from the settled trace's own
                                           wander.
  regularization.<config>_cost_sign_consistent
                                           1 only when both seeds put that
                                           contrast on the same side of zero on
                                           the headline reading. For a
                                           `<!-- fig-guard: KEY == 1 -->`, not a
                                           span.
  regularization.superadditive_excess_mnat THE POINT OF THE PANEL. The EXCESS
                                           OVER ADDITIVITY, not a total: the cost
                                           of removing both dropout sites MINUS
                                           the sum of the two single-site costs,
                                           every term a within-seed paired cost
                                           against the shipped fit on the
                                           headline reading, formed inside a seed
                                           and then averaged. Positive means the
                                           two removals compound — together they
                                           cost more than separately — which is
                                           §1.5's super-additivity; zero means
                                           they merely add; negative means they
                                           overlap.
  regularization.superadditive_excess_final_mnat
                                           the same excess at the final epoch.
  regularization.superadditive_every_seed  1 only when that excess is positive at
                                           EVERY seed, not merely on the average
                                           — an average could be carried by one
                                           seed alone. The indicator §1.5's
                                           "strongly super-additive" rests on.
  regularization.decay_substitution_gap_mnat
                                           the within-seed paired difference
                                           between the decay-with-dropout-off arm
                                           and the both-off arm, millinats,
                                           averaged over the seeds. Positive means
                                           adding weight decay to a fit with no
                                           dropout left it WORSE than no
                                           regularization at all; around zero
                                           means it changed nothing; negative
                                           means it recovered some of what
                                           dropout was doing.
  regularization.decay_no_substitute       §1.5'S ACTUAL CLAIM as an indicator: 1
                                           when the decay-with-both-dropouts-off
                                           arm is NO BETTER than the both-off arm
                                           — its held-out cost against the shipped
                                           fit is at least the both-off arm's — at
                                           every seed. That is what "no single
                                           norm penalty stands in for either"
                                           asserts: not that decay is harmful,
                                           only that it buys back none of what
                                           removing dropout cost.
  regularization.shipped_wander_range_mnat the resolution floor every contrast
                                           above is read against: the shipped
                                           arm's own peak-to-peak held-out spread
                                           across the last third of ITS OWN run,
                                           millinats, averaged over the seeds. One
                                           configuration, one seed, so it is
                                           wander and nothing else.

WHAT THIS PANEL CANNOT REACH. §1.5 gives a MECHANISM — "a low-norm readout of a
near-unique high-dimensional code still fingerprints" — and this panel measures
only its consequence on one number, the held-out-problem likelihood. Nothing here
reads a weight norm, counts a fingerprint, or separates general degradation from
a selective failure to suppress memorization; what is shown is that the decay
arms do not recover what removing dropout costs. The decay is ONE VALUE rather
than a sweep (the manifest says which and why), so no statement of the form "no
decay value substitutes" is available from it, and the knob is
`torch.optim.Adam`'s own `weight_decay`, which reaches the free per-user fields
as well as the network — the decay arms therefore also carry an L2 on ability,
which the shipped design leaves unpenalized. The claim that the two sites
suppress DIFFERENT failure modes is supported here only through the excess: a
positive excess is consistent with two distinct modes and does not identify
either. And §1.5's other regularization claims — column granularity beating block
dropout, and the dose-response plateau — have no arm in this panel and must not
be read off one.

    ./.venv/bin/python3 -m ablations.refit_regularization_figures
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
from pipeline.train_mirt import DROPOUT, INPUT_DROPOUT  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.model_constants import (  # noqa: E402
    K_AXES, PCA_DIM, PRODUCTION_BLOCKS)

TIER = "refit"

PANEL = "regularization"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")

# The single decay value both decay arms run, stated here as well as in the
# manifest: the producer attributes figures to a named configuration, so it
# declares what that configuration IS rather than inheriting it from a file that
# can be edited after the run. One point, not a sweep — the manifest carries the
# reasoning.
WEIGHT_DECAY = 1e-4

# WHAT EACH CONFIGURATION IS, as the regularization knobs it passes. A missing
# knob means the production constant, so the baseline is defined by ABSENCE:
# `shipped` passes none and therefore runs at the shipped dropout rates with no
# decay. Checked against the manifest AND against each record, because a manifest
# entry that grew a knob would otherwise rename some other configuration
# `shipped` and every contrast below would silently be taken against it.
REG_KNOBS = {
    "shipped": {},
    "input_off": {"input_dropout": 0.0},
    "trunk_off": {"dropout": 0.0},
    "both_off": {"dropout": 0.0, "input_dropout": 0.0},
    "decay": {"weight_decay": WEIGHT_DECAY},
    # The decay arm's gain needs ATTRIBUTING before it can be a candidate:
    # Adam's weight_decay reaches the free per-user fields too — an L2 on
    # ability the shipped design leaves unpenalized, which re-gauges the fit
    # (abilities shrink, slopes inflate to compensate). This arm decays the
    # NETWORK alone, fields exempt, so its contrast against `decay` says
    # whether the gain is regularization or the ridge's re-gauging.
    "decay_net": {"weight_decay": WEIGHT_DECAY, "weight_decay_fields": 0.0},
    "decay_both_off": {"dropout": 0.0, "input_dropout": 0.0,
                       "weight_decay": WEIGHT_DECAY},
}
CONFIGS = tuple(REG_KNOBS)
BASELINE = "shipped"
CONTRASTS = tuple(c for c in CONFIGS if c != BASELINE)
# The two dropout SITES, each removed alone. Their costs are what the both-off
# cost is compared against, so the pair is named once and read in this order
# everywhere.
SINGLE_SITES = ("input_off", "trunk_off")
BOTH_SITES = "both_off"
# The substitution §1.5 rejects: weight decay carrying the load with no dropout
# anywhere, against the same fit with nothing at all.
DECAY_ONLY = "decay_both_off"
SEEDS = (0, 1)
# Every arm scores on held-out PROBLEMS. See the module docstring: the row split
# cannot see the failure mode this whole panel is about.
STOP_UNIT = "problem"

# One clause per configuration, reused by every figure and table row that carries
# it so two configurations can never be described as the same thing.
CONFIG_NOTE = {
    "shipped": "the shipped fit — dropout at both sites, no weight decay",
    "input_off": "the feature region's COLUMN dropout removed, the trunk's kept",
    "trunk_off": "the trunk's dropout removed, the feature region's kept",
    "both_off": "dropout removed at BOTH sites",
    "decay": "weight decay added with both dropouts intact",
    "decay_net": "the same weight decay on the NETWORK alone, the per-user "
                 "fields exempt",
    "decay_both_off": "weight decay standing in for both dropouts, which is the "
                      "substitution §1.5 rejects",
}
# The same configurations as a verb phrase, for the figure notes, where the
# clause above has to sit inside a sentence. The baseline needs no phrase: every
# contrast is taken against it, so nothing is ever described as its cost.
CONFIG_ACTION = {
    "input_off": "removing the feature region's column dropout",
    "trunk_off": "removing the trunk's dropout",
    "both_off": "removing dropout at both sites",
    "decay": "adding weight decay with both dropouts intact",
    "decay_net": "adding the same weight decay to the network alone, the "
                 "per-user fields exempt",
    "decay_both_off": "weight decay in place of both dropouts",
}


def arm_name(config, seed):
    """The manifest's name for one (configuration, seed) arm. The baseline is
    called `shipped`, so the fit every contrast is taken against is obvious in a
    directory listing and in a failure message."""
    return f"{config}_s{seed}"


ARMS = {(c, s): arm_name(c, s) for c in CONFIGS for s in SEEDS}


# --- the panel record ---------------------------------------------------------

def load_panel():
    """Every arm's record, checked for what makes this panel quotable: the
    production budget, the manifest's own config, the regularization each arm is
    named for, the held-out-problem split, pins agreeing across arms, the
    within-seed pairing, and a snapshot still matching the current attempt log."""
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
                f"no {path} — the regularization panel has not run at every "
                f"configuration and seed (missing {arm}: {CONFIG_NOTE[key[0]]}, "
                f"seed {key[1]}). Run it with `python -m ablations.run_panel "
                f"regularization` (or `vast_orchestration.py ablate "
                f"regularization`) first; the resume unit is the arm, so arms "
                f"already recorded are skipped.")
        with open(path) as f:
            arms[key] = json.load(f)

    # ONE PANEL IS ONE REGIME, read off the arms rather than off the trainer's
    # current constants: the arms are only ever compared to each other, so a
    # later change to the shipped budget or schedule must not make a finished
    # panel unquotable.
    budget, schedule = panel_regime(arms)
    for key, r in arms.items():
        arm, (config, seed) = ARMS[key], key
        require(r["metrics"]["trace"]
                and r["metrics"]["trace"][-1]["epoch"] == budget,
                f"{arm}'s convergence trace does not reach the panel's "
                f"{budget}-epoch budget")
        require(all(math.isfinite(float(t["stop_bce"]))
                    for t in r["metrics"]["trace"]),
                f"{arm}'s held-out trace is not finite throughout. A fit that "
                f"diverges without dropout is a different finding from one that "
                f"costs likelihood, not a larger version of it, and this "
                f"producer has no key for it")

        # THE ARM RAN THE MANIFEST'S CONFIG. Every figure below is attributed to
        # a named configuration at a named seed, so a record disagreeing with the
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

        # THE REGULARIZATION EACH ARM IS NAMED FOR, stated by this producer and
        # checked against the manifest entry and the record together. Dropout and
        # decay are CONFIG rather than pins — nothing in `snapshot_pins` records
        # them — so the config block is the only witness that an arm ran the
        # configuration its name claims, and the check is exact in both
        # directions: a knob present that should be absent redefines the arm just
        # as surely as one absent that should be present, and the BASELINE is
        # defined by absence.
        for source, cfg in (("manifest entry", want), ("record", got)):
            passed = {k: cfg[k] for k in ("dropout", "input_dropout",
                                          "weight_decay",
                                          "weight_decay_fields") if k in cfg}
            declared = REG_KNOBS[config]
            require(set(passed) == set(declared)
                    and all(float(passed[k]) == float(v)
                            for k, v in declared.items()),
                    f"{arm}'s {source} sets the regularization knobs "
                    f"{passed!r}, but the {config!r} configuration is "
                    f"{REG_KNOBS[config]!r} ({CONFIG_NOTE[config]}) — the arm "
                    f"does not carry the regularization its name declares")

        # Read off the record rather than left to the manifest comparison above,
        # because the whole panel rests on these two: a row-split arm would
        # produce every number below in the same shape from the one split that
        # cannot see memorization at all, and an arm at the wrong seed would be
        # differenced against a baseline holding out other problems.
        require(got.get("stop_unit") == STOP_UNIT,
                f"{arm} was fitted with stop_unit={got.get('stop_unit')!r}, not "
                f"{STOP_UNIT!r} — its held-out score is not over unseen "
                f"problems, and a row split is the one split that rewards "
                f"memorizing the training problems instead of exposing it")
        require(int(got["seed"]) == seed,
                f"{arm} ran at seed {got['seed']}, not the {seed} its name "
                f"declares")

    # NO PIN VARIES ACROSS THESE ARMS. The panel sweeps dropout and decay, which
    # are config and not pins, so the cross-arm pin equality every panel applies
    # holds here unchanged and with no exemption — these are the arms of one
    # panel or they are not comparable at all.
    base = arms[(BASELINE, SEEDS[0])]["pins"]
    for key, r in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha", "blocks",
                    "pca_dim", "k_axes"):
            require(r["pins"][pin] == base[pin],
                    f"{ARMS[key]}'s pin {pin} ({r['pins'][pin]!r}) differs from "
                    f"the shipped arm's ({base[pin]!r}) — the arms are not one "
                    f"panel and their comparison is void")
    require(base["blocks"] == str(PRODUCTION_BLOCKS),
            f"the panel was fitted on blocks {base['blocks']!r}, this tree ships "
            f"{PRODUCTION_BLOCKS!r} — regularization measured on some other "
            f"feature side is not evidence about the shipped one")
    require(int(base["pca_dim"]) == int(PCA_DIM),
            f"the panel was fitted at pca_dim={base['pca_dim']}, this tree ships "
            f"PCA_DIM={PCA_DIM} — the width is what the feature-side dropout is "
            f"applied over, so this is a different question at a different width")
    require(int(base["k_axes"]) == int(K_AXES),
            f"the panel was fitted at K={base['k_axes']}, this tree ships "
            f"K_AXES={K_AXES} — the model these arms regularize has changed")

    # THE PAIRING, CHECKED RATHER THAN ASSUMED. The held-out draw comes from the
    # seed over the problem list, so arms at one seed on one attempt log hold
    # aside the same problems — which is what makes a millinat delta paired
    # instead of two independent estimates. Row, user and base-rate counts are
    # seed-independent and must match everywhere; the held-aside count is the
    # seed's own and must match within a seed.
    ship = arms[(BASELINE, SEEDS[0])]["metrics"]
    for key, r in arms.items():
        for field in ("rows", "users", "base_rate", "bank_spec"):
            require(r["metrics"][field] == ship[field],
                    f"{ARMS[key]} fitted {field}={r['metrics'][field]!r} against "
                    f"the baseline's {ship[field]!r} — the arms are not fitted "
                    f"over the same observations")
    for s in SEEDS:
        want_rows = arms[(BASELINE, s)]["metrics"]["stop_rows"]
        for c in CONFIGS:
            got_rows = arms[(c, s)]["metrics"]["stop_rows"]
            require(got_rows == want_rows,
                    f"{ARMS[(c, s)]} held aside {got_rows} rows against "
                    f"{ARMS[(BASELINE, s)]}'s {want_rows} — the two do not hold "
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
    """One arm's held-out cost against the SHIPPED arm of its own seed, in
    millinats, as a mean over the tail checkpoints and at the final checkpoint.
    Positive = worse than the shipped fit. Both are differences taken inside one
    seed, where the two arms hold aside the same problems, so both are paired."""
    tail_mean = (statistics.fmean(arm_bce[i] for i in tail)
                 - statistics.fmean(ship_bce[i] for i in tail)) * 1e3
    return tail_mean, (arm_bce[-1] - ship_bce[-1]) * 1e3


def same_sign(series):
    """1 when every seed's reading lands on the same side of zero, 0 otherwise. A
    reading of exactly zero agrees with nothing, so it is never consistent."""
    return int(all(v > 0 for v in series) or all(v < 0 for v in series))


# --- the measurement ----------------------------------------------------------

def measure():
    arms, budget, schedule = load_panel()
    curves = {key: stop_curve(r) for key, r in arms.items()}
    grid = curves[(BASELINE, SEEDS[0])][0]
    for key, (ep, _) in curves.items():
        require(ep == grid,
                f"{ARMS[key]}'s trace is scored at different epochs than the "
                f"baseline's — the deltas would not be like for like")
    # The last third of the budget, the same window the other refit producers
    # read their settled level over: past the stretch where these fits still
    # improve, so a mean over it reads the level an arm settled at rather than
    # whichever point of its wander the final checkpoint happened to land on.
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
    # ever subtracts across seeds, which hold aside different problems. The
    # per-seed lists are kept because every indicator below is a statement about
    # each seed rather than about the average.
    cost = {}
    for c in CONTRASTS:
        tails, finals = [], []
        for s in SEEDS:
            _, ship_bce = curves[(BASELINE, s)]
            _, arm_bce = curves[(c, s)]
            t, f = deltas_mnat(arm_bce, ship_bce, tail)
            tails.append(t)
            finals.append(f)
        cost[c] = {
            "tail": tails, "final": finals,
            "tail_mean": statistics.fmean(tails),
            "final_mean": statistics.fmean(finals),
            # On the headline reading, which is what the indicator is defined on.
            # The final-epoch series gets the same test for the table, where it
            # says whether a one-checkpoint reading would have agreed — context,
            # not a criterion.
            "sign_consistent": same_sign(tails),
            "final_sign_consistent": same_sign(finals),
        }

    # THE EXCESS OVER ADDITIVITY. Removing both sites, minus removing each alone
    # — every term already a within-seed paired cost against the shipped fit, so
    # the excess is itself formed inside one seed and the seeds are averaged
    # afterwards. Positive is §1.5's super-additivity: the two removals compound.
    # Zero is plain additivity, negative is overlap; both are reported.
    excess_tail = [cost[BOTH_SITES]["tail"][i]
                   - sum(cost[c]["tail"][i] for c in SINGLE_SITES)
                   for i in range(len(SEEDS))]
    excess_final = [cost[BOTH_SITES]["final"][i]
                    - sum(cost[c]["final"][i] for c in SINGLE_SITES)
                    for i in range(len(SEEDS))]

    # DOES DECAY STAND IN FOR DROPOUT? The comparison is the decay-with-no-dropout
    # arm against the no-dropout arm, both measured against the same shipped
    # baseline at the same seed, so the gap between them is a paired difference
    # too. "No better" is the claim, so the test is one-sided: decay substitutes
    # for nothing unless it recovers some of what removing dropout cost.
    subst_gap = [cost[DECAY_ONLY]["tail"][i] - cost[BOTH_SITES]["tail"][i]
                 for i in range(len(SEEDS))]

    # The shipped arm's own spread over the window every contrast is read on —
    # the resolution floor, and the reason the tail reading is the headline.
    wander = {s: {"sd": statistics.stdev([curves[(BASELINE, s)][1][i] * 1e3
                                          for i in tail]),
                  "range": (max(curves[(BASELINE, s)][1][i] for i in tail)
                            - min(curves[(BASELINE, s)][1][i]
                                  for i in tail)) * 1e3}
              for s in SEEDS}

    return {
        "tail_epochs": (grid[tail[0]], grid[tail[-1]], len(tail)),
        "budget": budget,
        "schedule": schedule,
        "level": level, "final": final, "cost": cost,
        "holdout_bce": {c: statistics.fmean(level[(c, s)] for s in SEEDS)
                        for c in CONFIGS},
        "excess_tail": excess_tail,
        "excess_final": excess_final,
        "excess_tail_mean": statistics.fmean(excess_tail),
        "excess_final_mean": statistics.fmean(excess_final),
        "excess_every_seed": int(all(v > 0 for v in excess_tail)),
        "subst_gap": subst_gap,
        "subst_gap_mean": statistics.fmean(subst_gap),
        "no_substitute": int(all(v >= 0 for v in subst_gap)),
        "wander": wander,
        "wander_range": statistics.fmean(w["range"] for w in wander.values()),
        "stop_rows": {s: arms[(BASELINE, s)]["metrics"]["stop_rows"]
                      for s in SEEDS},
        "rows": int(arms[(BASELINE, SEEDS[0])]["metrics"]["rows"]),
        "gpu": arms[(BASELINE, SEEDS[0])]["pins"]["gpu"],
    }


# --- the figures --------------------------------------------------------------

def figures():
    m = measure()
    over = f"averaged over the panel's {len(SEEDS)} seeds"
    out = []
    for c in CONTRASTS:
        k, act = m["cost"][c], CONFIG_ACTION[c]
        out.append(Figure(
            f"{PANEL}.{c}_cost_mnat", k["tail_mean"], ".2f",
            f"the held-out-PROBLEM cost of {act}, against the shipped fit, in "
            f"millinats over the last third of the budget and signed (positive "
            f"= worse than shipped). PAIRED inside a seed — both arms hold "
            f"aside the same problems — and then {over}"))
        out.append(Figure(
            f"{PANEL}.{c}_cost_final_mnat", k["final_mean"], ".2f",
            f"the same paired cost of {act}, read at the final epoch alone, "
            f"where one checkpoint is one draw from the settled trace's own "
            f"wander, {over}"))
        out.append(Figure(
            f"{PANEL}.{c}_cost_sign_consistent", k["sign_consistent"], "d",
            f"1 when both seeds put the cost of {act} on the same side of zero "
            f"on the last-third reading, 0 when they disagree — the two seeds "
            f"being independent estimates of one paired delta. An indicator for "
            f"a fig-guard beside a sentence naming that contrast's direction, "
            f"not for a span"))
    out.append(Figure(
        f"{PANEL}.superadditive_excess_mnat", m["excess_tail_mean"], ".2f",
        f"THE EXCESS OVER ADDITIVITY, in millinats, and not a total: the "
        f"held-out cost of removing dropout at BOTH sites MINUS the sum of the "
        f"costs of removing it at each site alone, every term a within-seed "
        f"paired cost against the shipped fit on the last-third reading, the "
        f"excess formed inside a seed and then {over}. Positive means the two "
        f"removals compound (§1.5's super-additivity), zero that they merely "
        f"add, negative that they overlap"))
    out.append(Figure(
        f"{PANEL}.superadditive_excess_final_mnat", m["excess_final_mean"],
        ".2f",
        f"the same excess over additivity read at the final epoch alone — "
        f"`{PANEL}.superadditive_excess_mnat` on one checkpoint instead of the "
        f"last third, and therefore carrying the settled trace's full wander"))
    out.append(Figure(
        f"{PANEL}.superadditive_every_seed", m["excess_every_seed"], "d",
        f"1 when that excess over additivity is positive at EVERY seed, 0 "
        f"otherwise — never an average, which one seed could carry alone. The "
        f"indicator §1.5's 'strongly super-additive' rests on; a "
        f"fig-guard's figure, not a span's"))
    out.append(Figure(
        f"{PANEL}.decay_substitution_gap_mnat", m["subst_gap_mean"], ".2f",
        f"the within-seed paired difference between the {DECAY_ONLY!r} arm and "
        f"the {BOTH_SITES!r} arm, millinats on the last-third reading, {over}. "
        f"Positive means weight decay left a dropout-free fit WORSE than no "
        f"regularization at all; around zero that it changed nothing; negative "
        f"that it recovered some of what dropout was doing"))
    out.append(Figure(
        f"{PANEL}.decay_no_substitute", m["no_substitute"], "d",
        f"§1.5'S CLAIM as an indicator: 1 when weight decay does not substitute "
        f"for dropout — the arm carrying decay with both dropouts off is NO "
        f"BETTER held out than the arm carrying nothing at all, its cost against "
        f"the shipped fit being at least the {BOTH_SITES!r} arm's, at every "
        f"seed. 0 means decay recovered part of what dropout was doing at some "
        f"seed. The claim is one-sided: it says decay buys back none of that "
        f"cost, not that decay is itself harmful"))
    out.append(Figure(
        f"{PANEL}.decay_gain_is_field_ridge",
        int(all(v < 0 for v in m["cost"]["decay"]["tail"])
            and all(v > 0 for v in m["cost"]["decay_net"]["tail"])), "d",
        f"THE ATTRIBUTION of the decay arm's held-out gain: 1 when, at every "
        f"seed, decaying network AND fields beats the shipped fit while "
        f"decaying the network alone is worse than it — i.e. the gain belongs "
        f"entirely to the L2 the fields receive, an ability ridge that "
        f"re-gauges the fit rather than regularizing the network. A fig-guard's "
        f"figure for the sentence declining to adopt the decay"))
    out.append(Figure(
        f"{PANEL}.shipped_wander_range_mnat", m["wander_range"], ".2f",
        f"the resolution floor every contrast here is read against: the shipped "
        f"arm's own peak-to-peak held-out spread across the last third of ITS "
        f"OWN run, millinats, {over}. One configuration at one seed, so it is "
        f"wander and nothing else"))
    return out


# --- the table ----------------------------------------------------------------

def main():
    m = measure()
    lo, hi, n = m["tail_epochs"]
    print(f"\nregularization panel — methodology §1.5, the two dropout sites and "
          f"the weight decay offered as their substitute")
    print(f"{len(CONFIGS)} configurations x {len(SEEDS)} seeds {list(SEEDS)}, "
          f"full budget ({m['budget']} epochs, {m['schedule']} rate), one "
          f"machine ({m['gpu']}),")
    print(f"{m['rows']:,d} attempt rows per arm. Shipped dropout: {DROPOUT} in "
          f"the trunk, {INPUT_DROPOUT} over the feature columns; shipped weight "
          f"decay: none.")
    print(f"Scored on HELD-OUT PROBLEMS (stop_unit={STOP_UNIT!r}) — the row "
          f"split leaves every problem partly trained and")
    print(f"is the one split that cannot see memorization. Held aside: "
          + ", ".join(f"s{s} {m['stop_rows'][s]:,d} rows" for s in SEEDS) + ".")
    print(f"'tail' is the mean over the {n} checkpoints from epoch {lo} to "
          f"{hi}; 'final' is epoch {m['budget']} alone.")

    print(f"\n{'configuration':>16}"
          + "".join(f"{f'tail s{s}':>11}{f'final s{s}':>11}" for s in SEEDS)
          + "   what it is")
    for c in CONFIGS:
        print(f"{c:>16}"
              + "".join(f"{m['level'][(c, s)]:>11.5f}{m['final'][(c, s)]:>11.5f}"
                        for s in SEEDS)
              + f"   {CONFIG_NOTE[c]}")
    print(f"  held-out cross-entropy in nats per attempt, over problems no arm "
          f"trained on.")

    print(f"\nheld-out cost against the shipped fit AT THE SAME SEED, millinats "
          f"(positive = worse than shipped)")
    head = "".join(f"{'s' + str(s) + ' tail':>11}{'s' + str(s) + ' final':>12}"
                   for s in SEEDS)
    print(f"{'contrast':>16}{head}{'mean tail':>12}{'mean final':>12}"
          f"{'signs':>9}{'(final)':>10}")
    for c in CONTRASTS:
        k = m["cost"][c]
        cells = "".join(f"{t:>11.2f}{f:>12.2f}"
                        for t, f in zip(k["tail"], k["final"]))
        print(f"{c:>16}{cells}{k['tail_mean']:>12.2f}{k['final_mean']:>12.2f}"
              f"{('agree' if k['sign_consistent'] else 'DIFFER'):>9}"
              f"{('agree' if k['final_sign_consistent'] else 'differ'):>10}")
    print(f"  Each seed's columns are PAIRED (the same held-out problems). The "
          f"two seeds hold aside DIFFERENT")
    print(f"  problems, so they are two independent estimates of one delta and "
          f"are averaged, never subtracted.")
    print(f"  'signs' is the figure; '(final)' is the same test on one "
          f"checkpoint and is context only.")

    print(f"\nSUPER-ADDITIVITY — the excess over additivity, millinats. "
          f"both_off cost MINUS ({' + '.join(SINGLE_SITES)}) costs;")
    print(f"positive = the two removals compound, 0 = they merely add, "
          f"negative = they overlap.")
    print(f"{'reading':>16}" + "".join(f"{f'seed {s}':>12}" for s in SEEDS)
          + f"{'mean':>12}")
    print(f"{'tail':>16}" + "".join(f"{v:>12.2f}" for v in m["excess_tail"])
          + f"{m['excess_tail_mean']:>12.2f}")
    print(f"{'final':>16}" + "".join(f"{v:>12.2f}" for v in m["excess_final"])
          + f"{m['excess_final_mean']:>12.2f}")
    for i, s in enumerate(SEEDS):
        print(f"    seed {s}: {BOTH_SITES} "
              f"{m['cost'][BOTH_SITES]['tail'][i]:.2f} "
              + " ".join(f"| {k} {m['cost'][k]['tail'][i]:.2f}"
                         for k in SINGLE_SITES)
              + f"  ->  excess {m['excess_tail'][i]:+.2f}")
    print(f"  positive at EVERY seed: "
          f"{'YES' if m['excess_every_seed'] else 'NO'}"
          f"   <- the figure. An excess that is negative is a measurement: it "
          f"says the")
    print(f"  two removals overlap, and it is §1.5's sentence rather than this "
          f"panel that would need rewriting.")

    print(f"\nDOES WEIGHT DECAY SUBSTITUTE FOR DROPOUT? {DECAY_ONLY!r} against "
          f"{BOTH_SITES!r}, both against the shipped fit,")
    print(f"paired within a seed. Positive gap = decay left the dropout-free fit "
          f"worse still; negative = it")
    print(f"recovered part of what dropout was doing.")
    print(f"{'seed':>16}{'decay_both_off':>16}{'both_off':>12}{'gap':>10}")
    for i, s in enumerate(SEEDS):
        print(f"{s:>16}{m['cost'][DECAY_ONLY]['tail'][i]:>16.2f}"
              f"{m['cost'][BOTH_SITES]['tail'][i]:>12.2f}"
              f"{m['subst_gap'][i]:>10.2f}")
    print(f"{'mean':>16}{m['cost'][DECAY_ONLY]['tail_mean']:>16.2f}"
          f"{m['cost'][BOTH_SITES]['tail_mean']:>12.2f}"
          f"{m['subst_gap_mean']:>10.2f}")
    print(f"  decay is NO BETTER than nothing at every seed: "
          f"{'YES' if m['no_substitute'] else 'NO'}   <- the figure")

    print(f"\nthe shipped fit's own held-out wander over epochs {lo}..{hi} — why "
          f"the tail reading is the headline")
    for s in SEEDS:
        print(f"{'seed ' + str(s):>16}{m['wander'][s]['sd']:>9.2f} mnat sd"
              f"{m['wander'][s]['range']:>10.2f} mnat peak-to-peak")
    print(f"{'averaged range':>16}{m['wander_range']:>19.2f} mnat   <- the "
          f"figure; a contrast read at one checkpoint carries this on top of "
          f"itself.")

    print(f"\nOut of this panel's reach: §1.5's MECHANISM (a low-norm readout "
          f"still fingerprinting) is not measured —")
    print(f"no weight norm is read and no fingerprint counted, only the held-out "
          f"consequence. The decay is ONE value,")
    print(f"so 'no decay value substitutes' is not available; and Adam's "
          f"weight_decay reaches the free per-user fields")
    print(f"too, so the decay arms also put an L2 on ability, which the shipped "
          f"design leaves unpenalized. §1.5's")
    print(f"column-vs-block granularity and dose-response claims have no arm "
          f"here.\n")


if __name__ == "__main__":
    main()
