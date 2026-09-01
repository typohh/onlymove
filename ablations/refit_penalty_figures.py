#!/usr/bin/env python3
"""The rating-orthogonality penalty's figures (methodology §1.5), read from the
`penalty` refit panel.

A refit-tier producer never trains: it reads the arm records a panel run wrote
under `dev_runs/<panel>/` (`ablations.run_panel`, locally or via
`vast_orchestration.py ablate`) and hard-fails when the panel is missing, short
of budget, running a config other than the manifest's, internally inconsistent,
or STALE against the current attempt log — the pins rule `docs/mirt_benchmark.md`
states, mechanized. Guards and spans keep evaluating against the last-measured
ledger values, so the panel's cost is paid only when it is deliberately re-run.

Five configurations: the shipped per-tail penalty, the whole-axis form, the
penalty removed, a free control carrying neither the penalty nor the teaching
pull, and `ortho_rating_only` — the shipped per-tail form with its target
narrowed to the RATING column alone, which is §1.5's displacement arm.

EVERY CONFIGURATION RUNS AT EVERY SEED THE MANIFEST DECLARES, and this producer
is built around that. Both kinds of quantity here need the replicate for the
same reason: a per-fit tail statistic has no spread at all from one fit, and the
shipped configuration's own held-out score wanders across the closing third of
its budget by more than the contrast this panel is asked about. The seed drives
the initialization AND the held-aside draw, so WITHIN one seed every arm scores
the same stop rows and a cross-arm delta is paired, while two seeds are two
independent paired estimates of it. Nothing below ever subtracts across seeds: a
contrast is taken inside a seed, the seeds are averaged afterwards, and their
SIGN agreement is reported separately.

A REFUTED CLAIM IS A MEASUREMENT, NOT A BROKEN INPUT. Every direction this panel
exists to establish is emitted as a 0/1 indicator rather than asserted, so a fit
that comes out the other way names the sentence that needs rewriting instead of
aborting the run. Hard failure is kept for inputs: a record missing, short of
budget, mismatched against its manifest entry, unbalanced across the seeds,
unpaired inside a seed, or stale against the attempt log.

  penalty.whole_axis_reported      the whole-axis arm's worst WHOLE-AXIS |corr|
                                   against the shipped triple — the statistic
                                   that arm optimises, and the one it would be
                                   read as rating-orthogonal on.
  penalty.whole_axis_tail_leak     the SAME arm's worst PER-TAIL |cosine|. The
                                   two together are §1.5's understatement pair:
                                   a linear statistic averages an axis whose two
                                   ends are laden in opposite directions to zero
                                   and is satisfied completely, while both named
                                   ends stay laden.
  penalty.whole_axis_understatement  their ratio, the fold-understatement, taken
                                   per fit and then meaned.
  penalty.whole_axis_understates   1 when every seed's whole-axis arm carries
                                   more on its ends than its own whole-axis
                                   statistic reports — the mechanism, as a
                                   direction rather than a magnitude.
  penalty.per_tail_leak            the shipped arm's worst per-tail |cosine| —
                                   what scoring the ends separately cuts the
                                   worst of them to. This is the panel's own
                                   measurement of the quantity the live fit
                                   reports as `axes.rating_leak_worst`; main()
                                   prints the two side by side as context.
  penalty.per_tail_beats_whole     1 when the shipped arm's worst end is below
                                   the whole-axis arm's at every seed — the
                                   per-tail form's stated purpose, as a
                                   direction.
  penalty.unpenalized_tail_leak    the same statistic with the penalty removed:
                                   what the axes absorb when nothing holds them
                                   off the rating side.

  The displacement arm (`ortho_rating_only`), which exists for §1.5's sentence
  "Closing difficulty alone only displaces the leak: each axis then finds a
  different parameter to absorb". Four magnitudes and one indicator, so the
  sentence can be checked or refuted rather than taken:

  penalty.rating_only_rating_leak  its worst per-tail |cosine| against the
                                   RATING — the one channel it closes.
  penalty.rating_only_open_leak    its worst per-tail |cosine| against SLOPE or
                                   FLOOR — the channels it leaves open, which
                                   the displacement claim says are where the
                                   leak reappears.
  penalty.shipped_rating_leak      the same two quantities under the SHIPPED
  penalty.shipped_open_leak        arm, where all three channels are closed —
                                   what closing all three buys over closing one.
  penalty.leak_displaces           the displacement claim itself, 0/1.

  penalty.free_rating_corr         the free control's worst whole-axis |corr|
  penalty.free_slope_corr          against the RATING and against SLOPE. §1.5's
                                   signature is the second exceeding the first —
                                   an unheld axis spent on a shipped response
                                   parameter, and on the slope rather than on
                                   difficulty — so both sides of that comparison
                                   are figures.
  penalty.free_leak_is_slope       that signature, 0/1.
  penalty.free_stop_tail_cost_mnat the free control's held-out (stop-split) cost
                                   against the shipped arm of its OWN seed, over
                                   the last third of the budget, in millinats —
                                   the headline reading, being the level the arm
                                   settled at.
  penalty.free_stop_cost_mnat      the same paired cost read at the final epoch.
  penalty.free_stop_cost_sign_consistent
                                   1 only when every seed agrees in sign on BOTH
                                   readings — an indicator meant for a
                                   `<!-- fig-guard: KEY == 1 -->` beside a
                                   sentence naming the direction of that cost,
                                   never for a span.

ONE STATISTIC ACROSS THE LIVE FIT AND THE PANEL. Both leakage quantities are
computed by `ablations.axes_fit_figures`'s own `leakage` and `whole_axis`,
imported rather than restated, so the hinge (`utils.rating.tail_agreement`), the
centring and the normalisation are single-sourced and a panel arm and the
shipped fit cannot drift into being scored by two different numbers.

TWO POPULATION CHOICES, both inherited from that producer rather than re-decided
here. Leakage runs over EVERY fitted problem of the arm, unweighted — what a fit
ships is one loading per problem over the whole corpus, and that is the
population a client draws from; the trainer optimises the same quantity
attempt-weighted over attempted problems, which is a different number. And a
"worst" is the maximum over both ends of all K axes against all three shipped
quantities (rating, slope, floor), the whole grid the penalty is a sum of squares
over — never one axis or one quantity, except where a restriction to a column IS
the claim: the free control's slope and rating signatures, and the displacement
arm's split between the channel it closes and the channels it leaves open.

The held-out column is reported for every arm and only the free control's is a
figure. §1.5 states outright that the per-end form's held-out cost against the
whole-axis form is NOT established, so a key for that pair would back a claim the
document declines to make.

    ./.venv/bin/python3 -m ablations.refit_penalty_figures
"""

import json
import math
import os
import sqlite3
import statistics
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, load_ledger, require, panel_regime  # noqa: E402
from ablations.axes_fit_figures import leakage, whole_axis  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.features import load_backend_problem_params  # noqa: E402
from utils.rating.model_constants import K_AXES, SEED, SIZES  # noqa: E402
from utils.rating.tail_agreement import TAIL_HINGE  # noqa: E402

TIER = "refit"

PANEL = "penalty"
PANEL_DIR = os.path.join(PROJECT_ROOT, "ablations", "dev_runs", PANEL)
MANIFEST = os.path.join(PROJECT_ROOT, "ablations", "panels", f"{PANEL}.json")
# One record per CONFIGURATION per seed. The manifest names a replicate by
# suffixing the configuration it repeats, which is what lets every delta be
# matched to the baseline of its own seed.
CONFIGS = ("shipped", "ortho_whole", "ortho_off", "free_control",
           "ortho_rating_only")
BASELINE = CONFIGS[0]
DISPLACEMENT = "ortho_rating_only"
# The column order `load_backend_problem_params` returns, and the order the
# penalty's own target matrix is built in — which is also why the displacement
# arm's target is the leading column: it takes `rsf[:, :1]`.
QUANTITIES = ("rating", "slope", "floor")
RATING_COL = QUANTITIES.index("rating")
SLOPE_COL = QUANTITIES.index("slope")
# The channels the displacement arm leaves out of its target, and therefore the
# ones §1.5 says the leak moves onto.
OPEN_COLS = tuple(j for j in range(len(QUANTITIES)) if j != RATING_COL)
# The live fit's own reading of `penalty.per_tail_leak`'s quantity. Printed as
# context, never required: the two are separate fits of the same configuration
# and §1.14's seed spread is real, so agreement is reassurance and disagreement
# is a question rather than a failure.
LIVE_KEY = "axes.rating_leak_worst"


# --- the panel record ---------------------------------------------------------

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


# What a held-aside score is scored ON, as the noun the printout uses. Read off
# the records rather than assumed: a held-aside ROW is one more attempt on a
# problem the fit has otherwise seen, a held-aside PROBLEM one the trunk alone
# has to place, and a reader cannot tell the two apart from the number.
UNIT_NOUN = {"row": "rows", "problem": "problems"}


def panel_unit(arms):
    """The holdout unit every arm of the panel scored on, or a hard failure if
    they differ — a contrast averaged over both units is between the units as
    much as between the arms."""
    units = {r["config"].get("stop_unit", "row") for r in arms.values()}
    require(len(units) == 1,
            f"the panel's arms held aside {sorted(units)} — those are different "
            f"quantities and a contrast pooled over them is not one contrast")
    unit = units.pop()
    require(unit in UNIT_NOUN, f"unknown stop unit {unit!r}")
    return unit


def load_panel():
    """Every arm's record, checked for the things that make the panel quotable:
    one budget, the manifest's own config per arm, pins agreeing across arms,
    and a snapshot still matching the current attempt log."""
    require(os.path.exists(MANIFEST),
            f"no {MANIFEST} — the panel manifest is what defines these arms")
    with open(MANIFEST) as f:
        manifest = json.load(f)
    suffixes = replicate_suffixes(manifest)
    arm_names = tuple(c + suf for suf in suffixes for c in CONFIGS)

    arms = {}
    for arm in arm_names:
        path = os.path.join(PANEL_DIR, f"{arm}.json")
        require(os.path.exists(path),
                f"no {path} — the penalty panel has not run at every seed. Run "
                f"it with `python -m ablations.run_panel penalty` (or "
                f"`vast_orchestration.py ablate penalty`) first; the resume "
                f"unit is the arm, so the arms already recorded are skipped.")
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
        require(all(math.isfinite(float(t["stop_bce"]))
                    for t in r["metrics"]["trace"]),
                f"{arm}'s held-out trace is not finite throughout — a blown-up "
                f"arm is a different finding from the ones below, not a larger "
                f"version of them")

    # THE ARM RAN THE MANIFEST'S CONFIG. Every claim below is about a named knob
    # setting, so an arm whose record disagrees with the manifest it is named
    # after — a manifest edited after the run, or a hand-invoked arm dropped into
    # the panel directory — is measuring something else under this arm's name.
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

    # The SEED is what the replicates vary, and no pin carries it (see
    # `ablation_mirt.snapshot_pins`), so this loop spans every arm of every
    # replicate unchanged — everything it names has to hold across the whole
    # panel for any of it to be one panel.
    base = arms[BASELINE]["pins"]
    for arm, r in arms.items():
        for pin in ("db_max_id", "gpu", "labels_sha", "proxies_sha",
                    "blocks", "pca_dim", "k_axes"):
            require(r["pins"][pin] == base[pin],
                    f"{arm}'s pin {pin} ({r['pins'][pin]!r}) differs from the "
                    f"shipped arm's ({base[pin]!r}) — the arms are not one "
                    f"panel and their comparison is void")
    require(int(base["k_axes"]) == K_AXES,
            f"the panel was fitted at K={base['k_axes']}, this tree ships "
            f"K_AXES={K_AXES} — the axis grid the worsts range over has changed")

    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        now = int(con.execute("SELECT MAX(id) FROM tsumego_reports").fetchone()[0])
    finally:
        con.close()
    require(base["db_max_id"] == now,
            f"the panel was measured at attempt-log frontier {base['db_max_id']} "
            f"but the log now ends at {now} — the stop-row draw moved, the "
            f"paired comparison is void. Re-run the panel.")
    return arms, budget, schedule, unit, seed_groups(arms, suffixes)


def seed_groups(arms, suffixes):
    """The panel regrouped into replicates: one entry per seed, holding that
    seed's arm name for every configuration.

    A group is ONE held-aside draw. The seed selects the stop rows, so a delta
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

    # THE PAIRING, CHECKED RATHER THAN ASSUMED. One seed at one attempt log
    # gives one stop split, and equal row / user / stop-row counts are what that
    # identity looks like from the record. The bank has to match too, or the
    # arms' axes are not the same axes by index.
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
            require(arms[arm]["metrics"]["bank_spec"] == ship["bank_spec"],
                    f"{arm} was taught a different rule bank from seed "
                    f"{g['seed']}'s shipped arm — its axes are not the same "
                    f"axes by index")
    return groups


# --- one arm's fit, as artifacts ----------------------------------------------

def load_arm_fit(arm):
    """The arm's own loadings and its own shipped triple, joined BY ID.

    Both files are the arm's, never production's: `<arm>/axes/per_problem.csv`
    and `<arm>/ratings/<size>/tsumego_*.csv`. The three triple CSVs are each
    sorted by their own value, so they are read through
    `load_backend_problem_params` — the shared loader, keyed by tsumego id —
    rather than stacked positionally, which would pair a problem's rating with a
    stranger's slope and be well-formed the whole way."""
    import pandas as pd
    per = os.path.join(PANEL_DIR, arm, "axes", "per_problem.csv")
    require(os.path.exists(per),
            f"no {per} — the {arm} arm's record exists but its axes do not")
    df = pd.read_csv(per)
    cols = [f"embed_{k}" for k in range(K_AXES)]
    missing = [c for c in cols + ["tsumego_id", "size"] if c not in df.columns]
    require(not missing, f"{per} lacks {missing}")
    extra = [c for c in df.columns if c.startswith("embed_") and c not in cols]
    require(not extra, f"{per} carries {extra} past K_AXES={K_AXES}")
    G = df[cols].to_numpy(np.float64)
    require(np.all(np.isfinite(G)), f"{per}: non-finite loading")
    # The gauge divides every loading by the corpus RMS of its own axis, so a
    # loading is in axis-standard-deviation units. Checked per arm, because a
    # collapsed axis under a freed arm is exactly the thing that would silently
    # rescale the correlations below.
    rms = np.sqrt((G ** 2).mean(axis=0))
    require(np.all(np.abs(rms - 1.0) < 0.01),
            f"{arm}: axis loadings are not standardized: per-axis RMS "
            f"{rms.round(4)}")

    keys = list(zip(df["size"].astype(int), df["tsumego_id"].astype(str)))
    ratings_dir = os.path.join(PANEL_DIR, arm, "ratings")
    require(os.path.isdir(ratings_dir),
            f"no {ratings_dir} — the {arm} arm wrote no per-problem triple")
    params = load_backend_problem_params(tuple(SIZES), ratings_dir=ratings_dir)
    absent = [t for _, t in keys if t not in params]
    require(not absent,
            f"{arm}: {len(absent)} fitted problems have no triple (e.g. "
            f"{absent[:3]}) — the arm's ratings/ and axes/ are from different "
            f"runs")
    tri = np.array([params[t] for _, t in keys], dtype=np.float64)
    require(np.all(np.isfinite(tri)), f"{arm}: a rating/slope/floor is not finite")
    require(np.all(tri[:, 1] > 0.0), f"{arm}: a slope is not positive")
    require(np.all((tri[:, 2] >= 0.0) & (tri[:, 2] < 1.0)),
            f"{arm}: a floor is outside [0, 1)")
    return {"keys": keys, "G": G, "tri": tri}


# --- the measurement ----------------------------------------------------------

def opposite_end_pair(ends):
    """The (axis, quantity) whose two ends lean OPPOSITE ways hardest — the cell
    where the whole-axis statistic's blindness is visible outright, since the two
    are what it averages toward zero. None when no cell has opposing ends."""
    best = None
    for k in range(ends["+"].shape[0]):
        for j in range(len(QUANTITIES)):
            p, n = float(ends["+"][k, j]), float(ends["-"][k, j])
            if p * n >= 0:
                continue
            if best is None or abs(p) + abs(n) > best[2]:
                best = (k, j, abs(p) + abs(n), p, n)
    return best


def same_sign(*series):
    """1 when every seed's reading lands on the same side of zero in each of the
    given series, 0 otherwise. A reading of exactly zero agrees with nothing, so
    it is never consistent."""
    return int(all(all(v > 0 for v in s) or all(v < 0 for v in s)
                   for s in series))


def all_true(flags):
    """1 when a per-seed direction holds at every seed, 0 otherwise. An empty
    panel cannot establish a direction, but `replicate_suffixes` has already
    refused one."""
    return int(bool(flags) and all(flags))


def arm_stats(fit, record):
    """One arm's leak grid and its final held-out score. Everything a per-fit
    figure below is read off, computed once per arm and averaged over the seeds
    afterwards — never averaged first, since the worsts are maxima."""
    G, tri = fit["G"], fit["tri"]
    ends = leakage(G, tri)
    tails = np.stack([ends["+"], ends["-"]])          # (2, K, 3)
    w = whole_axis(G, tri)                            # (K, 3)
    at = np.abs(tails)
    aw = np.abs(w)
    return {
        "ends": ends, "tails": tails, "whole": w,
        "tail_worst": float(at.max()),
        "tail_where": np.unravel_index(int(at.argmax()), tails.shape),
        "whole_worst": float(aw.max()),
        "whole_where": np.unravel_index(int(aw.argmax()), w.shape),
        # Per-quantity worsts, which is the split the displacement claim and the
        # free control's signature are both read on.
        "tail_by_q": [float(at[:, :, j].max()) for j in range(len(QUANTITIES))],
        "whole_by_q": [float(aw[:, j].max()) for j in range(len(QUANTITIES))],
        "stop_bce": float(record["metrics"]["trace"][-1]["stop_bce"]),
    }


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


def measure():
    arms, budget, schedule, unit, groups = load_panel()
    fits = {arm: load_arm_fit(arm) for arm in arms}
    keys = fits[groups[0]["arms"][BASELINE]]["keys"]
    for arm, fit in fits.items():
        require(fit["keys"] == keys,
                f"{arm} fitted a different problem set from the shipped arm — "
                f"the leakage populations are not comparable")

    stats = {arm: arm_stats(fits[arm], arms[arm]) for arm in arms}

    # The last third of the budget: past the stretch where these fits still
    # improve, so a mean over it reads the level an arm settled at rather than
    # whichever point of its wander the final checkpoint happened to land on.
    tail_from = 2 * budget // 3
    curves = {arm: stop_curve(r) for arm, r in arms.items()}
    grid = curves[groups[0]["arms"][BASELINE]][0]
    for arm, (ep, _) in curves.items():
        require(ep == grid,
                f"{arm}'s trace is scored at different epochs than the shipped "
                f"arm's — the deltas would not be like for like")
    tail = [i for i, e in enumerate(grid) if e >= tail_from]
    require(len(tail) > 1,
            f"the trace carries {len(tail)} checkpoint(s) at or past epoch "
            f"{tail_from} — the settled level and the wander around it are read "
            f"over the last third of the budget, and neither exists on a single "
            f"point")

    # Per-fit properties: one reading per seed, in seed order, never pooled
    # before the maximum is taken.
    def per_fit(cfg, pick):
        return [pick(stats[g["arms"][cfg]]) for g in groups]

    per_cfg = {}
    for cfg in CONFIGS:
        per_cfg[cfg] = {
            "tail_worst": per_fit(cfg, lambda s: s["tail_worst"]),
            "whole_worst": per_fit(cfg, lambda s: s["whole_worst"]),
            "tail_rating": per_fit(cfg, lambda s: s["tail_by_q"][RATING_COL]),
            # The two channels the displacement arm leaves out of its target,
            # read as ONE open channel: the claim is that the leak lands on a
            # parameter other than the rating, not on a particular one of them.
            "tail_open": per_fit(
                cfg, lambda s: max(s["tail_by_q"][j] for j in OPEN_COLS)),
            "whole_rating": per_fit(cfg, lambda s: s["whole_by_q"][RATING_COL]),
            "whole_slope": per_fit(cfg, lambda s: s["whole_by_q"][SLOPE_COL]),
            # Kept per quantity as well, for the table's seed-averaged row.
            "tail_q": [[stats[g["arms"][cfg]]["tail_by_q"][j] for g in groups]
                       for j in range(len(QUANTITIES))],
        }

    # The understatement is a RATIO OF TWO WORSTS OF ONE FIT, so it is taken per
    # fit and meaned afterwards; a ratio of the two seed-means is a different
    # quantity. A worst whole-axis correlation of exactly zero over the whole
    # grid is a degenerate fit rather than a finding, and it leaves the ratio
    # with no denominator.
    wa = per_cfg["ortho_whole"]
    for i, (t, w) in enumerate(zip(wa["tail_worst"], wa["whole_worst"])):
        require(w > 0.0,
                f"the whole-axis arm at seed {groups[i]['seed']} reports a "
                f"worst whole-axis correlation of exactly zero — the "
                f"understatement ratio has no denominator")
    understatement = [t / w for t, w in zip(wa["tail_worst"], wa["whole_worst"])]

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

    # The resolution floor every millinat contrast above is read against: what
    # the SHIPPED configuration does on its own over the same window. Printed
    # rather than emitted — `barrier.shipped_wander_sd_mnat` is the key that
    # states this quantity, and one claim wants one key.
    wander = []
    for g in groups:
        _, ship_bce = curves[g["arms"][BASELINE]]
        tv = [ship_bce[i] * 1e3 for i in tail]
        wander.append({"seed": g["seed"], "sd": statistics.stdev(tv),
                       "range": max(tv) - min(tv)})

    ro, sh = per_cfg[DISPLACEMENT], per_cfg[BASELINE]
    free = per_cfg["free_control"]
    return {
        "arms": arms, "stats": stats, "groups": groups,
        "seeds": [g["seed"] for g in groups],
        "budget": budget,
        "schedule": schedule,
        "unit_noun": UNIT_NOUN[unit],
        "tail_epochs": (grid[tail[0]], grid[tail[-1]], len(tail)),
        "n_problems": len(keys),
        "gpu": arms[groups[0]["arms"][BASELINE]]["pins"]["gpu"],
        "bank_spec": list(arms[groups[0]["arms"][BASELINE]]
                          ["metrics"]["bank_spec"]),
        "per_cfg": per_cfg,
        "understatement": understatement,
        "contrasts": contrasts,
        "wander": wander,
        # THE THREE DIRECTIONS, per seed. Each is a comparison between two
        # quantities this panel measured — no constant enters any of them.
        "whole_understates": [t > w for t, w in
                              zip(wa["tail_worst"], wa["whole_worst"])],
        "per_tail_better": [s < w for s, w in
                            zip(sh["tail_worst"], wa["tail_worst"])],
        # DISPLACEMENT AS §1.5 STATES IT: closing the rating alone leaves the
        # leak on the channels it did not close (larger than what it did close),
        # and larger than where closing all three lands it.
        "displaces": [o > r and o > so for o, r, so in
                      zip(ro["tail_open"], ro["tail_rating"], sh["tail_open"])],
        # §1.5's free-control signature: the unheld axis goes to SLOPE rather
        # than to difficulty.
        "free_is_slope": [s > r for s, r in
                          zip(free["whole_slope"], free["whole_rating"])],
        "live": load_ledger()["figures"].get(LIVE_KEY),
    }


def figures():
    m = measure()
    p = m["per_cfg"]
    mean = statistics.fmean
    over = f"averaged over the panel's {len(m['seeds'])} seeds"
    free = m["contrasts"]["free_control"]
    return [
        Figure("penalty.whole_axis_reported", mean(p["ortho_whole"]["whole_worst"]),
               ".2f",
               "worst whole-axis |correlation| between an axis and one of the "
               "shipped per-problem quantities (rating, slope, floor) in the "
               "arm penalized on the whole axis — the statistic that arm "
               f"optimises, over every fitted problem. A per-fit property, "
               f"{over}"),
        Figure("penalty.whole_axis_tail_leak", mean(p["ortho_whole"]["tail_worst"]),
               ".2f",
               "the same arm's worst PER-TAIL |cosine|: one hinged axis END "
               "against one centred shipped quantity, the ends being the only "
               f"part of an axis a name is read off; {over}"),
        Figure("penalty.whole_axis_understatement", mean(m["understatement"]),
               ".1f",
               "how many fold the whole-axis statistic understates what the "
               "ends of the same fit carry — the second figure over the first, "
               f"taken WITHIN each fit and then {over}"),
        Figure("penalty.whole_axis_understates",
               all_true(m["whole_understates"]), "d",
               "1 when the whole-axis arm's ends carry more rating than its own "
               "whole-axis statistic reports, at every seed — §1.5's claim that "
               "the linear form hides the dependence, as a direction rather "
               "than a magnitude. A 0 refutes the reason the per-tail form "
               "exists and is a sentence to rewrite, not a fault in the panel"),
        Figure("penalty.per_tail_leak", mean(p["shipped"]["tail_worst"]), ".2f",
               "worst per-tail |cosine| under the SHIPPED penalty, which scores "
               "the two ends separately — what penalizing per tail cuts the "
               f"worst end to, measured on the panel's own shipped arms, {over}"),
        Figure("penalty.per_tail_beats_whole", all_true(m["per_tail_better"]),
               "d",
               "1 when the shipped per-tail arm's worst end sits below the "
               "whole-axis arm's at every seed — the per-tail form's stated "
               "purpose, as a direction. A 0 says the two forms are not "
               "separated by this panel"),
        Figure("penalty.unpenalized_tail_leak", mean(p["ortho_off"]["tail_worst"]),
               ".2f",
               "worst per-tail |cosine| with the rating-orthogonality penalty "
               "removed — what the axes absorb when nothing holds them off the "
               f"shipped quantities; {over}"),
        Figure("penalty.rating_only_rating_leak",
               mean(p[DISPLACEMENT]["tail_rating"]), ".2f",
               "worst per-tail |cosine| against the RATING in the arm whose "
               "penalty targets the rating column ALONE — the one channel it "
               f"closes, so this is the closure working; {over}"),
        Figure("penalty.rating_only_open_leak",
               mean(p[DISPLACEMENT]["tail_open"]), ".2f",
               "worst per-tail |cosine| against SLOPE or FLOOR in that same arm "
               "— the channels its penalty leaves out of the target, where "
               f"§1.5 says the leak reappears; {over}"),
        Figure("penalty.shipped_rating_leak", mean(p["shipped"]["tail_rating"]),
               ".2f",
               "the rating-column half of that pair under the SHIPPED penalty, "
               f"which targets all three quantities; {over}"),
        Figure("penalty.shipped_open_leak", mean(p["shipped"]["tail_open"]),
               ".2f",
               "the slope-or-floor half under the SHIPPED penalty — what "
               "closing all three channels leaves on the two the rating-only "
               f"arm abandons, and the reference the displacement is read "
               f"against; {over}"),
        Figure("penalty.leak_displaces", all_true(m["displaces"]), "d",
               "1 when, at EVERY seed, the rating-only arm's worst end against "
               "a channel it leaves open (slope or floor) is larger BOTH than "
               "its own worst end against the rating, the channel it closes, "
               "AND than the shipped arm's worst open-channel end at the same "
               "seed. That is §1.5's displacement claim as stated: closing "
               "difficulty alone moves the leak onto another shipped parameter "
               "instead of removing it, which is why the penalty names all "
               "three. Every term is a quantity this panel measured and no "
               "threshold enters. A 0 refutes the sentence"),
        Figure("penalty.free_rating_corr",
               mean(p["free_control"]["whole_rating"]), ".2f",
               "largest |whole-axis correlation| with the RATING in the free "
               "control (no rating-orthogonality penalty, no teaching pull) — "
               f"the difficulty side of §1.5's signature; {over}"),
        Figure("penalty.free_slope_corr", mean(p["free_control"]["whole_slope"]),
               ".2f",
               "largest |whole-axis correlation| with SLOPE in the free control "
               "— an unheld axis spent on a shipped response parameter. §1.5 "
               "reads the signature as this exceeding the figure above; "
               f"{over}"),
        Figure("penalty.free_leak_is_slope", all_true(m["free_is_slope"]), "d",
               "1 when the free control's largest whole-axis correlation with "
               "SLOPE exceeds its largest with the rating at every seed — "
               "§1.5's 'on the slope rather than on difficulty'. A 0 says the "
               "unheld axis went to difficulty, or to the floor, and the "
               "sentence naming slope is what needs rewriting"),
        Figure("penalty.free_stop_tail_cost_mnat", free["tail_mean"], ".1f",
               "the free control's held-out (stop-split) cost against the "
               "shipped arm over the last third of the budget, millinats — "
               "PAIRED against the shipped arm of its own seed, then meaned "
               f"over the seeds. The headline reading: it is the level the arm "
               f"settled at, which a final checkpoint alone samples only once. "
               f"Negative means the free control fits BETTER held-out"),
        Figure("penalty.free_stop_cost_mnat", free["final_mean"], ".1f",
               "the same paired held-out cost read at the FINAL epoch — the fit "
               f"that would actually ship, {over}"),
        Figure("penalty.free_stop_cost_sign_consistent", free["sign_consistent"],
               "d",
               "1 when every seed puts the free control's held-out cost on the "
               "same side of zero on BOTH readings, 0 when the seeds disagree. "
               "An indicator for a `fig-guard: "
               "penalty.free_stop_cost_sign_consistent == 1` beside any "
               "sentence naming the direction of that cost, never for a span"),
    ]


# --- the table ----------------------------------------------------------------

def _axis_names(spec):
    return [s[:20] for s in spec]


def _tail_table(name, st, names):
    print(f"\n  {name}")
    print(f"    {'axis':<22}{'+rating':>9}{'+slope':>8}{'+floor':>8}"
          f"{'-rating':>10}{'-slope':>8}{'-floor':>8}   {'whole-axis r,s,f':>22}")
    for k in range(K_AXES):
        print(f"    {names[k]:<22}"
              + "".join(f"{v:>9.3f}" for v in st["ends"]["+"][k])
              + "".join(f"{v:>{10 if j == 0 else 8}.3f}"
                        for j, v in enumerate(st["ends"]["-"][k]))
              + "   " + " ".join(f"{v:+.3f}" for v in st["whole"][k]))


def _verdict(flag):
    return "YES" if flag else "NO"


def main():
    m = measure()
    stats, groups, seeds = m["stats"], m["groups"], m["seeds"]
    p = m["per_cfg"]
    names = _axis_names(m["bank_spec"])
    free_names = [f"axis {k}" for k in range(K_AXES)]
    lo, hi, npts = m["tail_epochs"]
    mean = statistics.fmean

    print("\npenalty panel — methodology §1.5, the rating-orthogonality term")
    print(f"{len(CONFIGS)} configurations x {len(seeds)} seeds {seeds} at the "
          f"panel's budget of {m['budget']} epochs at a {m['schedule']} rate, "
          f"one machine ({m['gpu']}),")
    print(f"{m['n_problems']:,d} fitted problems per arm, identical problem set "
          f"throughout.")
    print(f"A contrast is taken WITHIN a seed (same held-out {m['unit_noun']}, "
          f"paired) and the seeds are averaged after;")
    print(f"'tail' is the mean over the {npts} checkpoints from epoch {lo} to "
          f"{hi}, 'final' is epoch {m['budget']}.")
    print("\nLeakage is measured over EVERY fitted problem, unweighted, by the "
          "same two\nfunctions ablations.axes_fit_figures uses on the shipped "
          f"fit: a per-tail entry is\nthe cosine between one hinged axis end "
          f"(z >= {TAIL_HINGE}) and one centred shipped\nquantity, a whole-axis "
          "entry the plain correlation. A 'worst' is the maximum\nover both "
          f"ends of all {K_AXES} axes against the named quantities.")

    print("\nper-fit leak statistics (each arm on its own, no contrast)")
    print(f"{'arm':<20}{'seed':>5}   {'per-tail worst by quantity':^26}"
          f"{'any':>8}{'whole any':>11}{'ratio':>8}")
    print(f"{'':<20}{'':>5}   {'rating':>8}{'slope':>9}{'floor':>9}")
    for cfg in CONFIGS:
        for i, seed in enumerate(seeds):
            st = stats[groups[i]["arms"][cfg]]
            ratio = (st["tail_worst"] / st["whole_worst"]
                     if st["whole_worst"] > 0 else float("nan"))
            print(f"{cfg:<20}{seed:>5}   "
                  + "".join(f"{v:>{8 if j == 0 else 9}.3f}"
                            for j, v in enumerate(st["tail_by_q"]))
                  + f"{st['tail_worst']:>8.3f}{st['whole_worst']:>11.3f}"
                  f"{ratio:>8.2f}")
        print(f"{cfg:<20}{'mean':>5}   "
              + "".join(f"{mean(col):>{8 if j == 0 else 9}.3f}"
                        for j, col in enumerate(p[cfg]["tail_q"]))
              + f"{mean(p[cfg]['tail_worst']):>8.3f}"
              f"{mean(p[cfg]['whole_worst']):>11.3f}")
    print("  Every cell is a worst over both ends of all axes, so a row is "
          "meaned across the seeds\n  and never pooled before the maximum is "
          "taken.")
    print("  ratio = worst per-tail / worst whole-axis, both over the same "
          "axis x quantity grid.")

    print("\nheld-out cost against the shipped arm OF THE SAME SEED, millinats "
          "(positive = worse)")
    head = "".join(f"{f's{s} final':>12}{f's{s} tail':>12}" for s in seeds)
    print(f"{'arm':>20}{head}{'mean final':>13}{'mean tail':>12}{'signs':>9}")
    for cfg, c in m["contrasts"].items():
        cells = "".join(f"{f:>12.2f}{t:>12.2f}"
                        for f, t in zip(c["final"], c["tail"]))
        print(f"{cfg:>20}{cells}{c['final_mean']:>13.2f}{c['tail_mean']:>12.2f}"
              f"{('agree' if c['sign_consistent'] else 'DIFFER'):>9}")
    print("  'signs' agree only where every seed lands on the same side of zero "
          "on both readings.")
    print("  Only the free control's cost is a figure: §1.5 states the per-end "
          "form's held-out cost\n  against the whole-axis form is not "
          "established, so a key for that pair would back a\n  claim the "
          "document declines to make.")
    print(f"\n  the panel's resolution floor — the shipped arm's own held-out "
          f"spread over epochs {lo}..{hi}:")
    for w in m["wander"]:
        print(f"{'seed ' + str(w['seed']):>20}{w['sd']:>10.2f} mnat sd"
              f"{w['range']:>11.2f} mnat peak-to-peak")
    print("  No contrast smaller than this is resolved by one arm per "
          "configuration.")

    print("\n=== the understatement: what a whole-axis penalty reports vs what "
          "its ends carry ===")
    for i, seed in enumerate(seeds):
        st = stats[groups[i]["arms"]["ortho_whole"]]
        s_t, k_t, j_t = st["tail_where"]
        k_w, j_w = st["whole_where"]
        print(f"  seed {seed}: reported (whole-axis) {st['whole_worst']:.3f} at "
              f"{names[k_w]} / {QUANTITIES[j_w]}; ends carry "
              f"{st['tail_worst']:.3f}\n          at {names[k_t]} / "
              f"{QUANTITIES[j_t]} ({'+' if s_t == 0 else '-'} end) — "
              f"{m['understatement'][i]:.1f}-fold.")
        op = opposite_end_pair(st["ends"])
        if op is None:
            print("          no (axis, quantity) cell has ends leaning opposite "
                  "ways — the mechanism is\n          present as magnitude "
                  "only.")
        else:
            k, j, _, pv, nv = op
            print(f"          the mechanism outright: {names[k]} carries "
                  f"{pv:+.3f} on its + end and {nv:+.3f} on its\n          - "
                  f"end against {QUANTITIES[j]}, which a linear statistic "
                  f"averages toward zero\n          (whole-axis "
                  f"{st['whole'][k, j]:+.3f}).")
    print(f"  ends carry more than the whole-axis statistic at every seed: "
          f"{_verdict(all_true(m['whole_understates']))}")
    print(f"  the shipped per-tail arm's worst end beats the whole-axis arm's "
          f"at every seed: "
          f"{_verdict(all_true(m['per_tail_better']))}")
    for i, seed in enumerate(seeds):
        _tail_table(f"the whole-axis arm, seed {seed}, per axis:",
                    stats[groups[i]["arms"]["ortho_whole"]], names)
    for i, seed in enumerate(seeds):
        _tail_table(f"the shipped per-tail arm, seed {seed}, per axis:",
                    stats[groups[i]["arms"]["shipped"]], names)

    print("\n=== with the penalty removed ===")
    for i, seed in enumerate(seeds):
        st = stats[groups[i]["arms"]["ortho_off"]]
        s, k, j = st["tail_where"]
        print(f"  seed {seed}: worst end {st['tail_worst']:.3f} — {names[k]}'s "
              f"{'+' if s == 0 else '-'} end against {QUANTITIES[j]}, against "
              f"{p['shipped']['tail_worst'][i]:.3f}\n          under the "
              f"shipped penalty.")

    print("\n=== closing the rating alone: does the leak displace? ===")
    print("  The rating-only arm keeps the shipped per-tail form and narrows "
          "its target to the\n  rating column, leaving slope and floor "
          "unpenalized. §1.5 claims each axis then finds\n  a different shipped "
          "parameter to absorb.")
    print(f"{'':>6}{'rating-only':^35}{'shipped':^35}".rstrip())
    print(f"{'seed':>6}{'closed: rating':>17}{'open: slope/floor':>18}"
          f"{'closed: rating':>17}{'open: slope/floor':>18}{'displaces':>11}")
    for i, seed in enumerate(seeds):
        print(f"{seed:>6}{p[DISPLACEMENT]['tail_rating'][i]:>17.3f}"
              f"{p[DISPLACEMENT]['tail_open'][i]:>18.3f}"
              f"{p['shipped']['tail_rating'][i]:>17.3f}"
              f"{p['shipped']['tail_open'][i]:>18.3f}"
              f"{_verdict(m['displaces'][i]):>11}")
    print(f"{'mean':>6}{mean(p[DISPLACEMENT]['tail_rating']):>17.3f}"
          f"{mean(p[DISPLACEMENT]['tail_open']):>18.3f}"
          f"{mean(p['shipped']['tail_rating']):>17.3f}"
          f"{mean(p['shipped']['tail_open']):>18.3f}"
          f"{_verdict(all_true(m['displaces'])):>11}")
    print("  'displaces' is YES at a seed when the rating-only arm's open "
          "channel carries more than\n  both its own closed channel and the "
          "shipped arm's open channel at that seed.")
    for i, seed in enumerate(seeds):
        _tail_table(f"the rating-only arm, seed {seed}, per axis:",
                    stats[groups[i]["arms"][DISPLACEMENT]], names)

    print("\n=== the free control (no penalty, no teaching) ===")
    print("  Its axes are taught nothing, so they carry no rule names; they are "
          "indexed.")
    for i, seed in enumerate(seeds):
        _tail_table(f"the free control, seed {seed}, per axis:",
                    stats[groups[i]["arms"]["free_control"]], free_names)
    for i, seed in enumerate(seeds):
        st = stats[groups[i]["arms"]["free_control"]]
        col = st["whole"][:, SLOPE_COL]
        k = int(np.abs(col).argmax())
        rat = st["whole"][:, RATING_COL]
        kr = int(np.abs(rat).argmax())
        print(f"  seed {seed}: worst against SLOPE axis {k} at {col[k]:+.3f}, "
              f"that axis sitting at {st['whole'][k, RATING_COL]:+.3f} against "
              f"the rating;\n          worst against the RATING is axis {kr} at "
              f"{rat[kr]:+.3f}.")
    print(f"  §1.5's signature — slope above difficulty — at every seed: "
          f"{_verdict(all_true(m['free_is_slope']))}")
    print(f"  held-out cost against the shipped arm: "
          f"{m['contrasts']['free_control']['tail_mean']:+.2f} mnat over the "
          f"last third,\n  "
          f"{m['contrasts']['free_control']['final_mean']:+.2f} mnat at the "
          f"final epoch, both seed-averaged.")

    print("\n=== context: the panel's shipped arms against the live fit ===")
    live = m["live"]
    if live is None:
        print(f"  the ledger holds no {LIVE_KEY} to compare against.")
    else:
        print(f"  {LIVE_KEY} (the production fit, read back by "
              f"ablations.axes_fit_figures)\n  is {float(live['value']):.3f}; "
              f"this panel's shipped arms read "
              + ", ".join(f"{v:.3f}" for v in p["shipped"]["tail_worst"])
              + f" (mean {mean(p['shipped']['tail_worst']):.3f})\n  on the same "
              f"statistic.")
        print(f"  Measured at db max id {live['measured']['db_max_id']} "
              f"(git {live['measured']['git']}) against this panel's "
              f"{m['arms'][groups[0]['arms'][BASELINE]]['pins']['db_max_id']}.")
        print("  These are separate fits of one configuration, so they are "
              "compared and never\n  required to agree — the seed spread "
              "between runs of this fit is real, and the two\n  shipped arms "
              "above measure it.")


if __name__ == "__main__":
    main()
