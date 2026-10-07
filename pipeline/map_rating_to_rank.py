"""Map the combined cross-board rating to a CONSENSUS rank, and fit the per-server
rank couplings — self-contained, from rank_surveys alone.

The rating model (pipeline.train_mirt) puts problem difficulty and user ability on
ONE shared logit scale across both boards, so the rating→rank curve is a SINGLE shared
curve. This step produces, from the survey declarations:

  rating_to_rank.json   per-board (identical) {ratings, ranks} + a top-level
                        `time_adjust` block (the jointly-fit solve-time dock).
                        `ranks` is on the ONE project rank scale — 0 is the 1k/1d
                        boundary, every rank one unit wide (pwa/shared/rank-math.js).
  rank_map.json         per-server consensus↔server rank couplings — one per server,
                        EGF INCLUDED (EGF is not special; see below).

written to ratings/ (the backend). package_for_pwa is the single writer of the served
pwa/tsumegos/ — it copies these two curve files across.

THE MODEL — a latent CONSENSUS rank, every server (EGF included) a coupling off it:

    time-adjusted rating ──curve M──▶ CONSENSUS rank ──per server──▶ server rank

  * The user's rating is their 80%-CLEAR rating (CALIBRATION_P), pooled across boards
    (`fit_user_rating`, full 4PL with each problem's own slope/floor + the global
    ceiling λ) — the SAME quantity the client computes, so the loop stays closed.
  * ONE ANCHOR PER STABLE PROFILE. A user's observations on a server — a declaration read
    as its band's CENTRE (a declared 4d is 4.5 dan), or a linked federation account's
    verified rating (the EGD/AGAR heartbeats) as a continuous position, counted while its
    player is active — collapse into runs (stable_runs: a new one opens only when an
    observation sits MIN_RANK_CHANGE from the run's mean, with a doubled threshold for a
    return toward the previous run), and their first-time attempts are cut HALFWAY between
    the last observation of one run and the first of the next; each span's attempts give
    one 80%-clear rating, anchored to that run's rank with weight 1 (declaration_spans,
    build_anchor_rows). A run holding verified ratings is anchored at their mean and
    supersedes the declarations beside it; the fit scores it as a POINT (density at the
    coupling's interpolated position, times the band's width), a declaration as a band
    (mass). This is the mapping's assumption — no rank drift unless an observation says
    otherwise — stated directly: several ratings from one span tied to one declared rank
    would read to the fit as noise at a fixed rank, flattening the curve where
    declarations are fewest. It replaced 2-month career blocks each tied to their nearest
    declaration, which held out no better and sat lower at the top. Each anchor counts
    once, not each user or pair: frequent surveyors and one-time declarers do disagree,
    but reweighting toward either moved the bias without improving the held-out fit
    (ablations.eval_rank_map_cv). A pair with no span of MIN_ATTEMPTS falls back to the
    pooled career rating against its latest observation.
  * RATINGS ARE TIME-ADJUSTED THROUGHOUT THE MAPPING. Each declaration's effective
    rating is `rating + α·(log10(median solve min) − center)`, with the slope α fit
    JOINTLY in the same objective (it is identified — the time feature is per-user
    while the curve is a function of rating and each coupling is a per-server constant,
    so neither can absorb a per-user time shift). The client reproduces α·(…) for every
    server translation; the ONLY place it drops the dock is the headline onlymove rank
    (a display choice — see pwa/js/rank-translation.js). α + center ride in
    rating_to_rank.json's `time_adjust` and fit_population_constants reads them through
    into population_constants.json (no separate fit). A SIMPLE log-time slope is all the
    survey data supports — no rating-interaction, no power-law (see
    docs/methodology.md), and no rank-residualization line (the joint
    couplings absorb the rank-correlated part of solve time on their own).
  * M: a SMOOTH MONOTONE curve with the EGD even_game_winrates.txt cubic SHAPE as a PRIOR.
    It is parameterised by its log-slope on a fixed rating grid (see fit_consensus for
    why not by the rating at each rank edge), and two ridges pull it toward the shape:
    LAMBDA_SHAPE on the log-slope's deviation from the shape's (up to a free scale),
    LAMBDA_SMOOTH on that deviation changing along the rating axis. The shape is the
    ΔAIC-best fit to the even-game win-rate table (utils.rating.rank_curve_shape) —
    EGD tournament outcomes between declared grades on 19×19 over the board, which is
    a DIFFERENT thing from the EGF coupling below: the shape is a prior on the curve,
    EGF is one server coupling off it, and neither privileges the other.
    The priors are STRONG but FINITE. Held out (ablations.eval_rank_map_cv), the survey
    does not yet support any departure from the shape: accuracy improves as the ridges
    tighten and then plateaus, and a weak ridge lets the curve fit noise at the sparse
    top. They are deliberately NOT made rigid (the shape with only a scale and an
    offset): the shape is 19×19 over-the-board evidence, unvalidated for online ranks
    or for onlymove's positions, so the curve must stay free to bend where enough
    survey data says otherwise — the ridges are fixed while the likelihood grows with
    every declaration.
  * EGF IS JUST ANOTHER SERVER. Every server, EGF included, gets free ordinal thresholds
    (a cumulative-link coupling, uniform-spacing prior SERVER_LAMBDA). The consensus
    rank is the hub; each server couples off it.
  * ROBUST TO JUNK DECLARATIONS. Each declaration's ordinal likelihood is a
    contamination mixture: with probability EPS_CONTAMINATION the declaration is
    junk (a lie/misclick — e.g. a lone Fox 6d declared on an account measuring
    ~19k), uniform over that server's categories. This caps a junk row's NLL and
    sends its gradient to 0 instead of letting it steer the coupling; honest
    extreme declarations (merely improbable, often cross-server-consistent) keep
    most of their weight. EPS IS DELIBERATELY SMALL: the paired CV
    (ablations.eval_rank_map_cv, the +rob variants, scored on honest declarations
    only via --drop-eval) shows held-out NLL degrades MONOTONICALLY in eps — the
    plain logistic's linear tails already bound junk influence, so a large junk
    floor lets the fit over-sharpen its couplings (worse calibration, point
    metrics flat across the whole range tried, including 1% and 1.5%). It is not
    zero because the junk SCREEN (detect_junk_declarations) reads this mixture's
    posterior, and because it caps any single junk row's pull at −log(eps/N) nats.
    Don't raise it without re-running that CV (`--rob-eps … --drop-eval …`).
  * THE GAUGE — TWO PRIORS, NOT ONE. With no server privileged the LIKELIHOOD leaves the
    consensus scale free up to a MONOTONE reparameterisation (remap the latent, absorb it
    into every coupling's free thresholds and the curve's free per-segment gaps). That is
    infinite-dimensional, so GAUGE_WEIGHT alone cannot pin it and never did:
      - SERVER_LAMBDA (the uniform-spacing prior pulling each coupling's spacings toward
        the baseline of 1 consensus unit per server rank) is what removes all but the
        AFFINE part, scale included — a non-affine remap unbalances the spacings of every
        coupling at once, a dilation scales them all. The curve pins nothing: its
        steepness (log β) is free and its shape prior is deliberately gentle.
      - GAUGE_WEIGHT then pins the one direction spacings are blind to, a rigid
        TRANSLATION: a VERY WEAK penalty on the mean coupling offset toward identity,
        where the mean runs over DECLARATIONS — each weighs the offset at the two
        boundaries of its declared band, across all servers in one sum, so a
        data-rich server counts for its declarations and a padded threshold nobody
        declared near counts for nothing. Averaging over threshold positions instead
        let the spacing prior's extrapolated tails steer the location (measured: it
        put the consensus a rank below eight of nine servers at 1d, balanced by OGS
        alone). Weak because the likelihood is flat along it, so the location has no
        motivation to drift — a strong pin is unneeded and would shrink the real
        per-server offsets.
    Both are soft: they select a preferred point on a flat manifold rather than removing
    the freedom. The data-rich servers set the scale, which lands ≈ EGF, so the headline
    barely moves from the previous EGF-anchored rank.

Run pipeline.fetch_database first to refresh db/d1.{sql,sqlite} (the survey +
attempts source); this step then needs neither GPU nor the heavy pipeline. Then publish
with pipeline.package_for_pwa.
"""

import argparse
import json
import csv
import os
import sqlite3
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.special import expit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.rank_surveys import (  # noqa: E402
    SERVERS, SQL_PATH_DEFAULT, VERIFIED_SERVERS, is_verified_active, load_rank_surveys,
    rank_band, value_to_rank, verified_rank_value, verified_tier_rank,
)
from utils.rating.rank_curve_shape import fit_shape, rank_to_axis  # noqa: E402
from utils.rating.fit_rank_curves import (  # noqa: E402
    CALIBRATION_P, fit_user_rating, interp_extrap_vec, junk_posterior_vec,
    ordinal_nll_vec, separated_head, _unpack_thresholds,
)
from utils.constants import ALL_BOARD_SIZES  # noqa: E402
from utils.rating.ratings_io import (  # noqa: E402
    FLOOR_CSV, RATINGS_CSV, SLOPE_CSV, load_lapse, load_value_map,
)

SIZES = list(ALL_BOARD_SIZES)

# --- Model config (baked; this module is production-only, no experimentation) ---
RATING_SHAPE = "cubic"      # even_game_winrates.txt shape used as the curve's prior MEAN
LAMBDA_SHAPE = 300.0        # ridge on the curve's log-slope deviating from the shape's (per rating)
LAMBDA_SMOOTH = 30.0        # ridge on that deviation CHANGING along the rating axis
RATING_KNOT_STEP = 0.05     # the curve's fixed rating-knot spacing (see fit_consensus)
RATING_KNOT_PAD = 0.4       # knots extend this far past the anchors' rating range
SERVER_LAMBDA = 5.0         # uniform-spacing prior on each server coupling
GAUGE_WEIGHT = 1.0          # VERY WEAK pin on the n-weighted mean coupling offset → 0
EPS_CONTAMINATION = 0.005   # P(a declaration is junk) — robust mixture; small ON PURPOSE
                            # (held-out NLL degrades monotonically in eps, see docstring)
LAMBDA_TIME = 0.1           # light ridge keeping the learned time slope bounded
# Rank BANDS (rank_surveys.rank_band): band b spans [b, b+1), so band 0 is 1d and
# band -1 is 1k, and a band's lower EDGE is the integer b itself. Every threshold
# below is such an edge.
EGF_LO, EGF_HI = -31, 9     # consensus-rank band range the curve spans (30k..9d); the low end extends itself to cover the lowest anchor
RANK_PADDING = 2            # extend a server's thresholds past its declared min/max
EXTEND_TO = 8               # force each server range to cover at least up to 9d
MIN_SERVER_SAMPLES = 3      # skip a server with less pooled declaration-weight
JUNK_FLOOR = 0.5            # exclude above this junk posterior — see the screening note
MIN_ATTEMPTS = 20           # drop survey users (and declaration spans) with fewer attempts
MIN_RANK_CHANGE = 0.5       # a profile splits only once a new observation sits this many ranks
                            # from the current profile's running mean (see stable_runs)
# The consensus fit runs to CONVERGENCE, not to a budget. The shown onlymove rank rides
# the direction the likelihood is flattest along (the couplings absorb the curve's
# moves, so server ranks barely change while the headline rank does), which is exactly
# where a budget-limited stop lands arbitrarily — at a point that moves with the data
# and the start. The fit therefore takes an exact gradient and Hessian from torch
# autograd and stops on the NEWTON DECREMENT rather than on the objective's relative
# change, aborting if it does not get there (see _minimize).
FIT_NEWTON_TOL = 1e-9       # converged: the Newton decrement (nats still to gain) is below this
FIT_NEWTON_STEPS = 50        # Newton converges in a handful; running out is a failure
FIT_GTOL = 1e-6              # L-BFGS's own inner stop (Newton finishes the job)
FIT_MAX_ITER = 200000        # a cap that never binds
FIT_HISTORY = 50             # L-BFGS curvature pairs kept
# The CPU, deliberately: an evaluation is a few dozen SMALL operations, where a GPU
# spends its time on kernel launches and the line search's per-iteration reads, not
# on arithmetic — measured slower than one CPU thread even fully device-resident.
FIT_DEVICE = "cpu"

# Solve-time feature (server side): non-repeat, timed, AFK-capped median over the
# most-recent WORKING_SET attempts, in minutes. This approximates but no longer
# exactly matches the client — the client's timeAdjustDeltaRating now uses a
# positional-weighted working-set median with no AFK cap (a deliberate divergence),
# so treat this as the mapping's own solve-time definition, not a strict mirror.
AFK_CAP_MS = 180000
MIN_TIMED = 20
WORKING_SET = 3000


def load_ratings(ratings_dir, size):
    """{tsumego_id: rating} from ratings/<size>/tsumego_ratings.csv — the 80%-clear
    black-box scalar package_for_pwa ships."""
    return load_value_map(
        os.path.join(ratings_dir, str(size), RATINGS_CSV), "rating")


def load_slopes(ratings_dir, size):
    """{tsumego_id: a_p} from tsumego_slopes.csv — the raw 2PL discrimination
    (`quality`). fit_user_rating uses these so the survey 80%-clear ratings anchor M with
    the SAME per-problem slope the client estimator uses."""
    return load_value_map(
        os.path.join(ratings_dir, str(size), SLOPE_CSV), "slope")


def load_floors(ratings_dir, size):
    """{tsumego_id: c_p} from tsumego_floors.csv — the per-problem 4PL guessing floor,
    fed (with the global ceiling λ) through fit_user_rating to match the client's irt.js."""
    return load_value_map(
        os.path.join(ratings_dir, str(size), FLOOR_CSV), "floor")


def _shape_at(ys):
    """even_game_winrates.txt RATING_SHAPE value at each consensus-boundary rank in `ys`, the
    curve's prior mean. even_game_winrates.txt is fitted on its OWN axis (its origin is part
    of the polynomial family), so convert at the boundary via rank_to_axis."""
    params, theta = fit_shape(RATING_SHAPE)
    return np.asarray(theta(rank_to_axis(ys), params), dtype=float)


def load_user_median_solve_minutes(sqlite_path):
    """{user_prefix8: median novel solve minutes} over each user's working-set window —
    reproducing the client's medianNovelSolveMinutes() on getMergedAnswerHistory(): the
    most-recent WORKING_SET attempts unified across boards (ordered by created_at so
    backfilled rows land at their true position), then filtered to non-repeat + timed
    (0, AFK cap]. None below MIN_TIMED."""
    con = sqlite3.connect(sqlite_path)
    try:
        cur = con.execute(
            "SELECT user_id, is_repeat, time_used FROM tsumego_reports ORDER BY created_at, id")
        windows = defaultdict(lambda: deque(maxlen=WORKING_SET))
        for uid, is_repeat, t in cur:
            windows[uid[:8]].append((is_repeat, t))
    finally:
        con.close()
    out = {}
    for pref, rows in windows.items():
        ts = [float(t) for (rep, t) in rows
              if not rep and t is not None and 0 < float(t) <= AFK_CAP_MS]
        if len(ts) >= MIN_TIMED:
            out[pref] = float(np.median(ts)) / 60000.0
    return out


def load_timed_attempts(sqlite_path, user_ids):
    """{user_id: (created_days ascending np.array, [(tsumego_id, success), ...]
    aligned)} — first-time attempts with event times (the block anchors need them;
    the .sql regex loader doesn't carry timestamps)."""
    ids = sorted(user_ids)
    con = sqlite3.connect(sqlite_path)
    try:
        q = ("SELECT user_id, problem, success, created_at FROM tsumego_reports "
             "WHERE (is_repeat IS NULL OR is_repeat != 1) AND user_id IN ({})"
             .format(",".join("?" * len(ids))))
        df = pd.read_sql_query(q, con, params=ids)
    finally:
        con.close()
    df["t"] = (pd.to_datetime(df["created_at"], utc=True, format="mixed")
               .astype("int64") / 86400e9)
    df = df.sort_values("t")
    return {uid: (grp["t"].to_numpy(),
                  list(zip(grp["problem"].tolist(), grp["success"].astype(int).tolist())))
            for uid, grp in df.groupby("user_id")}


def observation_frame(surveys, use_verified=True):
    """The mapping's observations: human declarations and ACTIVE verified ratings on
    one frame, columns user_id, server, rank_value, created_at, t_days, verified.

    `rank_value` is a POINT on the project rank scale in both cases: a declaration is
    its band's CENTRE (a declared 4d is read as 4.5 dan, rank_to_value), a verified
    rating its own continuous position (gor_to_value / aga_rating_to_value). The
    machine heartbeat rows (EGD, AGAR) are re-keyed onto the system they rate (EGF,
    AGA), so a user's declaration and rating on one system share a series. A rating
    counts only while its player is active AS OF THE ROW (is_verified_active at the
    row's own date — a stale rating no longer describes the player, and a dropped
    row leaves the declaration to speak). Rows with no readable position are
    dropped. `use_verified=False` leaves the heartbeats out entirely, so every anchor
    is a declaration — the mapping any comparison of verified ratings against the fit
    has to run against, since a rating that anchors the fit is not independent of it."""
    rows = surveys[surveys["server"].isin(VERIFIED_SERVERS)]
    if not use_verified:
        rows = rows.iloc[0:0]
    human = surveys[~surveys["server"].isin(VERIFIED_SERVERS)
                    & surveys["rank_value"].notna()].copy()
    human["verified"] = False
    keep, n_inactive = [], 0
    for r in rows.itertuples(index=False):
        v = verified_rank_value(r.server, r.rank)
        if v is None:
            continue
        label = r.rank_label if isinstance(r.rank_label, str) else None
        last = r.last_game_at if isinstance(r.last_game_at, str) else None
        as_of = pd.Timestamp(r.created_at).date()
        if not is_verified_active(verified_tier_rank(r.server, r.rank, label), last, as_of):
            n_inactive += 1
            continue
        keep.append({"user_id": r.user_id, "server": VERIFIED_SERVERS[r.server],
                     "rank": r.rank, "rank_value": v, "created_at": r.created_at,
                     "verified": True})
    ver = pd.DataFrame(keep, columns=[*human.columns.drop("verified"), "verified"])
    out = pd.concat([human, ver], ignore_index=True)
    out["t_days"] = (pd.to_datetime(out["created_at"], utc=True, format="mixed")
                     .astype("int64") / 86400e9)
    print(f"  {len(human)} declarations + {len(ver)} active verified ratings "
          f"({n_inactive} inactive dropped) from {out['user_id'].nunique()} users")
    return out


def stable_runs(values, min_change=MIN_RANK_CHANGE):
    """Start index of each run of a (user, server) observation series — where the
    player's profile splits.

    A new run opens when an observation sits at least `min_change` ranks from the
    MEAN of the run it would join, and the hysteresis is two-fold. The reference is
    the run's mean, not its last value: a rating hovering either side of a band edge
    never reaches the threshold, where a rule keyed on the band label would split at
    every crossing. And a return toward the PREVIOUS run's level (within `min_change`
    of its mean) must clear twice the threshold, so a series stepping back and forth
    by about the threshold splits once and stays, instead of once per swing; a real
    reversal of a full rank or more still splits. A declaration differs from the next
    by whole ranks (its band's centre), so a declaration-only series splits exactly
    where its band changes."""
    starts, total, n, prev_mean = [0], float(values[0]), 1, None
    for i in range(1, len(values)):
        v, mean = float(values[i]), total / n
        returning = prev_mean is not None and abs(v - prev_mean) < min_change
        if abs(v - mean) >= min_change * (2 if returning else 1):
            starts.append(i)
            prev_mean, total, n = mean, v, 1
        else:
            total += v
            n += 1
    return starts


def declaration_spans(times, rows, decl_times, decl_values, rate, verified=None):
    """[(band, rating, n_attempts, v)] — one anchor per STABLE profile of one user on
    one server. The observations (ascending decl_times, their point values on the
    rank scale) collapse into runs (stable_runs); the user's history (`times`
    ascending, `rows` aligned) is cut HALFWAY between the last observation of one run
    and the first of the next, and each span's attempts give one 80%-clear rating via
    `rate(rows) -> rating or None`. The first span reaches back to the start of the
    history and the last forward to its end: an observation describes the player
    until another one says otherwise.

    A run holding verified ratings is anchored at their MEAN, as a continuous
    position `v`, and the declarations beside it are superseded; a run of
    declarations alone is anchored at its declared band (`v` None). `band` is the
    band containing the anchor either way. A span too thin to rate contributes
    nothing; an empty list means no span could be rated."""
    values = np.asarray(decl_values, float)
    ver = np.zeros(len(values), bool) if verified is None else np.asarray(verified, bool)
    starts = stable_runs(values)
    cuts = [(decl_times[i - 1] + decl_times[i]) / 2 for i in starts[1:]]
    edges = [-np.inf] + cuts + [np.inf]
    bounds = starts + [len(values)]
    out = []
    for k, i in enumerate(starts):
        member = slice(i, bounds[k + 1])
        v = float(values[member][ver[member]].mean()) if ver[member].any() else None
        band = rank_band(v if v is not None else values[i])
        lo = int(np.searchsorted(times, edges[k], side="left"))
        hi = int(np.searchsorted(times, edges[k + 1], side="left"))
        R = rate(rows[lo:hi]) if hi - lo >= MIN_ATTEMPTS else None
        if R is not None:
            out.append((band, R, hi - lo, v))
    return out


def build_anchor_rows(surveys, attempts, med_min, ratings, slopes, floors, lapse):
    """The mapping's anchor rows, one per stable profile (declaration_spans).

    surveys: observation_frame's output (`rank_value`, `t_days`, and `verified` — absent
    reads as all declarations); attempts: {user_id: (times, rows)} first-time attempts;
    med_min: {user_prefix8: median solve minutes}. Returns (decls, keys, users,
    n_fallback, center): rows (server, span rating, band, T, weight 1, v) as
    fit_consensus takes them — v the continuous position of a verified anchor, None
    for a declared band; their (user, server, band) keys, so an exclusion drops a
    declaration whole; {user_id: (pooled career rating, T)}; and how many (user,
    server) pairs had no ratable span and fell back to the pooled career rating
    against their latest observation; and the time dock's zero point. T is the log10
    median solve time less that center, 0 for a user without one (no time term)."""
    def rate(rows):
        R, n = fit_user_rating(rows, ratings, min_n=MIN_ATTEMPTS,
                               slopes=slopes, floors=floors, lapse=lapse)
        return R if R is not None and n >= MIN_ATTEMPTS else None
    pooled, logT = {}, {}
    for uid in surveys["user_id"].unique():
        R = rate(attempts.get(uid, (np.zeros(0), []))[1])
        if R is None:
            continue
        pooled[uid] = R
        if uid[:8] in med_min and med_min[uid[:8]] > 0:
            logT[uid] = float(np.log10(med_min[uid[:8]]))
    timed = [logT[u] for u in pooled if u in logT]
    center = float(np.mean(timed)) if timed else 0.0   # the time dock's zero point
    T_of = {u: (logT[u] - center) if u in logT else 0.0 for u in pooled}
    if "verified" not in surveys:
        surveys = surveys.assign(verified=False)
    decls, keys, n_fallback = [], [], 0
    for (uid, server), srows in (surveys[surveys["user_id"].isin(pooled)]
                                 .sort_values("t_days").groupby(["user_id", "server"])):
        times, rows = attempts[uid]
        values = srows["rank_value"].to_numpy(float)
        ver = srows["verified"].to_numpy(bool)
        spans = declaration_spans(times, rows, srows["t_days"].to_numpy(float),
                                  values, rate, ver)
        if not spans:
            last = stable_runs(values)[-1]
            v = float(values[last:][ver[last:]].mean()) if ver[last:].any() else None
            spans = [(rank_band(v if v is not None else values[last]),
                      pooled[uid], len(rows), v)]
            n_fallback += 1
        for g, R, _, v in spans:
            decls.append((server, R, g, T_of[uid], 1.0, v))
            keys.append((uid, server, g))
    users = {u: (pooled[u], T_of[u]) for u in pooled}
    return decls, keys, users, n_fallback, center


# ---------------------------------------------------------------------------
# The consensus fit
# ---------------------------------------------------------------------------
def _curve_arrays():
    egf_thr_ranks = list(range(EGF_LO + 1, EGF_HI + 1))
    ys = np.array([float(g) for g in egf_thr_ranks], dtype=float)  # band lower edges
    S = _shape_at(ys)
    dS = np.diff(S)
    return egf_thr_ranks, ys, S, dS


def _bucket(decls):
    """One coupling block per server (EGF included): rating, declared-category index k,
    the de-meaned time feature T, the identity-line target, the per-row weight w and,
    for a VERIFIED anchor, its continuous position in server ranks (`cont`, and
    `frac` — how far into its band it sits; 0 and unused for a declared band). A row
    is (server, rating, band, T[, weight[, v]]): weight defaults to 1, v to None
    (a declared band). `g` is a rank BAND index (see rank_surveys.rank_band), and for
    a verified row the band containing v."""
    rows_by = defaultdict(list)
    for server, R, g, T, *rest in decls:
        w = float(rest[0]) if rest else 1.0
        v = rest[1] if len(rest) > 1 else None
        if v is not None and int(g) != rank_band(v):
            raise ValueError(f"verified anchor at {v:.3f} carries band {g}, not {rank_band(v)}")
        rows_by[server].append((R, int(g), T, w, v))
    blocks = {}
    for server, rows in rows_by.items():
        if sum(r[3] for r in rows) < MIN_SERVER_SAMPLES:
            continue
        declared = [r[1] for r in rows]
        g_min = min(declared) - RANK_PADDING
        g_max = max(max(declared) + RANK_PADDING, EXTEND_TO)
        K = g_max - g_min
        if K < 1:
            continue
        N = g_max - g_min + 1
        gs = list(range(g_min + 1, g_max + 1))
        blocks[server] = {
            "g_min": g_min, "g_max": g_max,
            "baseline": np.ones(K - 1) if K > 1 else np.array([]),
            "identity": np.array([float(g) for g in gs], float),
            "rating": np.array([r[0] for r in rows], float),
            "T": np.array([r[2] for r in rows], float),
            "w": np.array([r[3] for r in rows], float),
            "cont": np.array([r[4] is not None for r in rows], bool),
            "frac": np.array([0.0 if r[4] is None else r[4] - r[1] for r in rows], float),
            "k": np.array([max(0, min(N - 1, r[1] - g_min)) for r in rows], int),
            "thr_w": _threshold_mass(rows, g_min, K)}
    return blocks


def _threshold_mass(rows, g_min, K):
    """Declaration mass beside each of the K thresholds: a declaration in band g sits
    between the boundaries at g and g+1 and lends half its weight to each. This is
    the weight the gauge pin uses, so the pin reads the coupling where players
    actually sit and never at the padded thresholds the spacing prior extrapolates."""
    tw = np.zeros(K)
    for _, g, _, w, _ in rows:
        for i in (g - g_min - 1, g - g_min):
            if 0 <= i < K:
                tw[i] += 0.5 * w
    return tw


def fit_device():
    """FIT_DEVICE, or the GPU when it is None and torch sees one."""
    import torch
    return FIT_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")


def fit_consensus(decls, identity_servers=()):
    """Jointly fit the rating→consensus curve M, the global time slope α, and the
    per-server couplings (EGF included). decls rows are (server, rating, declared
    band, T[, weight]) — weight defaults to 1 for a plain one-row-per-declaration
    anchor. Returns (a, alpha, ys, egf_thr_ranks, server_fits) where `a` are the
    rating values at each consensus band edge `ys` (the integer ranks the published
    curve carries).

    THE CURVE is parameterised by its SLOPE over a FIXED grid of rating knots
    (RATING_KNOT_STEP apart, spanning the anchors ± RATING_KNOT_PAD): at each knot a
    free log-slope θ_j (ranks per rating, positive ⇒ monotone), the slope linear
    between knots, the rank its integral from a free level, and constant end slopes
    beyond. The rank is then smooth in the rating AND in every parameter (the time
    slope α included, which slides each anchor along the rating axis), so the
    objective has no kinks and its gradient can genuinely vanish. Parameterising the
    rating at each rank edge instead — the obvious form — makes those edges free
    KNOTS the anchors cross, and the fit then snaps edges onto anchor clusters:
    bands collapse and swell, and no optimizer converges on the resulting kinks.
    The published knots are this smooth curve read off at the integer band edges.

    PRIORS on the curve: the log-slope's deviation δ_j from the even-game SHAPE's
    (RATING_SHAPE, up to a free scale) is ridge-penalised (LAMBDA_SHAPE) and so is
    its change between neighbouring knots (LAMBDA_SMOOTH); both are integrals over
    rating (×/÷ the knot step), so they do not depend on the grid's density.

    `identity_servers` holds a server's coupling AT IDENTITY instead of fitting it,
    which makes that server the privileged hub the latent is defined by rather than
    one coupling among many — the design whose cost docs/methodology.md §1.12
    states, measured by ablations.rank_map_arm_figures. A pinned coupling fixes the
    gauge on its own, so GAUGE_WEIGHT's translation pin (which exists only to pin it
    when nothing else does) is not applied. PRODUCTION PASSES NOTHING HERE and the
    default is the consensus fit, unchanged."""
    import torch
    decls = [(*d[:4], *(d[4:5] or (1.0,)), *(d[5:6] or (None,))) for d in decls]
    identity_servers = set(identity_servers)
    egf_thr_ranks, ys, _, _ = _curve_arrays()
    blocks = _bucket(decls)
    if not blocks:
        sys.exit("No servers with enough declarations — cannot fit.")
    dev = fit_device()
    f64 = dict(dtype=torch.float64, device=dev)
    tt = lambda v: torch.as_tensor(np.asarray(v, dtype=float), **f64)
    ti = lambda v: torch.as_tensor(np.asarray(v, dtype=np.int64), device=dev)

    # Fixed rating knots over the anchors, and the shape prior's log-slope as a
    # function of rank: rating ∝ S(rank) ⇒ d rank/d rating ∝ 1 / S'(rank).
    h = RATING_KNOT_STEP
    all_R = np.array([d[1] for d in decls], float)
    all_g = np.array([d[2] for d in decls], float)
    r = np.arange(np.floor((all_R.min() - RATING_KNOT_PAD) / h) * h,
                  all_R.max() + RATING_KNOT_PAD + h / 2, h)
    m = len(r) - 1
    shape_params, theta = fit_shape(RATING_SHAPE)
    def shape_log_slope(g):
        u = g + float(rank_to_axis(0.0))      # rank_to_axis(g): the shape's axis is rank + a constant
        return -torch.log((theta(u + 1e-3, shape_params) - theta(u - 1e-3, shape_params)) / 2e-3)

    # Parameters: [rank at r_0, b (the shape's free log-scale), θ_0..θ_m, α, servers...]
    beta0, c0 = np.polyfit(all_R, all_g, 1)
    beta0 = max(beta0, 1e-3)
    x0 = [c0 + beta0 * r[0], 0.0] + [float(np.log(beta0))] * (m + 1) + [0.0]
    i_alpha = 2 + m + 1
    srv_layout, idx = [], len(x0)
    for server, blk in blocks.items():
        if server in identity_servers:
            srv_layout.append((server, None, 0))   # held at identity: no free thresholds
            continue
        n = len(blk["baseline"]) + 1
        srv_layout.append((server, idx, n))
        x0.append(float(blk["g_min"]))
        x0.extend([0.0] * len(blk["baseline"]))
        idx += n
    x0 = np.asarray(x0, float)

    # Every server in ONE vectorized pass: rows concatenated with a server index,
    # thresholds as a padded matrix (padding repeats the last threshold, never indexed).
    servers = [srv for srv, _, _ in srv_layout]
    nS, Kmax = len(servers), max(len(blocks[srv]["identity"]) for srv in servers)
    start = np.zeros(nS, np.int64); free = np.zeros(nS, bool); Kvec = np.zeros(nS, np.int64)
    sp_idx = np.zeros((nS, Kmax - 1), np.int64); sp_mask = np.zeros((nS, Kmax - 1))
    base = np.zeros((nS, Kmax - 1)); ident = np.zeros((nS, Kmax)); thr_w = np.zeros((nS, Kmax))
    rating, T, k, w, sid, cont, frac = [], [], [], [], [], [], []
    for i, (srv, st, n) in enumerate(srv_layout):
        blk = blocks[srv]; K = len(blk["identity"]); Kvec[i] = K
        ident[i, :K] = blk["identity"]; ident[i, K:] = blk["identity"][-1]
        thr_w[i, :K] = blk["thr_w"]
        if st is not None:
            free[i] = True; start[i] = st
            sp_idx[i, :n - 1] = np.arange(st + 1, st + n); sp_mask[i, :n - 1] = 1.0
            base[i, :n - 1] = blk["baseline"]
        rating.append(blk["rating"]); T.append(blk["T"]); k.append(blk["k"])
        w.append(blk["w"]); sid.append(np.full(len(blk["k"]), i))
        cont.append(blk["cont"]); frac.append(blk["frac"])
    rating, T, w = tt(np.concatenate(rating)), tt(np.concatenate(T)), tt(np.concatenate(w))
    cont_t, frac_t = torch.as_tensor(np.concatenate(cont), device=dev), tt(np.concatenate(frac))
    k, sid = ti(np.concatenate(k)), ti(np.concatenate(sid))
    start_t, free_t, sp_idx_t = ti(start), torch.as_tensor(free, device=dev), ti(sp_idx)
    sp_mask_t, base_t, ident_t, thr_w_t = tt(sp_mask), tt(base), tt(ident), tt(thr_w)
    K_row = ti(Kvec)[sid]
    junk_row = EPS_CONTAMINATION / tt(Kvec + 1)[sid]     # float64: int/float in torch is float32
    r_t, zcol, w_tot = tt(r), torch.zeros((nS, 1), **f64), thr_w_t.sum()

    def curve(params):
        th = params[2:2 + m + 1]
        s = torch.exp(th)
        rk = torch.cat([params[0].reshape(1),
                        params[0] + torch.cumsum(h * (s[:-1] + s[1:]) / 2, 0)])
        return th, s, rk

    def rank_of(x, s, rk):
        j = torch.clamp(torch.floor((x.detach() - r[0]) / h).long(), 0, m - 1)
        u = x - r_t[j]
        inside = rk[j] + s[j] * u + (s[j + 1] - s[j]) * u * u / (2 * h)
        return torch.where(x < r_t[0], rk[0] + s[0] * (x - r_t[0]),
                           torch.where(x > r_t[-1], rk[-1] + s[-1] * (x - r_t[-1]), inside))

    def objective(params):
        th, s, rk = curve(params)
        alpha = params[i_alpha]
        x = rank_of(rating + alpha * T, s, rk)
        thr_free = params[start_t][:, None] + torch.cumsum(
            torch.cat([zcol, base_t * torch.exp(params[sp_idx_t]) * sp_mask_t], 1), 1)
        thr = torch.where(free_t[:, None], thr_free, ident_t)
        # ordinal_nll_vec with the contamination mixture
        t_hi = thr[sid, torch.minimum(k, K_row - 1)]
        t_lo = thr[sid, torch.clamp(k - 1, min=0)]
        up = torch.where(k < K_row, torch.sigmoid(t_hi - x), 1.0)
        lw = torch.where(k > 0, torch.sigmoid(t_lo - x), 0.0)
        body = torch.clamp(up - lw, 1e-12, 1.0)
        # A verified anchor is the same latent observed at a POINT rather than a band:
        # the coupling places it at the interpolated threshold position, and the
        # likelihood is the logistic density there times the band's width in consensus
        # units (the Jacobian) — whose integral over the band is the ordinal mass.
        z = torch.sigmoid(t_lo + frac_t * (t_hi - t_lo) - x)
        dens = torch.clamp(z * (1.0 - z) * (t_hi - t_lo), 1e-12)
        p = (1.0 - EPS_CONTAMINATION) * torch.where(cont_t, dens, body) + junk_row
        f = -torch.sum(w * torch.log(p))
        f = f + SERVER_LAMBDA * torch.sum((params[sp_idx_t] * sp_mask_t) ** 2)
        delta = th - (params[1] + shape_log_slope(rk))
        f = f + LAMBDA_SHAPE * h * torch.sum(delta ** 2)
        f = f + LAMBDA_SMOOTH / h * torch.sum((delta[1:] - delta[:-1]) ** 2)
        f = f + LAMBDA_TIME * alpha ** 2
        if not identity_servers:
            # gauge: every declaration weighs the offset at its own two boundaries
            f = f + GAUGE_WEIGHT * (torch.sum(thr_w_t * (thr - ident_t)) / w_tot) ** 2
        return f

    p = tt(x0)
    with torch.no_grad():   # the free scale b starts where the initial deviation is centred
        th, _, rk = curve(p)
        p[1] = torch.mean(th - shape_log_slope(rk))
    x_opt = _minimize(objective, p, dev)

    # Read the smooth curve off at the integer band edges (the published knots): dense
    # samples inside the grid, the constant end slopes beyond it.
    with torch.no_grad():
        _, s, rk = curve(tt(x_opt))
        xd = tt(np.linspace(r[0], r[-1], 20 * m + 1))
        rd = rank_of(xd, s, rk).cpu().numpy()
    xd, s, rk = xd.cpu().numpy(), s.cpu().numpy(), rk.cpu().numpy()
    # Past the outermost knots readers extrapolate the end BAND's slope, not the fitted
    # curve's, so an anchor out there would be scored on one curve and published on
    # another. The published knots therefore reach down as far as the lowest anchor
    # does: EGF_LO is where they start, not a limit on the data.
    x_anchor = (rating + float(x_opt[i_alpha]) * T).cpu().numpy()
    x_low = float(x_anchor.min())
    lowest = (float(np.interp(x_low, xd, rd)) if x_low >= xd[0]
              else float(rd[0] + (x_low - xd[0]) * s[0]))
    if lowest < ys[0]:
        egf_thr_ranks = list(range(int(np.floor(lowest)), EGF_HI + 1))
        ys = np.array([float(g) for g in egf_thr_ranks], dtype=float)
        print(f"  lowest anchor fits at rank {lowest:.2f}: published knots extended "
              f"down to {ys[0]:g}")
    a = np.interp(ys, rd, xd)
    a = np.where(ys < rd[0], xd[0] + (ys - rd[0]) / s[0], a)
    a = np.where(ys > rd[-1], xd[-1] + (ys - rd[-1]) / s[-1], a)
    alpha = float(x_opt[i_alpha])
    server_fits = {}
    for server, st, n in srv_layout:
        blk = blocks[server]
        thr = (blk["identity"] if st is None
               else _unpack_thresholds(x_opt[st:st + n], blk["baseline"]))
        gs = list(range(blk["g_min"] + 1, blk["g_max"] + 1))
        server_fits[server] = {gs[i]: float(thr[i]) for i in range(len(gs))}
    return a, alpha, ys, egf_thr_ranks, server_fits


def _minimize(objective, p0, dev):
    """L-BFGS, then Newton on the exact Hessian, both on `dev` with the solution read
    back once. CONVERGENCE is the Newton decrement — ½·gᵀH⁻¹g, the objective's
    predicted remaining improvement — reaching FIT_NEWTON_TOL. It is scale-free where
    a gradient threshold is not (the stronger the priors, the steeper the objective
    and the higher its rounding floor), and it cannot be met by a stop that merely
    ran out of progress: a parameter only a few declarations inform, far from its
    optimum, shows up in the decrement however little it moves the total. A fit
    whose Hessian is not positive definite, or whose decrement will not fall, aborts."""
    import torch
    p = p0.clone().requires_grad_(True)

    def closure():
        opt.zero_grad()
        f = objective(p)
        f.backward()
        return f
    # L-BFGS gets close cheaply; torch's gives up whenever its direction stops being a
    # descent direction, so it restarts with fresh curvature history until a round
    # stops lowering the objective.
    n_iter, prev = 0, float("inf")
    while True:
        opt = torch.optim.LBFGS([p], lr=1, max_iter=FIT_MAX_ITER, max_eval=FIT_MAX_ITER * 5,
                                tolerance_grad=FIT_GTOL, tolerance_change=0.0,
                                history_size=FIT_HISTORY, line_search_fn="strong_wolfe")
        opt.step(closure)
        n_iter += opt.state[opt._params[0]]["n_iter"]
        fv = float(closure())
        if not fv < prev or n_iter >= FIT_MAX_ITER:
            break
        prev = fv
    # Newton to convergence.
    x = p.detach().clone()
    for n_newton in range(FIT_NEWTON_STEPS + 1):
        xg = x.clone().requires_grad_(True)
        f = objective(xg)
        g, = torch.autograd.grad(f, xg)
        H = torch.autograd.functional.hessian(objective, x)
        L, info = torch.linalg.cholesky_ex(H)
        if int(info) != 0:
            sys.exit(f"consensus fit: Hessian not positive definite at the L-BFGS solution "
                     f"({n_iter} iterations, {n_newton} Newton steps) — the fit is not at a minimum")
        step = torch.cholesky_solve(-g.unsqueeze(1), L).squeeze(1)
        decrement = float(-(g @ step)) / 2
        if decrement <= FIT_NEWTON_TOL:
            break
        if n_newton == FIT_NEWTON_STEPS:
            sys.exit(f"consensus fit did not converge: Newton decrement {decrement:.2e} > "
                     f"{FIT_NEWTON_TOL:g} after {FIT_NEWTON_STEPS} Newton steps")
        t, fx = 1.0, float(f)
        while float(objective(x + t * step)) > fx + 1e-4 * t * float(g @ step):
            t /= 2
            if t < 1e-10:
                sys.exit(f"consensus fit: Newton line search failed (decrement {decrement:.2e})")
        x = x + t * step
    print(f"  converged on {dev}: objective {float(f):.4f}, Newton decrement {decrement:.1e}, "
          f"max |gradient| {float(g.abs().max()):.1e} ({n_iter} L-BFGS iterations + "
          f"{n_newton} Newton steps)")
    return x.cpu().numpy()


EXCLUSIONS_CSV = "declaration_exclusions.csv"


def detect_junk_declarations(decls, keys, a, ys, alpha, server_fits):
    """Declarations the fit itself calls junk, separated from the rest by a gap.

    A declaration whose author is describing a different player than the attempt
    log does — a claim typed for effect, a rank carried over from an account that
    is not this one, a registration grade never revisited — is not noise the
    likelihood should average over. The contamination mixture already caps what
    one can pull, and a capped pull is still a pull, applied by rows that are not
    measurements of anything.

    What separates them is not the size of the miss. Servers differ in how far
    apart their categories sit and in how far their populations spread, so a
    six-rank residual is unremarkable in one place and impossible in another;
    the fitted thresholds carry exactly that, which is why the statistic here is
    the mixture's own posterior rather than the residual (`junk_posterior_vec`).
    Each declaration is scored at the attempt-mass-weighted mean of its block
    rows' time-adjusted positions, the position the fit saw.

    The cut is the posterior's own decision boundary: above one half the mixture
    calls a declaration likelier junk than measurement. That bar is severe rather
    than nominal, because the junk prior is small — clearing it takes a declared
    category the fitted thresholds give a likelihood on the order of one in
    thousands, which is a claim no ordinary miss reaches.

    A knee is the tempting alternative and is not used as the cut. The largest
    gap in a sample is the largest gap somewhere even when the sample is one
    smooth population, so cutting there excludes down to wherever the decay
    flattens and reports the result as a discovered boundary; the gap is printed
    (`separated_head`) as a diagnostic of whether a genuinely detached group
    exists, which is a different question from where to stop.

    This runs ONCE against the first fit rather than iterating to convergence:
    each pass would tighten the curve onto the survivors and re-flag whatever now
    sits furthest out, converging on an arbitrarily clean set rather than on the
    truth.

    Returns (dropped, cut) — dropped is a list of dicts, one per excluded
    declaration, ordered by descending posterior; cut is (n, lowest excluded
    posterior, the separated head's gap)."""
    x_by_key, p_by_key, w_by_key = defaultdict(float), defaultdict(float), defaultdict(float)
    cat_by_key, v_by_key = {}, defaultdict(float)
    for d, key in zip(decls, keys):
        server, R, g, T, w = d[:5]
        v = d[5] if len(d) > 5 else None
        thr = server_fits.get(server)
        if not thr or len(thr) < 2:
            continue                      # no coupling: nothing to be junk against
        gs = sorted(thr)
        tau = np.array([thr[k] for k in gs], float)
        x = float(interp_extrap_vec(np.array([R + alpha * T]), a, ys)[0])
        # the fit's own category index: band g lies between the edges at g and g+1
        k = int(np.searchsorted(np.array([float(u) for u in gs]), g, side="right"))
        if v is None:
            p = float(junk_posterior_vec(tau, np.array([x]), np.array([k]),
                                         EPS_CONTAMINATION)[0])
        else:
            # the posterior of the same mixture the fit scores a verified anchor under
            lo, hi = tau[k - 1], tau[k]
            z = float(expit(lo + (v - g) * (hi - lo) - x))
            dens = max(z * (1.0 - z) * (hi - lo), 1e-12)
            junk = EPS_CONTAMINATION / (len(tau) + 1)
            p = junk / ((1.0 - EPS_CONTAMINATION) * dens + junk)
        x_by_key[key] += w * x
        p_by_key[key] += w * p
        w_by_key[key] += w
        v_by_key[key] += w * (v if v is not None else g + 0.5)
        cat_by_key[key] = (server, g, v is not None)

    rows = []
    for key, wsum in w_by_key.items():
        server, g, verified = cat_by_key[key]
        rows.append({"key": key, "server": server, "declared_band": g,
                     "internal": x_by_key[key] / wsum, "p_junk": p_by_key[key] / wsum,
                     "declared_value": v_by_key[key] / wsum, "verified": verified})
    if not rows:
        return [], (0, float("nan"), 0.0)

    rows.sort(key=lambda r: -r["p_junk"])
    n_cut = sum(1 for r in rows if r["p_junk"] > JUNK_FLOOR)
    n_knee, knee_at, gap = separated_head([r["p_junk"] for r in rows], floor=JUNK_FLOOR)
    print(f"\nDeclaration screening over {len(rows)} declarations: "
          f"{n_cut} above the {JUNK_FLOOR} junk posterior")
    if not n_cut:
        print("  none — nothing excluded")
        return [], (0, float("nan"), 0.0)
    print(f"  separated head ends after {n_knee} (largest gap {gap:.3f} below "
          f"{knee_at:.3f}); the cut is the {JUNK_FLOOR} boundary, at {n_cut}")
    for i, r in enumerate(rows[:n_cut + 2]):
        uid, server, g = r["key"]
        mark = "  cut" if i < n_cut else "  keep"
        head = " <- head" if i < n_knee else ""
        what = "verified" if r["verified"] else "declared"
        print(f"   {mark:<6}{uid[:8]:<10}{server:<9}{what} {value_to_rank(r['declared_value']):>4}  "
              f"we say {value_to_rank(r['internal']):>4}  P(junk)={r['p_junk']:.3f}{head}")
    return rows[:n_cut], (n_cut, float(rows[n_cut - 1]["p_junk"]), gap)


def write_exclusions(ratings_dir, dropped, cut, n_declarations):
    """The excluded declarations, for anything that scores the mapping against the
    same survey — an evaluation keeping rows the fit did not see is measuring a
    different comparison. Written every run, empty header included, so a reader
    never has to branch on the file's absence. The `p_junk` column is the value at
    the FIRST fit, the one the cut was taken on; `kind` says whether the row was a
    declaration or a verified rating, which a declarations-only refit cannot screen
    itself (ablations.rank_calibration_figures)."""
    path = os.path.join(ratings_dir, EXCLUSIONS_CSV)
    n_cut, threshold, gap = cut
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_prefix", "server", "declared_band", "declared_rank",
                    "our_internal", "our_rank", "p_junk", "kind"])
        for r in dropped:
            uid, server, g = r["key"]
            w.writerow([uid[:8], server, g, value_to_rank(r["declared_value"]),
                        f"{r['internal']:.4f}", value_to_rank(r["internal"]),
                        f"{r['p_junk']:.6f}",
                        "verified" if r["verified"] else "declared"])
    print(f"  wrote {path} ({len(dropped)} of {n_declarations} declarations"
          + (f", lowest excluded {threshold:.3f}; separated head ends "
             f"{gap:.3f} above it)" if dropped else ")"))
    return path


def emit_server_curves(server_fits):
    """Per-server interpolation knots {server: {our, server}} in scalar units — `our` =
    the consensus-rank scalar at each server-rank boundary, `server` the server scalar
    there. EGF is emitted like any other server (no identity special-case). In SERVERS
    order for a stable display."""
    out = {}
    for server in SERVERS:
        thr = server_fits.get(server)
        if not thr or len(thr) < 2:
            continue
        gs = sorted(thr)
        out[server] = {"our": [float(thr[g]) for g in gs],
                       "server": [float(g) for g in gs]}
    return out


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--sql", default=SQL_PATH_DEFAULT)
    ap.add_argument("--sqlite", default=DB_SQLITE)
    ap.add_argument("--no-verified", action="store_true",
                    help="Fit on declarations alone, ignoring the EGD/AGAR heartbeats. "
                         "For measurements that compare verified ratings against the "
                         "fit (ablations.rank_calibration_figures).")
    args = ap.parse_args(argv)

    print(f"Reading {args.sql} ...")
    surveys = load_rank_surveys(args.sql)
    if surveys is None or surveys.empty:
        sys.exit("No rank_surveys rows found — cannot fit the mapping.")
    # EVERY declaration anchors (weight 1 each; its blocks split that weight) —
    # a re-survey is a new anchor at its own time, not a replacement.
    surveys = observation_frame(surveys, use_verified=not args.no_verified)

    merged_ratings, merged_slopes, merged_floors = {}, {}, {}
    for s in SIZES:
        merged_ratings.update(load_ratings(args.dir, s))
        merged_slopes.update(load_slopes(args.dir, s))
        merged_floors.update(load_floors(args.dir, s))
    if not merged_ratings:
        sys.exit("No per-problem ratings found — run train_mirt first.")
    lapse = load_lapse(args.dir)
    # First-time attempts only: the client's rank estimator anchors on first-time
    # attempts, so build the survey anchor from novel problems too — repeats'
    # spaced-repetition success rate would otherwise inflate review-heavy users'
    # 80%-clear ratings. (Both boards land in one time-ordered list; the ratings
    # dicts are merged across sizes, so pooling is implicit.)
    attempts = load_timed_attempts(args.sqlite, set(surveys["user_id"].unique()))
    med_min = load_user_median_solve_minutes(args.sqlite)

    # One anchor per stable declaration (see the docstring and declaration_spans).
    print(f"\nEstimating {int(CALIBRATION_P * 100)}%-clear ratings per stable-declaration "
          f"span (pooled boards, 4PL, λ={lapse:.4f}) ...")
    decls, keys, users, n_fallback, center = build_anchor_rows(
        surveys, attempts, med_min, merged_ratings, merged_slopes, merged_floors, lapse)
    n_pairs = len({(u, srv) for u, srv, _ in keys})
    n_verified = sum(1 for d in decls if d[5] is not None)
    print(f"  {len(users)} users with a usable rating; {len(decls)} spans ({n_verified} verified) over {n_pairs} "
          f"(user, server) pairs ({n_fallback} fell back to the pooled career rating)")
    w_egf = sum(d[4] for d in decls if d[0] == "EGF")
    print(f"\nConsensus fit over {len(decls)} anchor rows [{w_egf:.0f} EGF, a "
          f"coupling like any server; cubic shape prior λ_shape={LAMBDA_SHAPE}, "
          f"server λ={SERVER_LAMBDA}, gauge weight={GAUGE_WEIGHT}]")

    a, alpha, ys, egf_thr_ranks, server_fits = fit_consensus(decls)
    print(f"  learned time slope α = {alpha:+.4f} rating per de-meaned log10-min")

    dropped, cut = detect_junk_declarations(decls, keys, a, ys, alpha, server_fits)
    write_exclusions(args.dir, dropped, cut, len(keys))
    if dropped:
        drop_keys = {d["key"] for d in dropped}
        decls = [d for d, k in zip(decls, keys) if k not in drop_keys]
        keys = [k for k in keys if k not in drop_keys]
        print(f"  refitting without them ({len(decls)} anchor rows remain)")
        a, alpha, ys, egf_thr_ranks, server_fits = fit_consensus(decls)
        print(f"  learned time slope α = {alpha:+.4f} rating per de-meaned log10-min")

    # In-sample EGF calibration check (EGF is now forward-mapped through its
    # coupling); weighted so it reads in declaration units, not block-row counts.
    egf = [(R, g, T, w, v) for srv, R, g, T, w, v in decls if srv == "EGF"]
    if "EGF" in server_fits and len(egf) >= 2 and len(server_fits["EGF"]) >= 2:
        gs = sorted(server_fits["EGF"])
        our = np.array([server_fits["EGF"][k] for k in gs], float)
        srv = np.array([float(k) for k in gs], float)
        order = np.argsort(our)
        R = np.array([e[0] for e in egf])
        # Declared BAND -> its centre, so the residual against a boundary-scale
        # prediction reads in the same units it always has.
        gdec = np.array([g + 0.5 if v is None else v for _, g, _, _, v in egf], float)
        T = np.array([e[2] for e in egf]); W = np.array([e[3] for e in egf])
        x_int = interp_extrap_vec(R + alpha * T, a, ys)
        pred = interp_extrap_vec(x_int, our[order], srv[order])
        res = pred - gdec
        print(f"  EGF-coupling calibration: RMSE {np.sqrt(np.average(res ** 2, weights=W)):.2f} | "
              f"bias {np.average(res, weights=W):+.2f} ranks")

    # --- write rating_to_rank.json (both board keys identical) + time_adjust ---
    # Knots sit at consensus-rank band LOWER EDGES, which under this convention are
    # the band integers themselves.
    curve = {"ratings": [float(r) for r in a],
             "ranks": [float(g) for g in egf_thr_ranks]}
    rr = {str(s): curve for s in SIZES}
    rr["time_adjust"] = {"alpha_per_log10_min": round(alpha, 6),
                         "center_log10_min": round(center, 6)}
    server_curves = emit_server_curves(server_fits)

    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, "rating_to_rank.json"), "w") as f:
        json.dump(rr, f, indent=2)
    with open(os.path.join(args.dir, "rank_map.json"), "w") as f:
        json.dump(server_curves, f, indent=2)

    fitted = [s for s in SERVERS if s in server_curves]
    print(f"\nconsensus range mapped: {value_to_rank(egf_thr_ranks[0])}→rating "
          f"{a[0]:.2f} ... {value_to_rank(egf_thr_ranks[-1])}→rating {a[-1]:.2f}")
    print(f"wrote {args.dir}/rating_to_rank.json (shared 9≡13 + time_adjust) + rank_map.json "
          f"({len(fitted)} server couplings incl EGF: {', '.join(fitted)})\n"
          f"  → publish with: python -m pipeline.package_for_pwa")


if __name__ == "__main__":
    main()
