#!/usr/bin/env python3
"""Validate the shipped recall (retention) model against the real attempt log.

WHAT THE MODEL CLAIMS
---------------------
`pwa/js/model.js` schedules a repeat when its predicted retention
`R(t) = exp(-t/S)` has decayed into the window `[0.60, 0.80]`, with
`S = exp(LN_C + KAPPA*(z - Z0) + GROWTH*r)` — z the cold-solve log-odds, r the
consecutive-success streak, LN_C the per-user base durability. Two consequences
are directly checkable against `tsumego_reports`, without needing anything the
client did not record:

  * A served recall should clear at the window-average retention, and that rate
    should be FLAT in r. The whole point of the GROWTH term is that a longer
    streak buys a proportionally longer interval, so rung 4 is served at the
    same predicted retention as rung 0. A rate that decays in r means stability
    grows faster than memory does; a rate that rises means the opposite.
  * `recallServeShare()` goes further and hardcodes `X_SERVE = 0.80` as the
    per-serve success probability at EVERY rung when it solves the volume
    equilibrium. That is the same flatness claim plus a level claim.

WHAT THIS SCRIPT MEASURES
-------------------------
  1. Served-recall clear rate, overall and split by rung r (Wilson intervals
     throughout), against the mean retention the model predicted for that rung.
  2. The same split by calendar month — client scheduling code changed over the
     logged era (the stability ceiling removal, the serve gate), so a pooled
     rate could hide a regime change.
  3. Decay within a rung: the closest the log comes to a retention curve.
  4. Observed vs predicted retention, binned by predicted R, with log-loss and
     Brier against a constant rate and the model's own 0.80 target.
  5. Where serves actually land in the window.
  6. Held-out value, splitting by user: refit constants, a Platt recalibration,
     a z-ablation, and a rung-only rate that uses no retention arithmetic.
  7. A forward test in the SHIPPED configuration: per-user LN_C fit on each
     user's earlier events, scoring their later ones.
  8. What the measured per-rung rates do to recallServeShare()'s arithmetic.

WHAT IT CANNOT MEASURE
----------------------
Every interval in the log was CHOSEN by the scheduler, so the predictor is
range-restricted by construction: the client only serves while its own R sits in
[0.60, 0.80]. A weak held-out score for R on this sample is therefore not the
same statement as "the retention curve is a bad model of memory" — the log
contains no unselected intervals with which to test that, and never will while
the scheduler picks them. What these checks do test is the operational question:
whether the policy achieves on its own serves what it is built to achieve. The
per-rung clear rate is the sharpest form of it, because equalizing that rate
across rungs is precisely what the GROWTH term exists to do.

RECONSTRUCTION CAVEATS (read before quoting a number)
-----------------------------------------------------
  * `z` is reconstructed the way `pipeline.fit_population_constants` builds it —
    a 3PL cold-solve log-odds from the user's fitted ability and the problem's
    shipped rating/slope/floor. The CLIENT instead reads z off its own trained
    per-user logistic (which also carries solving-time and review-time
    percentiles). The two agree in construction but not row by row, so the
    predicted-R checks measure the population model, not the exact number the
    device used at serve time. The observed clear rates (checks 1-2) are free of
    this: they are just outcomes.
  * Ratings are retrospective (docs/onlymove.md section 2.5): the
    problem parameters used here are today's, not the ones shipped on the day of
    the attempt.
  * Check 4 re-scores the same events the constants were fit on, so it is
    optimistic by construction. Checks 6 and 7 are the honest ones.
  * Chains whose first attempt predates REPEAT_LOG_CUTOFF are dropped by
    default: before it, repeats were not reliably logged, so the elapsed gap is
    not a true retention interval and r is not established. `--all-chains`
    keeps them for comparison.
  * Pairs closer together than MIN_RECALL_HOURS are dropped by default too, and
    for a different reason: the stability floor makes them UNSCHEDULABLE, so
    whatever produced them (a device serving a problem whose newer attempt it
    had not yet synced, a cross-tab race) is not the scheduler. They are counted
    and reported separately rather than silently discarded. `--keep-short`
    folds them back in.

USAGE
  ./.venv/bin/python3 -m auxiliary.eval_recall_model
  ./.venv/bin/python3 -m auxiliary.eval_recall_model --all-chains --min-n 50
"""

import argparse
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MODEL_INPUT_HISTORY_LENGTH = 500    # pwa/js/constants.js — the client's fit window

from pipeline.fit_population_constants import (  # noqa: E402
    FLOOR_MAX, REPEAT_LOG_CUTOFF, Z0,
    cold_solve_logit, fit_ability, load_problem_params, load_rows,
)

# Client scheduling constants (pwa/js/model.js) — mirrored, not re-derived.
WINDOW_OPEN_R = 0.80      # recall becomes due here
WINDOW_CLOSE_R = 0.60     # ...and leaves the window here
X_SERVE = 0.80            # recallServeShare()'s assumed per-serve success rate
MIN_RECALL_HOURS = 6.0
SHARE_WEEK_MS_S = 7 * 24 * 3600.0   # recallServeShare()'s trailing-volume window
SHARE_VOLUME_FRACTION = 0.5         # ...solved at this share of the daily average
WORKING_WINDOW = 3000               # RECALL_CANDIDATE_HISTORY_LENGTH, in attempts
LN_C_LAMBDA = 5.0         # client L2 on LN_C: lambda=10 mean-reduced == 5 on the sum
LN_S_TRAIN_MIN, LN_S_TRAIN_MAX = -5.0, 8.0


# ---------------------------------------------------------------------------
# Small stats helpers.
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial rate — well behaved at n small and
    at rates near 1, where the normal approximation is not."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def rate_line(label, k, n, width=22):
    if n == 0:
        return f"  {label:<{width}}      —        n=0"
    lo, hi = wilson(k, n)
    return (f"  {label:<{width}} {k / n:6.1%}   [{lo:5.1%}, {hi:5.1%}]   n={n}")


def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((y - p) ** 2))


# ---------------------------------------------------------------------------
# Event construction — mirrors the client's own bookkeeping.
# ---------------------------------------------------------------------------
def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def build_events(rows, params, abilities, all_chains=False):
    """One event per served repeat, carrying everything the checks need.

    successSinceFailure is walked EXACTLY as `user-stats.js` walks it, because
    the rung r a serve is scheduled at is the previous attempt's value and the
    two definitions differ on novel attempts:
      novel + correct  -> undefined (chain has never failed; not recall-eligible)
      novel + wrong    -> 0         (this is how a problem enters the ladder)
      repeat + wrong   -> 0
      repeat + correct -> prev + 1  (or 1 if the previous value was undefined)
    """
    events = []
    skipped = defaultdict(int)
    by_chain = defaultdict(list)
    for r in rows:
        by_chain[(r["user_id"], r["problem"], r["size"])].append(r)

    for (user, problem, size), chain in by_chain.items():
        chain.sort(key=lambda r: r["created_at"])
        if not all_chains and chain[0]["created_at"][:10] < REPEAT_LOG_CUTOFF:
            skipped["pre-cutoff chain"] += sum(1 for a in chain if a["is_repeat"])
            continue
        has_params = (size, problem) in params and user in abilities
        z = (cold_solve_logit(abilities[user], *params[(size, problem)])
             if has_params else None)

        ssf = None
        prev = None
        for a in chain:
            if prev is not None and a["is_repeat"]:
                gap_days = (parse_ts(a["created_at"])
                            - parse_ts(prev["created_at"])).total_seconds() / 86400.0
                if gap_days <= 0:
                    skipped["non-positive gap"] += 1
                elif ssf is None:
                    # Not recall-eligible (the chain had never failed), so this
                    # repeat did not come from the scheduler.
                    skipped["chain never failed"] += 1
                else:
                    events.append({
                        "user": user, "problem": problem, "size": size,
                        "y": 1 if a["success"] else 0,
                        "gap_days": gap_days,
                        "r": ssf,
                        "z": z,
                        "at": a["created_at"],
                        "month": a["created_at"][:7],
                    })
            # Advance the streak AFTER reading it for the gap above.
            if not a["success"]:
                ssf = 0
            elif a["is_repeat"]:
                ssf = 1 if ssf is None else ssf + 1
            # novel + correct leaves ssf as-is (undefined until a first failure)
            prev = a

    return events, skipped


# ---------------------------------------------------------------------------
# The retention model itself.
# ---------------------------------------------------------------------------
def predict_R(gap, z, r, kappa, growth, ln_c):
    lnS = np.clip(ln_c + kappa * (z - Z0) + growth * r, LN_S_TRAIN_MIN, LN_S_TRAIN_MAX)
    return np.clip(np.exp(-gap / np.exp(lnS)), 1e-9, 1 - 1e-9)


def fit_constants(gap, z, r, y, x0):
    """Refit (KAPPA, GROWTH, LN_C) by retention NLL — the same objective
    pipeline.fit_population_constants uses, repeated here so a held-out split
    can be scored without importing its I/O."""
    def nll(p):
        return float(-np.sum(y * np.log(predict_R(gap, z, r, *p))
                             + (1 - y) * np.log(1 - predict_R(gap, z, r, *p))))
    best = minimize(nll, x0=x0, method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 5000})
    return best.x


def fit_user_ln_c(gap, z, r, y, kappa, growth, prior):
    """Per-user base durability, L2-shrunk to the population prior exactly as
    the client shrinks it (lambda=10 mean-reduced over the training rows)."""
    def obj(p):
        R = predict_R(gap, z, r, kappa, growth, p[0])
        nll = -np.sum(y * np.log(R) + (1 - y) * np.log(1 - R))
        return float(nll + LN_C_LAMBDA * (p[0] - prior) ** 2)
    best = minimize(obj, x0=[prior], method="Nelder-Mead",
                    options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000})
    return float(best.x[0])


# ---------------------------------------------------------------------------
# Reports.
# ---------------------------------------------------------------------------
def report_by_rung(events, min_n, retention=None, max_rung=6, header=True):
    if not header:
        _print_rung_table(events, min_n, retention, max_rung)
        return
    print("\n" + "=" * 78)
    print("1. SERVED-RECALL CLEAR RATE BY STREAK RUNG")
    print("   Memory decays from the moment of answering: the window OPENS once")
    print("   %.0f%% is forgotten and CLOSES once %.0f%% is. So %.2f is the LEAST"
          % (100 * (1 - WINDOW_OPEN_R), 100 * (1 - WINDOW_CLOSE_R), WINDOW_OPEN_R))
    print("   forgetting the schedule ever serves at — a CEILING, tested one-sided.")
    print("   Where in that window a user returns is unobservable, but they are")
    print("   not waiting in the app, so serves spread across it and should")
    print("   average near its middle. BELOW the ceiling = consistent, however")
    print("   much a rung differs from its neighbour; AT or ABOVE = the model")
    print("   understates that rung's stability. Return promptness is a per-user")
    print("   habit, which is why check 2's within-user contrast is the sharp one.")
    print("   mean z is the cold-solve log-odds: if it is flat while the clear")
    print("   rate climbs, the ladder is selecting on something the model does")
    print("   NOT see (per-item memorability), not on problem difficulty.")
    print("   predicted R is what the model expected for those same serves;")
    print("   the sd figure beside it is that gap in units of the observed SE.")
    print("=" * 78)
    _print_rung_table(events, min_n, retention, max_rung)


def _print_rung_table(events, min_n, retention, max_rung):
    k = sum(e["y"] for e in events)
    print(rate_line("all repeats", k, len(events)))
    print()
    by_r = defaultdict(list)
    for e in events:
        by_r[min(e["r"], max_rung)].append(e)
    for rung in sorted(by_r):
        label = f"r = {rung}" + ("+" if rung == max_rung else "")
        evs = by_r[rung]
        if len(evs) < min_n:
            label += " (thin)"
        zs = [e["z"] for e in evs if e["z"] is not None]
        gaps = [e["gap_days"] for e in evs]
        scored = [e for e in evs if e["z"] is not None]
        pred = (np.mean(predict_R(np.array([e["gap_days"] for e in scored]),
                                  np.array([e["z"] for e in scored]),
                                  np.array([e["r"] for e in scored], float),
                                  *retention))
                if scored and retention else None)
        k_r, n_r = sum(e["y"] for e in evs), len(evs)
        rate = k_r / n_r
        se = math.sqrt(rate * (1 - rate) / n_r)          # binomial SE of the rate
        line = (rate_line(label, k_r, n_r)
                + f"   sd {se:.3f}"
                + f"   median gap {np.median(gaps):6.2f}d"
                + (f"   mean z {np.mean(zs):+.2f}" if zs else ""))
        if pred is not None:
            # Observed minus what the model expected for these same serves, in
            # units of the observed rate's own SE. This is the per-rung
            # calibration test; the pooled trend below cannot show which rung
            # carries the miss.
            line += f"   predicted R {pred:.3f} ({(rate - pred) / se:+5.1f} sd)"
        # The design does NOT pin the rate to a point: it serves anywhere in
        # [0.60, 0.80] and cannot choose where inside that window the user
        # returns. So the test that matters is whether the cell sits OUTSIDE
        # the band, not whether it differs from a neighbouring cell.
        # 0.80 is the window's OPENING, and retention only decays from there,
        # so it is a CEILING and the test against it is ONE-SIDED. A serve can
        # land at 0.80 only if no forgetting happened at all; anything at or
        # above it says the model understates that item's stability. Serves
        # spread through the window should average near its middle.
        if rate >= WINDOW_OPEN_R:
            line += f"   CEILING BREACH +{(rate - WINDOW_OPEN_R) / se:.1f} sd"
        else:
            line += f"   {(WINDOW_OPEN_R - rate) / se:5.1f} sd below ceiling"
        print(line)

    # A rate trending in r is the failure mode worth naming, so test it directly:
    # a weighted least-squares slope of the per-rung rate against the rung index.
    rungs = np.array([r for r in sorted(by_r) if len(by_r[r]) >= min_n], float)
    if len(rungs) >= 3:
        rates = np.array([np.mean([e["y"] for e in by_r[int(r)]]) for r in rungs])
        ns = np.array([len(by_r[int(r)]) for r in rungs], float)
        w = ns / np.maximum(rates * (1 - rates), 1e-6)     # inverse binomial variance
        A = np.vstack([np.ones_like(rungs), rungs]).T
        WA = A * w[:, None]
        cov = np.linalg.inv(A.T @ WA)
        beta = cov @ (WA.T @ rates)
        se = math.sqrt(cov[1, 1])
        print(f"\n  trend across rungs: {beta[1]:+.4f} clear-rate per rung "
              f"(SE {se:.4f}, {abs(beta[1] / se):.1f} sigma)")
        # The pooled trend leans on the thin deep rungs; the first step carries
        # most of the mass, so report its own two-proportion separation too.
        if 0 in by_r and 1 in by_r:
            y0 = np.array([e["y"] for e in by_r[0]], float)
            y1 = np.array([e["y"] for e in by_r[1]], float)
            s0 = math.sqrt(y0.mean() * (1 - y0.mean()) / len(y0))
            s1 = math.sqrt(y1.mean() * (1 - y1.mean()) / len(y1))
            print(f"  rung 0 vs rung 1: {y1.mean() - y0.mean():+.4f} "
                  f"({(y1.mean() - y0.mean()) / math.sqrt(s0 ** 2 + s1 ** 2):.1f} sd)")


def report_within_user(events, max_rung=6):
    """The rung comparison with user composition removed.

    Far more users reach rung 0 than reach rung 2, so the higher rungs are a
    SUBSET of users — the ones who kept coming back — and a raw between-rung
    comparison confounds the rung with whoever is standing on it. Two things
    here separate those: the rung-0 rate among users who do climb versus among
    those who do not (the size and sign of the composition effect), and a
    Mantel-Haenszel odds ratio stratified by user, which compares each user
    only against themselves and pools the strata. MH rather than a mean of
    per-user rates because most users contribute a handful of events per cell,
    where a rate is mostly noise but a 2x2 table is still information.
    """
    print("\n" + "=" * 78)
    print("2. THE RUNG EFFECT WITHIN USER (composition removed)")
    print("=" * 78)
    by_user = defaultdict(lambda: defaultdict(list))
    for e in events:
        by_user[e["user"]][min(e["r"], max_rung)].append(e["y"])

    users_at = defaultdict(set)
    for user, cells in by_user.items():
        for rung in cells:
            users_at[rung].add(user)
    print("  rung   users   events   (the subset shrinks as the ladder deepens)")
    for rung in sorted(users_at):
        n_ev = sum(len(by_user[u].get(rung, [])) for u in users_at[rung])
        print(f"  {rung:>4}   {len(users_at[rung]):>5}   {n_ev:>6}")

    climbers = {u for u, cells in by_user.items() if any(r >= 1 for r in cells)}
    a = [y for u in by_user if u in climbers for y in by_user[u].get(0, [])]
    b = [y for u in by_user if u not in climbers for y in by_user[u].get(0, [])]
    if a and b:
        print(f"\n  rung-0 clear rate, users who reach rung 1+: "
              f"{np.mean(a):6.1%}  n={len(a)}")
        print(f"  rung-0 clear rate, users who never do:      "
              f"{np.mean(b):6.1%}  n={len(b)}")
        print("  (the gap between these two IS the composition effect on rung 0)")

    def mantel_haenszel(hi, lo):
        """Stratified odds ratio, hi-rung vs lo-rung, one stratum per user."""
        num = den = 0.0
        rbg_p = rbg_q = rbg_r = rbg_s = 0.0
        strata = 0
        for user, cells in by_user.items():
            up = [y for r in cells if (r >= hi if hi == max_rung else r == hi)
                  for y in cells[r]]
            dn = [y for r in cells if (r >= lo if lo == max_rung else r == lo)
                  for y in cells[r]]
            if not up or not dn:
                continue
            A, B = float(sum(up)), float(len(up) - sum(up))
            C, D = float(sum(dn)), float(len(dn) - sum(dn))
            n = A + B + C + D
            num += A * D / n
            den += B * C / n
            P, Q = (A + D) / n, (B + C) / n
            rbg_p += P * (A * D) / n
            rbg_q += P * (B * C) / n + Q * (A * D) / n
            rbg_r += Q * (B * C) / n
            strata += 1
        if den <= 0 or num <= 0 or strata == 0:
            return None
        orr = num / den
        var = (rbg_p / (2 * num ** 2) + rbg_q / (2 * num * den)
               + rbg_r / (2 * den ** 2))
        return orr, math.sqrt(var), strata

    print()
    for hi, lo, label in ((1, 0, "rung 1 vs rung 0"), (2, 1, "rung 2 vs rung 1"),
                          (2, 0, "rung 2 vs rung 0")):
        res = mantel_haenszel(hi, lo)
        if res is None:
            print(f"  {label:<18} too few users with both cells")
            continue
        orr, se, strata = res
        lo_ci, hi_ci = math.exp(math.log(orr) - 1.96 * se), math.exp(math.log(orr) + 1.96 * se)
        # Restate the odds ratio as a rate at the lower rung's own base, which
        # is the form the band test can be applied to.
        base = np.mean([y for u in by_user for r in by_user[u]
                        if (r >= lo if lo == max_rung else r == lo)
                        for y in by_user[u][r]])
        odds = base / (1 - base) * orr
        print(f"  {label:<18} OR {orr:5.2f}  [{lo_ci:.2f}, {hi_ci:.2f}]  "
              f"{abs(math.log(orr) / se):4.1f} sd   {strata} users   "
              f"implies {odds / (1 + odds):.1%} from a {base:.1%} base")


def report_recent_windows(events, min_n, retention, windows=(30, 60)):
    """Does any of this hold on recent data alone?

    The scheduler is a moving target and the population turns over, so a result
    pooled over the whole logged span can be carried by a period that no longer
    resembles the present one. Re-runs the rung table on the trailing windows.
    """
    latest = max(e["at"] for e in events)
    latest_ts = parse_ts(latest).timestamp()
    for days in windows:
        cut = latest_ts - days * 86400
        sub = [e for e in events if parse_ts(e["at"]).timestamp() >= cut]
        print("\n" + "-" * 78)
        print(f"  TRAILING {days} DAYS ONLY  ({len(sub)} serves)")
        print("-" * 78)
        if len(sub) < min_n:
            print("  too few serves in this window")
            continue
        report_by_rung(sub, min_n, retention, header=False)


def report_stability_correction(events, x0, min_n, max_rung=4):
    """How much longer should each rung's interval be to land where rung 0 lands?

    This is a MOMENT MATCH, not a likelihood fit, and the distinction is the
    whole point. The retention likelihood is evaluated on intervals the
    scheduler itself chose, so it is nearly uninformative about what would
    happen if the schedule changed — which is exactly what a scheduling fix
    does. The scheduler's objective is not to predict the outcome of the serves
    it already makes; it is to place them so the realized clear rate lands where
    intended, which is a moment condition and can be solved directly.

    Under the model's own exponential form, scaling the interval by f scales
    log-retention by f, so the interval multiplier that moves a rung from its
    realized rate p to a target q is simply ln(q)/ln(p). Because the due time is
    proportional to stability, that same factor is the correction to stability
    at that rung, and the ratio between consecutive rungs is the step the growth
    term is missing at that transition.
    """
    print("\n" + "=" * 78)
    print("5. INTERVAL CORRECTION IMPLIED BY THE REALIZED RATES")
    print("=" * 78)
    kappa, growth, prior = x0
    by_r = defaultdict(list)
    for e in events:
        by_r[min(e["r"], max_rung)].append(e["y"])
    if 0 not in by_r or len(by_r[0]) < min_n:
        print("  not enough events at rung 0 — skipping")
        return
    rate0 = float(np.mean(by_r[0]))
    mid = math.sqrt(WINDOW_OPEN_R * WINDOW_CLOSE_R)   # geometric mid of the band
    print(f"  band midpoint {mid:.3f}; rung-0 realized rate {rate0:.3f}")
    print("  rung      n   realized   x interval for band mid   for rung-0 parity")
    prev = None
    for rung in sorted(by_r):
        ys = by_r[rung]
        if len(ys) < min_n:
            continue
        obs = float(np.mean(ys))
        f_mid = math.log(mid) / math.log(obs)
        f_par = math.log(rate0) / math.log(obs)
        line = (f"  {rung:>4}  {len(ys):>5}   {obs:8.3f}   {f_mid:>21.2f}   "
                f"{f_par:>17.2f}")
        if prev is not None:
            line += f"   step vs prev {f_par / prev:.2f}x"
        prev = f_par
        print(line)
    # How much of the first-step correction the model ALREADY sees. The ladder
    # selects on difficulty as well as on the streak, and the model does carry
    # difficulty through z — so before attributing the step to the streak term,
    # net out the part the difficulty term already accounts for.
    zs = {r: [e["z"] for e in evs if e["z"] is not None]
          for r, evs in ((r, [e for e in events if min(e["r"], max_rung) == r])
                         for r in (0, 1))}
    if zs.get(0) and zs.get(1):
        dz = float(np.mean(zs[1])) - float(np.mean(zs[0]))
        print(f"\n  mean cold-solve log-odds rises {dz:+.2f} from rung 0 to rung 1, which")
        print(f"  the difficulty term already turns into x{math.exp(kappa * dz):.2f} of stability —")
        print(f"  so model-visible selection covers only that much of the step above.")
    print("\n  Above 1 means the rung is served too early and its interval should be")
    print("  that much longer. The step between consecutive rungs is what the growth")
    print("  term is missing at that transition: a step of about 1 needs no change.")
    print("  NOTE the interaction with delivery: longer intervals mean fewer serves")
    print("  per entry inside any fixed horizon, and delivery is already the binding")
    print("  constraint (check 11). These two pull against each other.")


def report_within_rung_decay(events, min_n):
    """The retention FORM, with the confound removed.

    Pooled over rungs the clear rate RISES with the elapsed gap, which looks
    like memory improving with time. It is selection: a long gap is scheduled
    only for a high-stability item. Within a rung the scheduler's spread of
    intervals is much narrower and mostly reflects when the user came back, so
    this is the closest thing the log offers to a retention curve.
    """
    print("\n" + "=" * 78)
    print("4. RETENTION DECAY WITHIN A RUNG")
    print("   Pooled across rungs, elapsed gap is confounded with stability.")
    print("=" * 78)
    edges = [0.25, 0.5, 1, 2, 4, 8, 16, 10_000]
    for rung in (0, 1, 2):
        evs = [e for e in events if e["r"] == rung]
        if len(evs) < min_n:
            continue
        print(f"\n  r = {rung}   (n={len(evs)}, clear {np.mean([e['y'] for e in evs]):.1%})")
        for lo, hi in zip(edges[:-1], edges[1:]):
            sub = [e["y"] for e in evs if lo <= e["gap_days"] < hi]
            if len(sub) < min_n // 2:
                continue
            clo, chi = wilson(sum(sub), len(sub))
            span = f"{lo:g}-{hi:g}d" if hi < 1000 else f">{lo:g}d"
            print(f"    {span:<10} n={len(sub):>5}   {np.mean(sub):6.1%}  "
                  f"[{clo:5.1%}, {chi:5.1%}]")


def report_by_month(events, min_n, max_rung=3):
    print("\n" + "=" * 78)
    print("3. THE SAME, BY CALENDAR MONTH")
    print("   Client scheduling changed over the logged era; a pooled rate")
    print("   would hide a regime change.")
    print("=" * 78)
    by_month = defaultdict(list)
    for e in events:
        by_month[e["month"]].append(e)
    header = "  month    n      clear      " + "".join(
        f"  r={r}{'+' if r == max_rung else ' '}      " for r in range(max_rung + 1))
    print(header)
    for month in sorted(by_month):
        evs = by_month[month]
        k, n = sum(e["y"] for e in evs), len(evs)
        cells = []
        for rung in range(max_rung + 1):
            sub = [e["y"] for e in evs
                   if (e["r"] >= rung if rung == max_rung else e["r"] == rung)]
            cells.append(f"{np.mean(sub):6.1%}({len(sub):>4})" if len(sub) >= min_n
                         else f"     —({len(sub):>4})")
        print(f"  {month}  {n:>5}  {k / n:6.1%}    " + " ".join(cells))


def report_calibration(events, kappa, growth, ln_c_prior, per_user_ln_c, label):
    scored = [e for e in events if e["z"] is not None]
    if not scored:
        print("\n  (no events with resolvable problem parameters — skipping)")
        return
    y = np.array([e["y"] for e in scored], float)
    gap = np.array([e["gap_days"] for e in scored], float)
    z = np.array([e["z"] for e in scored], float)
    r = np.array([e["r"] for e in scored], float)

    if per_user_ln_c:
        ln_c = np.empty(len(scored))
        by_user = defaultdict(list)
        for i, e in enumerate(scored):
            by_user[e["user"]].append(i)
        for _, idx in by_user.items():
            idx = np.array(idx)
            v = fit_user_ln_c(gap[idx], z[idx], r[idx], y[idx],
                              kappa, growth, ln_c_prior)
            ln_c[idx] = v
    else:
        ln_c = np.full(len(scored), ln_c_prior)

    R = predict_R(gap, z, r, kappa, growth, ln_c)
    base = float(np.mean(y))
    print(f"\n  {label}: n={len(scored)}  mean predicted R={np.mean(R):.3f}  "
          f"observed={base:.3f}")
    print(f"    log-loss  model {logloss(y, R):.4f} | "
          f"constant {logloss(y, np.full_like(y, base)):.4f} | "
          f"0.80 target {logloss(y, np.full_like(y, X_SERVE)):.4f}")
    print(f"    Brier     model {brier(y, R):.4f} | "
          f"constant {brier(y, np.full_like(y, base)):.4f} | "
          f"0.80 target {brier(y, np.full_like(y, X_SERVE)):.4f}")
    edges = [0.0, 0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]
    print("    predicted-R bin    n     predicted   observed")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (R >= lo) & (R < hi)
        if m.sum() == 0:
            continue
        clo, chi = wilson(int(y[m].sum()), int(m.sum()))
        print(f"    [{lo:.2f},{hi:.2f})      {int(m.sum()):>5}   {np.mean(R[m]):8.3f}   "
              f"{np.mean(y[m]):8.3f}  [{clo:.3f}, {chi:.3f}]")
    return R


def report_window(R):
    print("\n" + "=" * 78)
    print("7. WHERE SERVES LAND IN THE WINDOW")
    print(f"   The scheduler only offers a repeat while its predicted retention")
    print(f"   sits in [{WINDOW_CLOSE_R:.2f}, {WINDOW_OPEN_R:.2f}]. Reconstructed here from the")
    print("   POPULATION model, so spread outside the window is partly the")
    print("   reconstruction and partly real (gate deferrals, notification clicks).")
    print("=" * 78)
    inside = float(np.mean((R >= WINDOW_CLOSE_R) & (R <= WINDOW_OPEN_R)))
    print(f"  inside the window: {inside:.1%}")
    print(f"  above 0.80 (early): {float(np.mean(R > WINDOW_OPEN_R)):.1%}   "
          f"below 0.60 (late): {float(np.mean(R < WINDOW_CLOSE_R)):.1%}")
    qs = np.percentile(R, [5, 25, 50, 75, 95])
    print("  predicted R percentiles  p5=%.3f p25=%.3f p50=%.3f p75=%.3f p95=%.3f" % tuple(qs))


def report_heldout(events, x0, seed=0):
    print("\n" + "=" * 78)
    print("8. OUT-OF-SAMPLE VALUE (split by USER, so no user is in both halves)")
    print("   'recalibrated' rescales the model's log-odds by a single slope and")
    print("   intercept fit on the training half — it keeps the ORDERING the model")
    print("   produces and throws away its scale, which separates 'no signal' from")
    print("   'right signal, wrong spread'.")
    print("=" * 78)
    scored = [e for e in events if e["z"] is not None]
    users = sorted({e["user"] for e in scored})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(users))
    train_users = {users[i] for i in perm[: len(users) // 2]}

    def arrays(subset):
        return (np.array([e["gap_days"] for e in subset], float),
                np.array([e["z"] for e in subset], float),
                np.array([e["r"] for e in subset], float),
                np.array([e["y"] for e in subset], float))

    tr = [e for e in scored if e["user"] in train_users]
    te = [e for e in scored if e["user"] not in train_users]
    if len(tr) < 200 or len(te) < 200:
        print("  too few events to split meaningfully — skipping")
        return
    g_tr, z_tr, r_tr, y_tr = arrays(tr)
    g_te, z_te, r_te, y_te = arrays(te)
    kappa, growth, ln_c = fit_constants(g_tr, z_tr, r_tr, y_tr, x0)
    print(f"  refit on {len(tr)} events / {len(train_users)} users: "
          f"KAPPA={kappa:.3f} GROWTH={growth:.3f} LN_C={ln_c:.3f}")
    print(f"  shipped:                            "
          f"KAPPA={x0[0]:.3f} GROWTH={x0[1]:.3f} LN_C={x0[2]:.3f}")
    R_refit = predict_R(g_te, z_te, r_te, kappa, growth, ln_c)
    R_ship = predict_R(g_te, z_te, r_te, *x0)
    base = float(np.mean(y_tr))          # constant baseline from TRAIN, honestly

    # Platt recalibration of the SHIPPED model, fit on the training half only.
    def platt(p_tr, y_tr_, p_te):
        lo_tr = np.log(np.clip(p_tr, 1e-6, 1 - 1e-6) / (1 - np.clip(p_tr, 1e-6, 1 - 1e-6)))
        lo_te = np.log(np.clip(p_te, 1e-6, 1 - 1e-6) / (1 - np.clip(p_te, 1e-6, 1 - 1e-6)))

        def nll(ab):
            q = 1 / (1 + np.exp(-(ab[0] + ab[1] * lo_tr)))
            q = np.clip(q, 1e-9, 1 - 1e-9)
            return float(-np.sum(y_tr_ * np.log(q) + (1 - y_tr_) * np.log(1 - q)))
        ab = minimize(nll, x0=[0.0, 1.0], method="Nelder-Mead",
                      options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 4000}).x
        return 1 / (1 + np.exp(-(ab[0] + ab[1] * lo_te))), ab

    R_ship_tr = predict_R(g_tr, z_tr, r_tr, *x0)
    R_cal, ab = platt(R_ship_tr, y_tr, R_ship)

    # Two ablations that say WHERE any signal lives.
    #  - z fixed at the training mean: strips the reconstructed cold-solve
    #    log-odds (the one input this script approximates rather than reads),
    #    leaving the interval and the streak.
    #  - rung-only: the training half's empirical clear rate at each rung, with
    #    no retention arithmetic at all. This is the bar the model has to clear
    #    to be earning its structure.
    R_noz = predict_R(g_te, np.full_like(z_te, float(np.mean(z_tr))), r_te, *x0)
    max_rung = 6
    rung_rate, pooled = {}, float(np.mean(y_tr))
    for rung in range(max_rung + 1):
        m = (np.minimum(r_tr, max_rung) == rung)
        rung_rate[rung] = float(np.mean(y_tr[m])) if m.sum() >= 30 else pooled
    R_rung = np.array([rung_rate[int(min(v, max_rung))] for v in r_te])

    print(f"\n  held-out log-loss ({len(te)} events / {len(users) - len(train_users)} users)")
    print(f"    refit constants   {logloss(y_te, R_refit):.4f}")
    print(f"    shipped constants {logloss(y_te, R_ship):.4f}")
    print(f"    shipped, z fixed  {logloss(y_te, R_noz):.4f}")
    print(f"    recalibrated      {logloss(y_te, R_cal):.4f}   "
          f"(intercept {ab[0]:+.3f}, slope {ab[1]:.3f} on the model's log-odds)")
    print(f"    rung-only rate    {logloss(y_te, R_rung):.4f}   "
          f"(no retention arithmetic: the train half's rate at each rung)")
    print(f"    constant rate     {logloss(y_te, np.full_like(y_te, base)):.4f}")
    print(f"    0.80 target       {logloss(y_te, np.full_like(y_te, X_SERVE)):.4f}")
    print("    (a recalibration slope well below 1 means the ordering is real and"
          " the spread is too wide)")

    # The interval is what the model is FOR, so also report how much of the
    # gap-length signal it captures: does R beat a model that ignores the gap?
    print("\n  observed clear rate by elapsed gap, POOLED over rungs (confounded —"
          " see check 3)")
    edges = [0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 10_000]
    g_all, _, _, y_all = arrays(scored)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (g_all >= lo) & (g_all < hi)
        if m.sum() < 30:
            continue
        clo, chi = wilson(int(y_all[m].sum()), int(m.sum()))
        span = f"{lo:g}-{hi:g}d" if hi < 1000 else f">{lo:g}d"
        print(f"    {span:<10} n={int(m.sum()):>5}   "
              f"{np.mean(y_all[m]):6.1%}  [{clo:5.1%}, {chi:5.1%}]")


def report_growth_forms(events, x0, seed=0):
    """Can any reshaping of the streak term rescue the model?

    The observed rates step up at the first success and then plateau, which the
    shipped `growth*r` cannot express. The obvious repair is to add a
    first-success step. The upper bound on ANY such repair is a free effect per
    rung, which can reproduce whatever shape the rungs have. Fitting both on
    half the users and scoring the other half bounds what reshaping is worth
    before anyone spends effort on the shape itself.
    """
    print("\n" + "=" * 78)
    print("9. IS THE STREAK TERM'S SHAPE THE PROBLEM?")
    print("=" * 78)
    scored = [e for e in events if e["z"] is not None]
    users = sorted({e["user"] for e in scored})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(users))
    train_users = {users[i] for i in perm[: len(users) // 2]}

    def arrays(subset):
        return (np.array([e["gap_days"] for e in subset], float),
                np.array([e["z"] for e in subset], float),
                np.array([e["r"] for e in subset], float),
                np.array([e["y"] for e in subset], float))

    tr = [e for e in scored if e["user"] in train_users]
    te = [e for e in scored if e["user"] not in train_users]
    if len(tr) < 200 or len(te) < 200:
        print("  too few events to split meaningfully — skipping")
        return
    g_tr, z_tr, r_tr, y_tr = arrays(tr)
    g_te, z_te, r_te, y_te = arrays(te)

    def ln_s(form, p, z, r):
        if form == "linear":                    # the shipped form
            return p[2] + p[0] * (z - Z0) + p[1] * r
        if form == "step":                      # + a first-success step
            return p[2] + p[0] * (z - Z0) + p[1] * r + p[3] * (r >= 1)
        return (p[2] + p[0] * (z - Z0)          # free per-rung: the upper bound
                + p[3] * (r == 1) + p[4] * (r == 2) + p[5] * (r >= 3))

    n_extra = {"linear": 0, "step": 1, "free": 3}
    for form in ("linear", "step", "free"):
        def nll(p, form=form):
            R = np.clip(np.exp(-g_tr / np.exp(np.clip(ln_s(form, p, z_tr, r_tr),
                                                      LN_S_TRAIN_MIN, LN_S_TRAIN_MAX))),
                        1e-9, 1 - 1e-9)
            return float(-np.sum(y_tr * np.log(R) + (1 - y_tr) * np.log(1 - R)))
        best = minimize(nll, x0=list(x0) + [0.0] * n_extra[form],
                        method="Nelder-Mead",
                        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 20000})
        R_te = np.clip(np.exp(-g_te / np.exp(np.clip(ln_s(form, best.x, z_te, r_te),
                                                     LN_S_TRAIN_MIN, LN_S_TRAIN_MAX))),
                       1e-9, 1 - 1e-9)
        print(f"  {form:<7} held-out LL {logloss(y_te, R_te):.4f}")
    base = float(np.mean(y_tr))
    print(f"  {'constant':<7} held-out LL {logloss(y_te, np.full_like(y_te, base)):.4f}")
    print("  A free per-rung effect is the best any reshaping of the streak term")
    print("  could ever do. If it does not close the gap to the constant, the")
    print("  shape of that term is not where the loss lives.")


def report_forward_per_user(events, x0, min_events=20, max_rung=6):
    """The fairest reproduction of what the client actually does.

    LN_C is a PER-USER parameter the device fits on that user's own past, so
    scoring the model with the population LN_C understates it. Here each user's
    events are split chronologically: LN_C is fit on the earlier half (shrunk to
    the population prior exactly as the client shrinks it) and the later half is
    predicted. Every baseline is built from the same earlier halves, so nothing
    sees its own evaluation rows.
    """
    print("\n" + "=" * 78)
    print("10. FORWARD TEST WITH A PER-USER LN_C (the shipped configuration)")
    print("   Each user: fit LN_C on their earlier events, predict their later ones.")
    print("=" * 78)
    kappa, growth, prior = x0
    by_user = defaultdict(list)
    for e in events:
        if e["z"] is not None:
            by_user[e["user"]].append(e)

    past, fut, fut_lnc, fut_user_base = [], [], [], []
    for user, evs in by_user.items():
        if len(evs) < min_events:
            continue
        evs = sorted(evs, key=lambda e: e["at"])
        cut = len(evs) // 2
        a, b = evs[:cut], evs[cut:]
        ln_c = fit_user_ln_c(np.array([e["gap_days"] for e in a]),
                             np.array([e["z"] for e in a]),
                             np.array([e["r"] for e in a], float),
                             np.array([e["y"] for e in a], float),
                             kappa, growth, prior)
        base_u = float(np.mean([e["y"] for e in a]))
        past.extend(a)
        fut.extend(b)
        fut_lnc.extend([ln_c] * len(b))
        fut_user_base.extend([base_u] * len(b))
    if len(fut) < 200:
        print(f"  only {len(fut)} forward events — skipping")
        return
    y = np.array([e["y"] for e in fut], float)
    gap = np.array([e["gap_days"] for e in fut], float)
    z = np.array([e["z"] for e in fut], float)
    r = np.array([e["r"] for e in fut], float)
    y_past = np.array([e["y"] for e in past], float)
    r_past = np.array([e["r"] for e in past], float)

    R_user = predict_R(gap, z, r, kappa, growth, np.array(fut_lnc))
    R_pop = predict_R(gap, z, r, kappa, growth, prior)
    pooled = float(np.mean(y_past))
    rung_rate = {}
    for rung in range(max_rung + 1):
        m = (np.minimum(r_past, max_rung) == rung)
        rung_rate[rung] = float(np.mean(y_past[m])) if m.sum() >= 30 else pooled
    R_rung = np.array([rung_rate[int(min(v, max_rung))] for v in r])

    print(f"  {len(by_user)} users, {len(fut)} forward events "
          f"(users with >= {min_events} events)")
    print(f"  fitted LN_C spread: p10={np.percentile(fut_lnc, 10):.2f} "
          f"p50={np.percentile(fut_lnc, 50):.2f} p90={np.percentile(fut_lnc, 90):.2f} "
          f"(prior {prior:.2f})")
    print(f"\n  forward log-loss")
    print(f"    per-user LN_C     {logloss(y, R_user):.4f}")
    print(f"    population LN_C   {logloss(y, R_pop):.4f}")
    print(f"    rung-only rate    {logloss(y, R_rung):.4f}")
    print(f"    user's own rate   {logloss(y, np.array(fut_user_base)):.4f}")
    print(f"    constant rate     {logloss(y, np.full_like(y, pooled)):.4f}")
    print(f"    0.80 target       {logloss(y, np.full_like(y, X_SERVE)):.4f}")


def report_equilibrium(events, rows, min_n, x0, max_rung=6):
    """What recallServeShare() believes about the recall lifecycle, priced out
    on the users it is actually run for.

    The share it returns is rho/(1+rho) with rho = e*E, where e is ladder
    entries per novel serve and **E is the expected number of recall serves one
    ladder entry consumes before graduating** — the same unit as check 11's
    measured count, so the two are directly comparable. E follows from a reset
    chain: climb one rung on a success, fall to rung 0 on a failure, graduate
    at the rung k where the interval outgrows the working window,

        N_r = 1 + x*N_{r+1} + (1 - x)*N_0,   N_k = 0   ==>   E = (x^-k - 1)/(1 - x)

    with x the per-serve success probability, hardcoded at X_SERVE. Both inputs
    are computed here from the log rather than swept over a made-up range: k
    from each user's own volume and base delay exactly as the client computes
    it, and x from that user's own observed rates.
    """
    kappa, growth, prior = x0
    print("\n" + "=" * 78)
    print("12. WHAT THE EQUILIBRIUM BELIEVES, PRICED ON REAL USERS")
    print("    k = consecutive successes an entry needs to GRADUATE: each success")
    print("        multiplies stability by e^GROWTH, so the interval doubles per")
    print("        rung, and the entry leaves the pool once that interval exceeds")
    print("        the calendar time the working window spans at the user's volume.")
    print("    E = recall SERVES that takes, which is much larger than k because a")
    print("        failure at any rung drops the entry back to the bottom.")
    print("=" * 78)
    by_r = defaultdict(list)
    for e in events:
        by_r[min(e["r"], max_rung)].append(e["y"])
    rates = {r: float(np.mean(ys)) for r, ys in by_r.items() if len(ys) >= min_n}
    if 0 not in rates:
        print("  not enough events at rung 0 — skipping")
        return

    def expected_serves(k, x_of_rung):
        """N_0 by backward substitution, writing N_r = a + b*N_0."""
        if k <= 0:
            return 0.0
        a, b = 0.0, 0.0
        for rung in range(int(math.ceil(k)) - 1, -1, -1):
            x = x_of_rung(rung)
            a, b = 1 + x * a, x * b + (1 - x)
        return a / (1 - b) if b < 1 else float("inf")

    # Per user: A = half the trailing-week daily volume (SHARE_VOLUME_FRACTION),
    # tau0 = the base window-open delay from their own fitted durability.
    by_user_rows = defaultdict(list)
    for r in rows:
        by_user_rows[r["user_id"]].append(parse_ts(r["created_at"]).timestamp())
    by_user_ev = defaultdict(list)
    for e in events:
        if e["z"] is not None:
            by_user_ev[e["user"]].append(e)

    ks, es, serves = [], [], []
    for user, evs in by_user_ev.items():
        if len(evs) < 20:
            continue
        ts = sorted(by_user_rows[user])
        week_count = sum(1 for t in ts if t >= ts[-1] - SHARE_WEEK_MS_S)
        A = (week_count / 7.0) * SHARE_VOLUME_FRACTION
        if A <= 0:
            continue
        ln_c = fit_user_ln_c(np.array([e["gap_days"] for e in evs]),
                             np.array([e["z"] for e in evs]),
                             np.array([e["r"] for e in evs], float),
                             np.array([e["y"] for e in evs], float),
                             kappa, growth, prior)
        z0_med = float(np.median([e["z"] for e in evs]))
        S0 = math.exp(min(max(ln_c + kappa * (z0_med - Z0), LN_S_TRAIN_MIN),
                          LN_S_TRAIN_MAX))
        tau0 = max(-math.log(WINDOW_OPEN_R) * S0, MIN_RECALL_HOURS / 24)
        k = math.log(WORKING_WINDOW / (A * tau0)) / growth
        if not math.isfinite(k):
            continue
        ks.append(k)
        es.append(expected_serves(k, lambda _r: X_SERVE))
        serves.append(len(evs))

    if not ks:
        print("  no users with enough events to price")
        return
    ks, es = np.array(ks), np.array(es)
    print(f"  {len(ks)} users priced (>= 20 scheduled serves each)")
    print(f"  k (successes to graduate):  p10={np.percentile(ks, 10):.1f} "
          f"p50={np.percentile(ks, 50):.1f} p90={np.percentile(ks, 90):.1f}")
    print(f"  E (serves to get there):    p10={np.percentile(es, 10):.1f} "
          f"p50={np.percentile(es, 50):.1f} p90={np.percentile(es, 90):.1f}")
    e_obs = expected_serves(float(np.median(ks)),
                            lambda r: rates.get(min(r, max(rates)), rates[max(rates)]))
    print(f"  at the median k, substituting the OBSERVED per-rung rates "
          f"({'  '.join(f'r{r}={rates[r]:.2f}' for r in sorted(rates))})")
    print(f"    gives E = {e_obs:.1f} against {np.median(es):.1f} at the flat "
          f"{X_SERVE:.2f} — a factor of {e_obs / max(np.median(es), 1e-9):.2f}")
    print("  Check 11 measures the same quantity directly. The gap between that")
    print("  count and this E is the size of the lifecycle error, in serves.")


def expected_serves_in_horizon(S0, growth, horizon_days, x_of_rung,
                               dt=0.25, r_max=15):
    """Expected recall serves a fresh ladder entry receives WITHIN horizon_days,
    assuming the user returns exactly when each window opens.

    The interval at rung r is -ln(0.80)*S0*exp(growth*r), a success advances a
    rung and a failure returns to rung 0, so

        N(t, r) = 0                                          if t < interval(r)
                = 1 + x*N(t - interval(r), r+1)
                    + (1 - x)*N(t - interval(r), 0)          otherwise

    solved on a time grid. This is the quantity a bounded observation window can
    actually be compared against: an expectation taken all the way to graduation
    cannot be, because the calendar time required to climb that far exceeds the
    log itself.
    """
    steps = int(round(horizon_days / dt)) + 1
    iv = [max(-math.log(WINDOW_OPEN_R) * S0 * math.exp(growth * r),
              MIN_RECALL_HOURS / 24) for r in range(r_max + 1)]
    N = np.zeros((steps, r_max + 1))
    for ti in range(steps):
        t = ti * dt
        for r in range(r_max, -1, -1):
            if t < iv[r] or r == r_max:
                continue
            prev = int(round((t - iv[r]) / dt))
            x = x_of_rung(r)
            N[ti, r] = 1 + x * N[prev, r + 1] + (1 - x) * N[prev, 0]
    return float(N[steps - 1, 0])


def report_per_user_step(rows, events, thresholds=(5, 10, 20)):
    """Would a PER-USER post-lapse step recover anything over a global one?

    The correction in check 5 could ship as one global constant or as a second
    on-device parameter beside the base durability. That is a variance-component
    question, not a design preference: estimate each user's own rung-0 vs
    rung-1+ log odds ratio, subtract the sampling variance from the observed
    spread, and what is left is the genuine between-user variance. Shrinkage at
    the median user's precision says how far a per-user estimate would actually
    move from the global value.

    Also reports the density that would have to support it — rung-1+ events
    inside the client's own training window — since a parameter with no rows to
    estimate it from is the global value wearing extra machinery.
    """
    print("\n" + "=" * 78)
    print("12. HOW CONSISTENT IS THE RUNG-0 -> RUNG-1 STEP ACROSS USERS?")
    print("=" * 78)
    cells = defaultdict(lambda: {0: [0, 0], 1: [0, 0]})
    for e in events:
        c = cells[e["user"]][0 if e["r"] == 0 else 1]
        c[0] += e["y"]
        c[1] += 1
    for min_cell in thresholds:
        lo, se2 = [], []
        for _, c in cells.items():
            k0, n0 = c[0]
            k1, n1 = c[1]
            if n0 < min_cell or n1 < min_cell:
                continue
            # Haldane-Anscombe: the half-counts keep empty cells finite.
            a, b, cc, d = k1 + .5, n1 - k1 + .5, k0 + .5, n0 - k0 + .5
            lo.append(math.log(a / b) - math.log(cc / d))
            se2.append(1 / a + 1 / b + 1 / cc + 1 / d)
        if len(lo) < 10:
            print(f"  >= {min_cell:>2} per cell: only {len(lo)} users — skipping")
            continue
        lo, se2 = np.array(lo), np.array(se2)
        observed = float(np.var(lo, ddof=1))
        sampling = float(np.mean(se2))
        tau2 = max(0.0, observed - sampling)
        move = tau2 / (tau2 + float(np.median(se2)))
        # Cochran's Q and I^2: the standard way to ask how much of the spread
        # across strata is real rather than sampling noise.
        w = 1.0 / se2
        pooled = float(np.sum(w * lo) / np.sum(w))
        Q = float(np.sum(w * (lo - pooled) ** 2))
        df = len(lo) - 1
        I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
        qs = np.exp(np.percentile(lo, [10, 25, 50, 75, 90]))
        print(f"\n  >= {min_cell:>2} events per cell  ({len(lo)} users)")
        print(f"     pooled OR {math.exp(pooled):.2f}   "
              f"per-user OR p10={qs[0]:.2f} p25={qs[1]:.2f} p50={qs[2]:.2f} "
              f"p75={qs[3]:.2f} p90={qs[4]:.2f}")
        print(f"     above 1 (effect in the expected direction): "
              f"{float(np.mean(lo > 0)):.0%} of users")
        print(f"     Q={Q:.0f} on {df} df, I^2={I2:.0%}   "
              f"(share of spread that is real, not sampling noise)")
        print(f"     observed var {observed:.3f} - sampling {sampling:.3f} = "
              f"tau^2 {tau2:.3f}; a per-user estimate would move {move:.0%}")
        if tau2 > 0:
            print(f"     implied real spread: middle 80% of users span "
                  f"OR {math.exp(pooled - 1.28 * math.sqrt(tau2)):.2f} to "
                  f"{math.exp(pooled + 1.28 * math.sqrt(tau2)):.2f}")
    print("  (tau^2 swinging with the threshold means the component is not")
    print("   resolved: at the loosest cut the sampling variance already exceeds")
    print("   the observed spread, which is a degenerate answer, and the tighter")
    print("   cuts keep only heavy users.)")

    # Density in the window the client actually trains on.
    by_user_rows = defaultdict(list)
    for r in rows:
        by_user_rows[r["user_id"]].append(r["created_at"])
    horizon = {}
    for u, ts in by_user_rows.items():
        ts.sort()
        horizon[u] = ts[-MODEL_INPUT_HISTORY_LENGTH] if len(ts) > MODEL_INPUT_HISTORY_LENGTH else ""
    per_user = defaultdict(int)
    for e in events:
        if e["r"] >= 1 and e["at"] >= horizon.get(e["user"], ""):
            per_user[e["user"]] += 1
    users = sorted({e["user"] for e in events})
    v = np.array([per_user.get(u, 0) for u in users])
    print(f"\n  rung-1+ events inside the client's {MODEL_INPUT_HISTORY_LENGTH}-attempt "
          f"training window, per user:")
    print("   " + "  ".join(f"p{q}={np.percentile(v, q):.0f}"
                            for q in (10, 25, 50, 75, 90)))
    print(f"   {float(np.mean(v == 0)):.0%} of users have none at all; "
          f"{float(np.mean(v >= 20)):.0%} have twenty or more.")


def report_serves_per_entry(rows, events, params, abilities, x0, min_n,
                            all_chains=False, horizons=(30, 60)):
    """Serves per ladder entry, with the things that are NOT scheduler failures
    deducted one at a time.

    A raw count over every entry in the log is close to meaningless, for four
    reasons that have nothing to do with the scheduler:

      * the observation span truncates every lifecycle, so the comparison must
        be made inside a fixed horizon and against an expectation taken over
        that same horizon (intervals double per rung, and climbing to the
        graduation rung takes more calendar time than the log contains);
      * most entries belong to users who stopped playing, and an entry cannot
        be re-served to someone who never came back;
      * an entry on a problem at or above the user's level is **never eligible**
        — the candidate filter is deliberate, and failures concentrate on
        exactly those problems, so they dominate a raw count;
      * a user can pause recalls entirely, which is invisible in the attempt log
        and cannot be deducted here at all.

    What survives all of that is compared against a schedule walk: step through
    the model's own windows, and separately count the windows the user actually
    had a session inside. The difference between those two is what presence
    costs; the difference between the second and the realized count is the gate,
    the one-recall-per-decision limit, per-device history and pausing together —
    and the gate is *meant* to drop some, so this is not all shortfall.
    """
    kappa, growth, prior = x0
    print("\n" + "=" * 78)
    print("11. SERVES PER LADDER ENTRY, WITH THE CONFOUNDS DEDUCTED")
    print("=" * 78)
    latest_ts = parse_ts(max(r["created_at"] for r in rows)).timestamp()
    by_chain = defaultdict(list)
    for r in rows:
        by_chain[(r["user_id"], r["problem"], r["size"])].append(r)
    user_times = defaultdict(list)
    for r in rows:
        user_times[r["user_id"]].append(parse_ts(r["created_at"]).timestamp())
    for u in user_times:
        user_times[u].sort()
    D = 86400.0

    for T in horizons:
        cutoff = latest_ts - T * 86400
        raw, active, eligible = [], [], []
        opens, present = [], []
        for (user, problem, size), chain in by_chain.items():
            chain.sort(key=lambda r: r["created_at"])
            if not all_chains and chain[0]["created_at"][:10] < REPEAT_LOG_CUTOFF:
                continue
            fi = next((i for i, a in enumerate(chain) if not a["success"]), None)
            if fi is None:
                continue
            t0 = parse_ts(chain[fi]["created_at"]).timestamp()
            if t0 > cutoff:
                continue
            n = sum(1 for a in chain[fi + 1:] if a["is_repeat"]
                    and parse_ts(a["created_at"]).timestamp() <= t0 + T * 86400)
            raw.append(n)
            if user not in abilities or (size, problem) not in params:
                continue
            inh = [t for t in user_times[user] if t0 < t <= t0 + T * 86400]
            if len(inh) < 20 or not any(t > t0 + 0.75 * T * 86400 for t in inh):
                continue                       # user did not stay for the horizon
            active.append(n)
            if params[(size, problem)][0] >= abilities[user]:
                continue                       # never a recall candidate
            eligible.append(n)
            # Walk the model's own schedule, counting windows and attendance.
            z = cold_solve_logit(abilities[user], *params[(size, problem)])
            S = math.exp(min(max(prior + kappa * (z - Z0), LN_S_TRAIN_MIN),
                             LN_S_TRAIN_MAX))
            t, r, n_open, n_present = t0, 0, 0, 0
            while n_open < 40:
                Sr = max(S * math.exp(growth * r),
                         (MIN_RECALL_HOURS / 24) / -math.log(WINDOW_OPEN_R))
                o = t + (-math.log(WINDOW_OPEN_R)) * Sr * D
                c = t + (-math.log(WINDOW_CLOSE_R)) * Sr * D
                if o > t0 + T * 86400:
                    break
                n_open += 1
                hit = [x for x in inh if o <= x <= c]
                if hit:
                    n_present += 1
                    t, r = hit[0], r + 1       # served on arrival, cleared
                else:
                    t, r = c, 0                # aged out; the streak resets
            opens.append(n_open)
            present.append(n_present)

        if not eligible:
            print(f"  {T:>3}d horizon: no entries survive the filters")
            continue
        e = np.array(eligible)
        print(f"\n  {T}-day horizon")
        print(f"    every entry in the log                       "
              f"{np.mean(raw):5.2f}   n={len(raw)}")
        print(f"    ...user still playing at the end of it       "
              f"{np.mean(active):5.2f}   n={len(active)}")
        print(f"    ...and the problem below the user's level    "
              f"{e.mean():5.2f}   n={len(e)}   <- the schedulable population")
        print(f"    windows the schedule opens for those         "
              f"{np.mean(opens):5.2f}")
        print(f"    windows the user was actually present for    "
              f"{np.mean(present):5.2f}   "
              f"(presence costs {1 - np.mean(present) / max(np.mean(opens), 1e-9):.0%})")
        print(f"    of those opportunities, taken                "
              f"{e.mean() / max(np.mean(present), 1e-9):.0%}   "
              f"(gate + one-per-decision + per-device + pausing)")
        print(f"    never served {float(np.mean(e == 0)):.0%}, "
              f"never even present in a window "
              f"{float(np.mean(np.array(present) == 0)):.0%}")
    print("\n  The gate is DESIGNED to drop some of those opportunities, so the")
    print("  last line is not all shortfall. Pausing is invisible here and is not")
    print("  deducted. Read the schedulable row, not the first one.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.join(PROJECT_ROOT, "attempts", "database.sqlite"))
    ap.add_argument("--ratings-dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--constants", default=os.path.join(PROJECT_ROOT, "ratings",
                                                        "population_constants.json"))
    ap.add_argument("--all-chains", action="store_true",
                    help="keep chains that start before REPEAT_LOG_CUTOFF (their gaps "
                         "may hide unlogged reviews)")
    ap.add_argument("--min-n", type=int, default=100,
                    help="suppress a rate cell below this many events")
    ap.add_argument("--keep-short", action="store_true",
                    help=f"keep pairs closer than MIN_RECALL_HOURS={MIN_RECALL_HOURS:g}h; "
                         "the stability floor makes those unschedulable, so by default "
                         "they are reported separately and excluded")
    ap.add_argument("--per-user-ln-c", action="store_true",
                    help="also score calibration with a per-user LN_C, shrunk to the "
                         "population prior the way the client shrinks it")
    args = ap.parse_args()

    import json
    with open(args.constants) as f:
        constants = json.load(f)["retention"]
    kappa, growth, ln_c_prior = (constants["kappa"], constants["growth"],
                                 constants["ln_c_prior"])
    print(f"Shipped retention constants ({args.constants}):")
    print(f"  KAPPA={kappa}  GROWTH={growth} (stability x{math.exp(growth):.2f} per success)"
          f"  Z0={constants['z0']}  LN_C_PRIOR={ln_c_prior} "
          f"(base stability {math.exp(ln_c_prior):.2f}d at p=0.7)")

    print(f"\nLoading attempts from {args.db} ...")
    rows = load_rows(args.db)
    print(f"  {len(rows)} rows")
    needed = {(r["size"], r["problem"]) for r in rows}
    params = load_problem_params(args.ratings_dir, needed)
    print(f"  resolved 3PL params for {len(params)}/{len(needed)} problems")

    by_user = defaultdict(list)
    for r in rows:
        by_user[r["user_id"]].append(r)
    abilities = {}
    for user, urows in by_user.items():
        ev = [(params[(r["size"], r["problem"])], r["success"]) for r in urows
              if not r["is_repeat"] and (r["size"], r["problem"]) in params]
        if len(ev) < 10:
            continue
        abilities[user] = fit_ability(
            np.array([e[0][0] for e in ev]),
            np.array([e[0][1] for e in ev]),
            np.clip(np.array([e[0][2] for e in ev]), 0, FLOOR_MAX),
            np.array([1 if e[1] else 0 for e in ev], float))
    print(f"  fitted ability for {len(abilities)} users")

    events, skipped = build_events(rows, params, abilities, args.all_chains)
    print(f"\n{len(events)} scheduled-repeat events "
          f"({'all chains' if args.all_chains else f'chains starting >= {REPEAT_LOG_CUTOFF}'})")
    for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"  dropped {n:>6}  {reason}")
    print(f"  {sum(1 for e in events if e['z'] is None)} events lack problem params "
          f"(excluded from the model checks, kept in the clear-rate checks)")

    short = [e for e in events if e["gap_days"] * 24 < MIN_RECALL_HOURS]
    if short:
        k = sum(e["y"] for e in short)
        print(f"  {len(short)} events sit under the {MIN_RECALL_HOURS:g}h stability floor "
              f"(clear {k / len(short):.1%}) — unschedulable, so "
              f"{'KEPT (--keep-short)' if args.keep_short else 'EXCLUDED'}")
    if not args.keep_short:
        events = [e for e in events if e["gap_days"] * 24 >= MIN_RECALL_HOURS]
    if not events:
        print("no events — nothing to check")
        return

    report_by_rung(events, args.min_n, (kappa, growth, ln_c_prior))
    report_within_user(events)
    report_by_month(events, args.min_n)
    report_recent_windows(events, args.min_n, (kappa, growth, ln_c_prior))
    report_within_rung_decay(events, args.min_n)
    report_stability_correction(events, (kappa, growth, ln_c_prior), args.min_n)

    print("\n" + "=" * 78)
    print("6. OBSERVED VS PREDICTED RETENTION")
    print("   In-sample: these are the events the constants were fit on.")
    print("=" * 78)
    R = report_calibration(events, kappa, growth, ln_c_prior, False,
                           "population LN_C")
    if args.per_user_ln_c:
        report_calibration(events, kappa, growth, ln_c_prior, True,
                           "per-user LN_C (client-style shrinkage)")
    if R is not None:
        report_window(R)
    report_heldout(events, (kappa, growth, ln_c_prior))
    report_growth_forms(events, (kappa, growth, ln_c_prior))
    report_forward_per_user(events, (kappa, growth, ln_c_prior))
    report_equilibrium(events, rows, args.min_n,
                       (kappa, growth, ln_c_prior))
    report_per_user_step(rows, events)
    report_serves_per_entry(rows, events, params, abilities,
                            (kappa, growth, ln_c_prior), args.min_n, args.all_chains)


if __name__ == "__main__":
    main()
