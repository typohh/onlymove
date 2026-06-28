"""Does per-attempt Fisher information fall at higher ranks? (And does that make the
window-cap boundary worse for strong players?)

The live rank estimator weights each attempt by its Fisher information at the served
position (P≈0.8 for a problem served at the user's rank). That information is

    info = g^2 / (P(1-P)),  g = span * slope_rank * praw * (1-praw),

where the per-attempt slope in RANK space is slope_rank = a_p / |M'(rating)| — the
problem's 2PL discrimination divided by the local steepness of the rating->rank
curve M. So info scales as slope_rank^2: if M is shallower (|M'| larger) or a_p
smaller at the top, info collapses quadratically, the 500-sample window holds even
less total information (acc(500) = 500*info), and the cap binds at an even higher
departing weight -> a worse correct->down boundary artifact for strong players.

This bins the whole rated corpus by rank and reports the trend, plus where the
five post-update users sit.
"""
import math
import os
import sqlite3
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.append(REPO)

from utils.ratings_io import load_size_value_maps, load_rating_rank_curve  # noqa: E402
from utils.fit_rank_curves import interp_extrap_vec  # noqa: E402

SIZES = [9, 13]
RATINGS_DIR = os.path.join(REPO, "ratings")
DB = os.path.join(REPO, "attempts", "database.sqlite")
LAPSE, CAL_P, FLOOR_MAX = 0.0347, 0.8, 0.78
SLOPE_LO, SLOPE_HI, MSLOPE_FLOOR, SLOPE_FALLBACK, A_REF = 0.05, 3.0, 1e-3, 0.28, 1.6


def clamp_floor(c):
    return np.clip(c, 0.0, FLOOR_MAX)


def clear_kappa(c):
    q = np.clip((CAL_P - c) / (1 - c - LAPSE), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def main():
    ratings, slopes, floors = {}, {}, {}
    for s in SIZES:
        r, sl, f = load_size_value_maps(RATINGS_DIR, s, missing_ok=True)
        ratings.update(r); slopes.update(sl); floors.update(f)
    xs, ys = load_rating_rank_curve(RATINGS_DIR)

    tids = list(ratings.keys())
    rating = np.array([ratings[t] for t in tids], float)
    aP = np.array([slopes.get(t, A_REF) for t in tids], float)
    aP = np.where(aP > 0, aP, A_REF)
    floor = clamp_floor(np.array([floors.get(t, 0.0) for t in tids], float))

    rank = interp_extrap_vec(rating, xs, ys)               # M(rating) -> rank index
    h = 1e-3
    mprime = (interp_extrap_vec(rating + h, xs, ys) - interp_extrap_vec(rating - h, xs, ys)) / (2 * h)
    slope_rank = aP / np.maximum(np.abs(mprime), MSLOPE_FLOOR)
    clamped = (slope_rank < SLOPE_LO) | (slope_rank > SLOPE_HI)
    slope_rank = np.clip(slope_rank, SLOPE_LO, SLOPE_HI)

    # info per attempt when SERVED AT RANK (R == x => P == CAL_P).
    span = np.maximum(1 - floor - LAPSE, 1e-9)
    kappa = clear_kappa(floor)
    praw = 1.0 / (1.0 + np.exp(-kappa))                    # sigmoid(kappa) at R==x
    g = span * slope_rank * praw * (1 - praw)
    info = g * g / (CAL_P * (1 - CAL_P))

    # bin by rank index (≈ kyu→dan; idx 19 == 1k, idx 20 == 1d)
    edges = [-5, 5, 10, 14, 17, 19, 21, 23, 25, 28, 45]

    def lab(idx):
        s = idx - 19.0
        return f"{abs(round(s - 1)):.0f}k" if s < 0.5 else f"{round(s):.0f}d"

    print(f"\ninfo/attempt vs rank (served at rank, P={CAL_P}). corpus n={len(tids)}; curve "
          f"rank knots {ys[0]-19:+.0f}..{ys[-1]-19:+.0f} (1k=0).\n")
    print(f"{'rank band':<18}{'n':>7}{'med a_p':>9}{'med|Mp|':>10}{'med slope':>11}"
          f"{'med info':>11}{'acc(500)':>10}{'departW@16':>11}{'%clamp':>8}")
    print('-' * 95)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rank >= lo) & (rank < hi)
        if m.sum() < 20:
            continue
        med_info = float(np.median(info[m]))
        acc500 = 500 * med_info
        departW = max(0.0, 1 - acc500 / 16.0)
        band = f"{lab(lo)}..{lab(hi)} [{lo:g},{hi:g})"
        print(f"{band:<18}{m.sum():>7}{np.median(aP[m]):>9.2f}{np.median(np.abs(mprime[m])):>10.2f}"
              f"{np.median(slope_rank[m]):>11.3f}{med_info:>11.4f}{acc500:>10.2f}"
              f"{departW:>11.2f}{100*clamped[m].mean():>7.0f}%")

    # place the 5 post-update users by their median first-time served rank
    print("\nwhere the post-update users sit (median first-time served rank):")
    con = sqlite3.connect(DB)
    for uid in ["<redacted-user-id>", "<redacted-user-id>", "<redacted-user-id>", "<redacted-user-id>", "<redacted-user-id>"]:
        rows = con.execute("SELECT problem FROM tsumego_reports WHERE user_id LIKE ? AND is_repeat=0",
                           (uid + "%",)).fetchall()
        rks = [float(interp_extrap_vec(np.array([ratings[p]]), xs, ys)[0]) for (p,) in rows if p in ratings]
        if rks:
            md = float(np.median(rks))
            print(f"  {uid:<10} n={len(rks):>5}  median rank idx {md:6.1f} (~{lab(md)})")
    con.close()
    print()


if __name__ == "__main__":
    main()
