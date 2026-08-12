#!/usr/bin/env python3
"""PAIRED held-out logloss between the joint MIRT and a production fit, with a
user-clustered bootstrap.

Both models are scored on the SAME held-out rows — same problems (shared
holdout_set hash), same profile partition — so the comparison is paired
row-by-row and the difference has far less variance than either logloss.

THE BOOTSTRAP CLUSTERS BY USER, not by row. Attempts within a user share an
ability estimate and a skill, so rows are not independent; resampling them
individually would understate the interval, in the direction that makes a
difference look real. The cluster is the user, not the profile, because a user's
overlapping profiles share the same person.

Profile labels differ between the two models and are mapped, not assumed:
production names a profile "{user8}_{YYYY-MM}" while the joint dev copy names it
"{user8}_{month_index}" with month_index = year*12 + (month-1). The PARTITION is
identical (verified separately against build_profiles); only the label differs.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.compare_heldout_logloss \\
      --joint /tmp/joint_params_prodwindow_s0.npz --prod-dir /tmp/prod_ho \\
      --seed 0 --frac 0.15
"""

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

EPS = 1e-6


def _p4pl(theta, rating, slope, floor, lapse):
    """The client's response form, pivoting at the 80%-clear rating — the same
    reconstruction validate_ratings.heldout_metrics uses, so a number from here
    is comparable with one from there."""
    from auxiliary.rating_dev.validate_ratings import clear_logit_vec
    b = rating - clear_logit_vec(floor, lapse) / np.maximum(slope, 1e-9)
    z = np.clip(slope * (theta - b), -35, 35)
    return np.clip(floor + (1 - floor - lapse) / (1 + np.exp(-z)), EPS, 1 - EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", required=True)
    ap.add_argument("--prod-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    from auxiliary.rating_dev.holdout_set import held_out_keys
    from auxiliary.rating_dev.validate_ratings import load_ratings, SIZES
    from utils.rating.features import build_profiles, load_first_time_attempts

    ho_keys, ho_hash = held_out_keys(args.seed, args.frac)
    ho_tids = {t for _s, t in ho_keys}
    print(f"held-out hash {ho_hash} | {len(ho_tids)} problems")

    z = np.load(args.joint, allow_pickle=True)
    j_rate = {str(t): float(v) for t, v in zip(z["tid"], z["rating"])}
    j_slope = {str(t): float(v) for t, v in zip(z["tid"], z["slope"])}
    j_floor = {str(t): float(v) for t, v in zip(z["tid"], z["floor"])}
    j_lapse = float(z["lapse"])
    j_theta = {str(p): float(t) for p, t in zip(z["profile"], z["theta"])}
    p_rate, p_slope, p_floor, p_lapse, p_theta = load_ratings(args.prod_dir)
    print(f"joint lapse {j_lapse:.4f} | production lapse {p_lapse:.4f}")

    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)]
    prof = build_profiles(df)
    prof = prof[prof["problem"].isin(ho_tids)]

    def to_joint_label(lab):
        u8, _, ym = str(lab).partition("_")
        y, _, m = ym.partition("-")
        try:
            return f"{u8}_{int(y) * 12 + int(m) - 1}"
        except ValueError:
            return None

    keep, rows = [], []
    for lab, tid, succ in zip(prof["profile"], prof["problem"], prof["success"]):
        jl = to_joint_label(lab)
        if (tid in p_rate and tid in j_rate and str(lab) in p_theta
                and jl is not None and jl in j_theta):
            keep.append((str(lab), jl, tid, int(succ)))
    if not keep:
        print("no rows scorable by BOTH models — nothing to compare")
        return
    print(f"{len(keep)} held-out rows scorable by BOTH models")

    users = np.array([lab[:8] for lab, _jl, _t, _s in keep])
    y = np.array([s for _l, _j, _t, s in keep], float)
    tids = [t for _l, _j, t, _s in keep]
    Pp = _p4pl(np.array([p_theta[l] for l, _j, _t, _s in keep]),
               np.array([p_rate[t] for t in tids]),
               np.array([p_slope[t] for t in tids]),
               np.array([p_floor[t] for t in tids]), p_lapse)
    Pj = _p4pl(np.array([j_theta[j] for _l, j, _t, _s in keep]),
               np.array([j_rate[t] for t in tids]),
               np.array([j_slope[t] for t in tids]),
               np.array([j_floor[t] for t in tids]), j_lapse)
    llp = -(y * np.log(Pp) + (1 - y) * np.log(1 - Pp))
    llj = -(y * np.log(Pj) + (1 - y) * np.log(1 - Pj))
    base = float(np.clip(y.mean(), EPS, 1 - EPS))
    const = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    print(f"\nbase rate {base:.4f} | constant-rate baseline {const:.4f}")
    print(f"  production  logloss {llp.mean():.4f}   improvement {const - llp.mean():+.4f}")
    print(f"  joint MIRT  logloss {llj.mean():.4f}   improvement {const - llj.mean():+.4f}")
    print(f"  delta (joint - production) {llj.mean() - llp.mean():+.5f} nats "
          f"({(llj.mean() - llp.mean()) * 1000:+.2f} mnat)")

    uu, inv = np.unique(users, return_inverse=True)
    by_u = [np.where(inv == i)[0] for i in range(len(uu))]
    rng = np.random.default_rng(args.seed)
    d = []
    for _ in range(args.boot):
        pick = rng.integers(0, len(uu), len(uu))
        idx = np.concatenate([by_u[i] for i in pick])
        d.append(llj[idx].mean() - llp[idx].mean())
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\nuser-clustered paired bootstrap ({len(uu)} users, {args.boot} reps)")
    print(f"  delta mean {d.mean():+.5f}   95% CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  P(joint better) = {float((d < 0).mean()):.3f}")
    print(f"  verdict: " + ("JOINT better" if hi < 0 else
                            "PRODUCTION better" if lo > 0 else
                            "INDISTINGUISHABLE — interval spans zero"))


if __name__ == "__main__":
    main()
