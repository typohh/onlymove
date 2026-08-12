#!/usr/bin/env python3
"""Phase 2 for the joint MIRT: the global slope scale that undoes dropout
compression, measured on HELD-OUT problems.

`pipeline.train_rating` trains with dropout and ships parameters scored without
it, which compresses the slopes toward the feature-conditional mean; its phase 2
freezes the trunk, refits the ability field free, and fits ONE global scalar to
undo it (shipped s = 1.694, worth 7.181 mnat by its own in-sample criterion). The
joint MIRT has no such stage and ships its compressed slopes raw. This measures
what the same correction is worth here, on held-out problems rather than
in-sample, so the number is directly comparable with the benchmark table.

Runs entirely from a --dump-params npz: the per-problem parameters are frozen
constants in phase 2, so nothing needs retraining.

TWO THINGS ARE LOAD-BEARING, both inherited from the production stage:

  THE ABILITY FIELD IS REFIT, not plugged in. Holding theta fixed at its phase-1
  value attenuates the fitted scale — error in the regressor — which is why
  production profiles the likelihood rather than reusing its own abilities.
  Both the unity baseline and the scaled fit refit it, so the reported gain is
  like-for-like.

  THE PIVOT IS THE RATING, not the curve's midpoint. Scoring is
  sigmoid(s*a*(theta - R) + kappa) with R the 80%-clear point from phase 1, so
  the rating stays put and only the steepness moves. Pivoting at the 50%-point
  instead rescales the rating axis itself, which is a different operation with a
  different meaning.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.mirt_phase2 \\
      --dump /tmp/joint_params_prodwindow_s0.npz --seed 0 --frac 0.15
"""

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

EPS = 1e-6
STEPS = 3000
LR = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None, help="a --dump-params npz (MIRT)")
    ap.add_argument("--prod-dir", default=None,
                    help="a ratings/ directory instead — runs the SAME held-out "
                         "test on production's parameters. Its phase 2 gain of "
                         "7.181 mnat is IN-SAMPLE and has never been checked "
                         "out-of-sample; if the correction hurts here too, the "
                         "MIRT is not missing anything.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.15)
    args = ap.parse_args()

    import torch
    from auxiliary.rating_dev.holdout_set import held_out_keys
    from auxiliary.rating_dev.validate_ratings import clear_logit_vec
    from utils.rating.features import build_profiles, load_first_time_attempts

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ho_keys, ho_hash = held_out_keys(args.seed, args.frac)
    ho_tids = {t for _s, t in ho_keys}
    print(f"held-out hash {ho_hash} | {len(ho_tids)} problems", flush=True)

    if args.prod_dir:
        from auxiliary.rating_dev.validate_ratings import load_ratings
        rate, slope, floor, lam, theta0 = load_ratings(args.prod_dir)
        prof_names = sorted(theta0)
        label_is_dump = False
    else:
        z = np.load(args.dump, allow_pickle=True)
        lam = float(z["lapse"])
        rate = {str(t): float(v) for t, v in zip(z["tid"], z["rating"])}
        slope = {str(t): float(v) for t, v in zip(z["tid"], z["slope"])}
        floor = {str(t): float(v) for t, v in zip(z["tid"], z["floor"])}
        prof_names = [str(p) for p in z["profile"]]
        label_is_dump = True
    print(f"dump: {len(rate)} problems, lapse {lam:.4f}, "
          f"{len(prof_names)} profiles", flush=True)

    df = load_first_time_attempts()
    df = df[df["size"].isin([9, 13])]
    prof = build_profiles(df)

    def to_dump_label(lab):
        u8, _, ym = str(lab).partition("_")
        y, _, m = ym.partition("-")
        try:
            return f"{u8}_{int(y) * 12 + int(m) - 1}"
        except ValueError:
            return None

    pidx = {p: i for i, p in enumerate(prof_names)}
    rows = []
    for lab, tid, succ in zip(prof["profile"], prof["problem"], prof["success"]):
        dl = to_dump_label(lab) if label_is_dump else str(lab)
        if dl in pidx and str(tid) in rate:
            rows.append((pidx[dl], str(tid), int(succ)))
    if not rows:
        print("no rows matched the dump's profiles")
        return
    pi = np.array([r[0] for r in rows], dtype=np.int64)
    tids = [r[1] for r in rows]
    y = np.array([r[2] for r in rows], dtype=np.float32)
    is_ho = np.array([t in ho_tids for t in tids])
    a = np.array([slope[t] for t in tids], dtype=np.float32)
    c = np.array([floor[t] for t in tids], dtype=np.float32)
    R = np.array([rate[t] for t in tids], dtype=np.float32)
    kap = clear_logit_vec(c, lam).astype(np.float32)
    print(f"{len(y)} rows | {int(is_ho.sum())} held-out", flush=True)

    ta = torch.tensor(a, device=device)
    tc = torch.tensor(c, device=device)
    tR = torch.tensor(R, device=device)
    tk = torch.tensor(kap, device=device)
    ty = torch.tensor(y, device=device)
    tp = torch.tensor(pi, device=device)
    tr = torch.tensor(np.where(~is_ho)[0], device=device)
    th_i = torch.tensor(np.where(is_ho)[0], device=device)

    def fit(free_scale):
        theta = torch.zeros(len(prof_names), device=device, requires_grad=True)
        ls = torch.zeros((), device=device, requires_grad=True)
        opt = torch.optim.Adam([theta, ls] if free_scale else [theta], lr=LR)
        for _ in range(STEPS):
            s = torch.exp(ls) if free_scale else 1.0
            i = tr
            z_ = s * ta[i] * (theta[tp[i]] - tR[i]) + tk[i]
            p = (tc[i] + (1 - tc[i] - lam)
                 * torch.sigmoid(z_.clamp(-35, 35))).clamp(EPS, 1 - EPS)
            loss = torch.nn.functional.binary_cross_entropy(p, ty[i])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            s = float(torch.exp(ls)) if free_scale else 1.0
            out = {}
            for name, idx in (("train", tr), ("heldout", th_i)):
                z_ = s * ta[idx] * (theta[tp[idx]] - tR[idx]) + tk[idx]
                p = (tc[idx] + (1 - tc[idx] - lam)
                     * torch.sigmoid(z_.clamp(-35, 35))).clamp(EPS, 1 - EPS)
                out[name] = float(
                    torch.nn.functional.binary_cross_entropy(p, ty[idx]))
            out["s"] = s
        return out

    base = fit(False)
    scaled = fit(True)
    print(f"\nPHASE 2 on {len(th_i)} held-out rows")
    print(f"  s = 1 (unity)     train {base['train']:.4f}   "
          f"held-out {base['heldout']:.4f}")
    print(f"  s = {scaled['s']:.4f} (fitted) train {scaled['train']:.4f}   "
          f"held-out {scaled['heldout']:.4f}")
    print(f"  HELD-OUT gain     {(base['heldout'] - scaled['heldout']) * 1000:+.3f} "
          f"mnat/attempt")
    print(f"  (production's shipped s = 1.694, its own in-sample gain 7.181 mnat)")


if __name__ == "__main__":
    main()
