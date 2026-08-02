"""Phase-2 ability-unit study: does the OVERLAPPING profile grid (phase 1's convention) fit the
global slope/floor scales better than the non-overlapping calendar blocks phase 2 ships?

pipeline.train_rating uses two different ability-unit conventions:
  phase 1  utils.rating.features.build_profiles — 2-month windows STEPPED ONE MONTH, so every
           attempt is exploded into two profile rows and consecutive units share half their data.
  phase 2  fit_global_scales — user x (month // P2_BIN_MONTHS), disjoint blocks on a fixed calendar
           grid, every attempt counted once.

The estimands differ (phase 1 trains a feature net, where duplicating rows is a near-uniform
reweighting; phase 2 is a profile likelihood for two scalars, whose bias depends on how the
nuisance ability field is parameterized), so symmetry is not itself an argument. But the shipped
grid has an arbitrary PHASE — whether a user's career splits cleanly into blocks or straddles a
boundary depends on the calendar — and the overlapping grid is exactly the average over phases.
This harness measures whether that matters.

Three criteria, in order of authority:

  1. CERTIFICATE (the only one that speaks to correctness). Outcomes are regenerated from the
     frozen per-problem parameters at true s = f = 1 under a KNOWN continuous-time ability process,
     and every variant refits. Truth is 1.0; the variant closest to it is less biased. The ability
     process is a per-user Brownian path in continuous time, deliberately aligned to NO grid, so it
     cannot advantage either candidate. Run at several drift levels: with no drift the coarsest
     unit is correct by construction and finer grids only add incidental parameters; with drift the
     finer/overlapping grids can track it. The shipped choice must hold up across the range.
  2. GRID-PHASE STABILITY. The same non-overlapping grid shifted one month. A material move between
     the two phases means the shipped number depends on a knob nobody chose, which is an argument
     for the phase-averaged (overlapping) estimator regardless of the certificate.
  3. Real-data s and f per variant, for magnitude. NOT a selection criterion: in-sample BCE is not
     comparable across variants (they differ in row count AND in nuisance-parameter count, and a
     more flexible ability field always looks better in sample).

Everything except the unit assignment is the production recipe, imported from pipeline.train_rating
so this cannot drift against what ships. Frozen per-problem parameters come from the shipped CSVs,
un-scaled back through ratings/rating_constants.json (phase 2 fits on phase 1's raw output).

  ./.venv/bin/python3 -m auxiliary.rating_dev.eval_phase2_units
  ./.venv/bin/python3 -m auxiliary.rating_dev.eval_phase2_units --seeds 5 --real-only
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pipeline.train_rating import (  # noqa: E402
    CALIBRATION_P, CLIENT_FLOOR_MAX, EPS, P2_BIN_MONTHS, P2_LR, P2_STEPS,
)
from utils.constants import ALL_BOARD_SIZES  # noqa: E402
from utils.rating.features import load_first_time_attempts  # noqa: E402
from utils.rating.ratings_io import FLOOR_CSV, RATINGS_CSV, SLOPE_CSV  # noqa: E402
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402

RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")

# Synthetic-ability process. The certificate generates theta as a per-user Brownian path whose
# career-long standard deviation is DRIFT_SD (in gauge units, so directly comparable to the
# corpus difficulty spread of 1). Levels are swept: 0 makes the coarsest unit correct by
# construction, larger values reward grids that track time.
DRIFT_LEVELS = (0.0, 0.15, 0.5)
USER_OFFSET_SD = 0.3     # per-user offset of true theta from its attempt-mean rating


def unit_codes(user_codes, mon, spec):
    """Map attempts to ability units. Returns (row_idx, codes, n_units) where row_idx indexes the
    attempt arrays — LONGER than the attempt count for overlapping grids, which enter each attempt
    once per window covering it (phase 1's convention: an attempt at month M trains the units
    starting at M, M-1, ... M-window+1)."""
    kind = spec[0]
    n = len(mon)
    if kind == "user":
        row_idx, tcode = np.arange(n), np.zeros(n, dtype=np.int64)
    elif kind == "nonoverlap":
        months, offset = spec[1], spec[2]
        row_idx, tcode = np.arange(n), (mon + offset) // months
    elif kind == "overlap":
        window = spec[1]                                   # stepped one month
        row_idx = np.tile(np.arange(n), window)
        tcode = np.concatenate([mon - w for w in range(window)])
    else:
        raise ValueError(spec)
    key = user_codes[row_idx].astype(np.int64) * (int(tcode.max() - tcode.min()) + 2) + \
        (tcode - tcode.min())
    codes, _ = pd.factorize(key)
    return row_idx, codes.astype(np.int64), int(codes.max()) + 1


VARIANTS = {
    "nonoverlap2_p0": ("nonoverlap", P2_BIN_MONTHS, 0),   # SHIPPED
    "nonoverlap2_p1": ("nonoverlap", P2_BIN_MONTHS, 1),   # same grid, phase-shifted one month
    "overlap2": ("overlap", P2_BIN_MONTHS),               # phase 1's convention
    "nonoverlap1": ("nonoverlap", 1, 0),                  # granularity control (finer)
    "user": ("user",),                                    # granularity control (no time split)
}


def merge_thin_units(codes, row_users, min_rows):
    """Relabel every unit below `min_rows` into a single per-user catch-all, keeping all rows.
    Fewer nuisance parameters where they are least affordable, without a cull."""
    sizes = np.bincount(codes)
    key = np.where(sizes[codes] < min_rows, -1 - row_users, codes)
    new = pd.factorize(key)[0].astype(np.int64)
    return new, int(new.max()) + 1


def fit_scales(R0, A, C, y, row_idx, codes, n_units, lapse, device, steps, lr,
               steps_free=None, snapshot_every=None, pin_s=None, pin_f=None, pin_theta=None):
    """The production phase-2 fit (pipeline.train_rating.fit_global_scales) with the ability unit
    injected. Identical in every other respect: client response form pivoting at the shipped
    rating, kappa recomputed from the scaled floor, full-batch Adam, unity baseline first and the
    free run warm-started from its thetas."""
    tR0 = torch.tensor(R0[row_idx], device=device)
    tA = torch.tensor(A[row_idx], device=device)
    tC = torch.tensor(C[row_idx], device=device)
    ty = torch.tensor(y[row_idx].astype(np.float64), device=device)
    tcodes = torch.tensor(codes, device=device)
    th0 = np.bincount(codes, weights=R0[row_idx], minlength=n_units) / \
        np.maximum(np.bincount(codes, minlength=n_units), 1)

    traj = []

    # pin_theta holds a PER-ROW ability (the generating truth in a certificate run), which removes
    # the nuisance parameters entirely: whatever bias survives is structural, not incidental.
    tth_fixed = None if pin_theta is None else \
        torch.tensor(np.asarray(pin_theta, dtype=np.float64)[row_idx], device=device)

    def run(free_scales, th_init):
        th = nn.Parameter(torch.tensor(th_init, device=device, dtype=torch.float64))
        # A pinned scale is held at its given value and kept out of the optimizer, so the fit is
        # the profile likelihood of the OTHER scale (theta still refits). Pinning one at its known
        # truth is what separates a genuine bias in the other from leakage between the two.
        ls = nn.Parameter(torch.zeros((), device=device, dtype=torch.float64)) if pin_s is None \
            else torch.tensor(float(np.log(pin_s)), device=device, dtype=torch.float64)
        lf = nn.Parameter(torch.zeros((), device=device, dtype=torch.float64)) if pin_f is None \
            else torch.tensor(float(np.log(pin_f)), device=device, dtype=torch.float64)
        free = [p for p in (ls, lf) if isinstance(p, nn.Parameter)]
        params = ([] if tth_fixed is not None else [th]) + (free if free_scales else [])
        if not params:                                   # nothing to train: score once
            params = [nn.Parameter(torch.zeros((), device=device, dtype=torch.float64))]
            n_steps = 1
        else:
            n_steps = steps_free if (free_scales and steps_free) else steps
        opt = torch.optim.Adam(params, lr=lr)
        for _step in range(n_steps):
            s = torch.exp(ls) if free_scales else 1.0
            f = torch.exp(lf) if free_scales else 1.0
            c2 = (f * tC).clamp(0.0, CLIENT_FLOOR_MAX)
            span = (1.0 - c2 - lapse).clamp(min=EPS)
            q = ((CALIBRATION_P - c2) / span).clamp(EPS, 1.0 - EPS)
            kap = torch.log(q / (1.0 - q))
            th_row = tth_fixed if tth_fixed is not None else th[tcodes]
            z = (s * tA * (th_row - tR0) + kap).clamp(-35.0, 35.0)
            P = (c2 + span * torch.sigmoid(z)).clamp(EPS, 1.0 - EPS)
            loss = -(ty * torch.log(P) + (1.0 - ty) * torch.log(1.0 - P)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if free_scales and snapshot_every and (_step + 1) % snapshot_every == 0:
                traj.append((_step + 1, float(torch.exp(ls)), float(torch.exp(lf))))
        return (float(loss), float(torch.exp(ls)), float(torch.exp(lf)),
                th.detach().cpu().numpy())

    bce_unity, _, _, th_base = run(False, th0)
    bce_scaled, s, f, _ = run(True, th_base)
    sizes = np.bincount(codes, minlength=n_units)
    return {"s": s, "f": f, "traj": traj, "bce": bce_scaled,
            "gain_mnat": (bce_unity - bce_scaled) * 1000,
            "n_units": n_units, "n_rows": len(row_idx),
            "median_unit_rows": float(np.median(sizes)),
            "thin_unit_share": float((sizes < 8).mean())}


def load_frozen_params():
    """Shipped per-problem values, un-scaled back to phase 1's raw output (phase 2 fits ON that:
    quality = s*slope and floor = f*floor are its OUTPUT, so feeding them back in would double-
    apply the correction). Rating is already unscaled — it anchors the curve phase 2 scales."""
    with open(os.path.join(RATINGS_DIR, "rating_constants.json")) as fh:
        const = json.load(fh)
    # floor_scale is absent once production stops scaling the floor; then the shipped floor already
    # IS the phase-1 value and needs no un-scaling.
    s0, f0 = float(const["slope_scale"]), float(const.get("floor_scale", 1.0))
    out = {}
    for size in ALL_BOARD_SIZES:
        sdir = os.path.join(RATINGS_DIR, str(size))
        rat = pd.read_csv(os.path.join(sdir, RATINGS_CSV))
        qual = pd.read_csv(os.path.join(sdir, SLOPE_CSV))
        flo = pd.read_csv(os.path.join(sdir, FLOOR_CSV))
        m = rat.merge(qual, on="tsumego_id").merge(flo, on="tsumego_id")
        for tid, r, q, c in zip(m.tsumego_id, m.rating, m.quality, m.floor):
            out[tid] = (float(r), float(q) / s0, float(c) / f0)
    return out, const


def synth_outcomes(R0, A, C, lapse, user_codes, t_days, drift_sd, rng, return_theta=False):
    """Outcomes from the frozen curves at TRUE s = f = 1 under a known continuous-time ability
    path: per user, theta starts near their attempt-mean rating (serving is difficulty-matched)
    and follows a Brownian path whose career-long sd is drift_sd. Grid-free by construction, so no
    unit convention is favoured."""
    n = len(R0)
    theta = np.empty(n)
    order = np.lexsort((t_days, user_codes))          # users contiguous, each in time order
    bounds = np.flatnonzero(np.diff(user_codes[order])) + 1
    for m in np.split(order, bounds):
        base = R0[m].mean() + rng.normal(0.0, USER_OFFSET_SD)
        if drift_sd > 0 and len(m) > 1:
            t = t_days[m]
            span = max(t[-1] - t[0], 1e-9)
            dt = np.diff(t, prepend=t[0])
            walk = np.cumsum(rng.normal(0.0, 1.0, len(m)) * np.sqrt(dt / span)) * drift_sd
        else:
            walk = 0.0
        theta[m] = base + walk
    c = np.clip(C, 0.0, CLIENT_FLOOR_MAX)
    span_p = np.maximum(1.0 - c - lapse, EPS)
    q = np.clip((CALIBRATION_P - c) / span_p, EPS, 1.0 - EPS)
    kap = np.log(q / (1.0 - q))
    P = c + span_p / (1.0 + np.exp(-np.clip(A * (theta - R0) + kap, -35, 35)))
    y = (rng.random(n) < P).astype(np.float64)
    return (y, theta) if return_theta else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="synthetic replicates per drift level")
    ap.add_argument("--steps", type=int, default=P2_STEPS)
    ap.add_argument("--lr", type=float, default=P2_LR)
    ap.add_argument("--real-only", action="store_true", help="skip the certificate")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--floor-study", action="store_true",
                    help="shipped variant only: per-seed floor-scale bias with its spread, plus "
                         "convergence trajectories that separate a genuine bias from an optimizer "
                         "still sliding away from its unity initialisation")
    ap.add_argument("--mechanism", action="store_true",
                    help="attribute the floor bias: pin abilities at the generating truth, and vary how many rows each ability is fit on")
    ap.add_argument("--identify", action="store_true",
                    help="separate a floor-scale bias from slope leakage: pin each scale at its "
                         "known truth in turn, and profile the likelihood over the floor scale")
    ap.add_argument("--long-factor", type=int, default=4,
                    help="step multiplier for the convergence trajectories")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params, const = load_frozen_params()
    df = load_first_time_attempts()
    df = df[df["problem"].isin(params)].copy()
    ts = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df, ts = df[ts.notna()].copy(), ts[ts.notna()]
    mon = (ts.dt.year.to_numpy() * 12 + ts.dt.month.to_numpy() - 1).astype(np.int64)
    t_days = (ts.astype("int64").to_numpy() / 8.64e13)
    user_codes = pd.factorize(df["user_id"].astype(str))[0].astype(np.int64)
    pr = [params[t] for t in df["problem"]]
    R0 = np.array([p[0] for p in pr])
    A = np.array([p[1] for p in pr])
    C = np.array([p[2] for p in pr])
    y_real = df["success"].to_numpy(dtype=np.float64)
    lapse = load_client_lapse(const.get("fat_finger_rate", 0.02))
    print(f"attempts {len(df)} | users {len(np.unique(user_codes))} | months "
          f"{mon.min()}..{mon.max()} | lapse {lapse:.4f} | shipped s {const['slope_scale']} "
          f"f {const.get('floor_scale', 1.0)} | device {device.type}")

    if args.mechanism:
        rng = np.random.default_rng(15)
        y_syn, th_true = synth_outcomes(R0, A, C, lapse, user_codes, t_days, 0.15, rng,
                                        return_theta=True)
        ri, co, nu = unit_codes(user_codes, mon, VARIANTS["nonoverlap2_p0"])
        ru, cu, nu_u = unit_codes(user_codes, mon, VARIANTS["user"])
        row_users = user_codes[ri]
        print("\n=== MECHANISM: how much of the floor deficit is the ABILITY NUISANCE? ===\n"
              "Every ability is fit on the same rows it then helps score, so it absorbs part of\n"
              "each unit's realised luck. Pinning abilities at the generating truth removes the\n"
              "nuisance entirely; coarser and thin-merged units keep it but make it cheaper per\n"
              "row. Truth is f = 1.000 throughout.\n", flush=True)
        arms = [("production units (user x 2-month)", ri, co, nu, None),
                ("ABILITIES PINNED AT TRUTH", ri, co, nu, th_true),
                ("one ability per user (coarser)", ru, cu, nu_u, None)]
        for min_rows in (8, 30):
            mc, mn = merge_thin_units(co, row_users, min_rows)
            arms.append((f"thin units (<{min_rows} rows) merged per user", ri, mc, mn, None))
        for label, rix, cx, nx, pin in arms:
            r = fit_scales(R0, A, C, y_syn, rix, cx, nx, lapse, device, args.steps, args.lr,
                           pin_theta=pin)
            tag = "  <- pure structural bias" if pin is not None else ""
            print(f"  {label:<38} units {r['n_units']:>6}  s {r['s']:.4f}  f {r['f']:.4f}  "
                  f"(f bias {r['f'] - 1:+.4f}){tag}", flush=True)
        return

    if args.identify:
        ri, co, nu = unit_codes(user_codes, mon, VARIANTS["nonoverlap2_p0"])
        fit = lambda y, **kw: fit_scales(R0, A, C, y, ri, co, nu, lapse, device,  # noqa: E731
                                         args.steps, args.lr, **kw)
        rng = np.random.default_rng(15)
        y_syn = synth_outcomes(R0, A, C, lapse, user_codes, t_days, 0.15, rng)
        print("\n=== IDENTIFICATION: is the floor deficit its own effect, or slope leakage? ===\n"
              "The two are not orthogonal — a higher floor with a steeper slope mimics a lower\n"
              "floor with a shallower one — so a joint fit alone cannot attribute the deficit.\n"
              "Pinning one scale at its KNOWN truth removes that route: if the other still misses,\n"
              "the miss is its own. The profile then shows whether the likelihood can resolve the\n"
              "difference at all, or is flat enough that the deficit is an unidentified plateau.\n",
              flush=True)
        for label, y_case, truth in [("SYNTHETIC (true s = f = 1)", y_syn, True),
                                     ("REAL DATA (truth unknown)", y_real, False)]:
            print(f"-- {label} --", flush=True)
            both = fit(y_case)
            print(f"  both free            : s {both['s']:.4f}  f {both['f']:.4f}", flush=True)
            if truth:
                ps = fit(y_case, pin_s=1.0)
                pf = fit(y_case, pin_f=1.0)
                print(f"  slope PINNED at truth: s 1.0000  f {ps['f']:.4f}   "
                      f"<- floor deficit surviving with no slope freedom", flush=True)
                print(f"  floor PINNED at truth: s {pf['s']:.4f}  f 1.0000   "
                      f"<- slope with no floor freedom", flush=True)
            print("  profile over pinned floor (theta and slope refit at each point):", flush=True)
            prof = []
            for fv in (0.85, 0.90, 0.95, 1.00, 1.05):
                r = fit(y_case, pin_f=fv)
                prof.append((fv, r["s"], r["bce"]))
            best = min(p[2] for p in prof)
            for fv, sv, bce in prof:
                print(f"    f={fv:.2f}  best s {sv:.4f}   BCE {bce:.6f}   "
                      f"{(bce - best) * 1000:+.3f} mnat vs best", flush=True)
            ds = (prof[-1][1] - prof[0][1]) / (prof[-1][0] - prof[0][0])
            print(f"  trade-off ds/df over the grid: {ds:+.3f} "
                  f"(large magnitude = the two scales substitute for each other)\n", flush=True)
        return

    if args.floor_study:
        ri, co, nu = unit_codes(user_codes, mon, VARIANTS["nonoverlap2_p0"])
        fit = lambda y, **kw: fit_scales(R0, A, C, y, ri, co, nu, lapse, device,  # noqa: E731
                                         args.steps, args.lr, **kw)
        print("\n=== FLOOR SCALE: per-seed recovery of a TRUE f = 1 (shipped unit definition) ===",
              flush=True)
        for drift in DRIFT_LEVELS:
            ss, ff = [], []
            for seed in range(args.seeds):
                rng = np.random.default_rng(1000 * seed + int(drift * 100))
                r = fit(synth_outcomes(R0, A, C, lapse, user_codes, t_days, drift, rng))
                ss.append(r["s"]); ff.append(r["f"])
                print(f"  drift {drift:.2f} seed {seed}:  s {r['s']:.4f}   f {r['f']:.4f}",
                      flush=True)
            s_a, f_a = np.array(ss), np.array(ff)
            n = len(f_a)
            print(f"  drift {drift:.2f} SUMMARY: s {s_a.mean():.4f} (sd {s_a.std(ddof=1):.4f}, "
                  f"se {s_a.std(ddof=1) / np.sqrt(n):.4f})  |  f {f_a.mean():.4f} "
                  f"(sd {f_a.std(ddof=1):.4f}, se {f_a.std(ddof=1) / np.sqrt(n):.4f})  "
                  f"| f bias {f_a.mean() - 1:+.4f}\n", flush=True)

        # Convergence: a genuine bias settles; an unconverged optimizer keeps sliding. Both scales
        # start at exactly 1.0, so a monotone drift that has not flattened by the shipped step
        # count would explain the deficit without any bias being present.
        long_steps = args.steps * args.long_factor
        snap = max(long_steps // 10, 1)
        print(f"=== CONVERGENCE: free stage run to {long_steps} steps "
              f"({args.long_factor}x shipped), f snapshots ===", flush=True)
        cases = [("REAL DATA", y_real)]
        for seed in range(2):
            rng = np.random.default_rng(1000 * seed + 15)
            cases.append((f"synthetic drift 0.15 seed {seed}",
                          synth_outcomes(R0, A, C, lapse, user_codes, t_days, 0.15, rng)))
        for label, y_case in cases:
            r = fit(y_case, steps_free=long_steps, snapshot_every=snap)
            print(f"  {label}:", flush=True)
            print("    " + "  ".join(f"{st}:f={fv:.4f}" for st, _sv, fv in r["traj"]), flush=True)
            print(f"    final s {r['s']:.4f}  f {r['f']:.4f}   "
                  f"(shipped step count would stop at {args.steps})", flush=True)
        return

    units = {v: unit_codes(user_codes, mon, VARIANTS[v]) for v in variants}
    print("\n=== REAL DATA (magnitude only — in-sample BCE is NOT comparable across variants) ===")
    print(f"{'variant':<16}{'s':>8}{'f':>8}{'gain_mnat':>11}{'units':>8}{'rows':>9}"
          f"{'med/unit':>10}{'thin<8':>8}")
    for v in variants:
        row_idx, codes, n_units = units[v]
        r = fit_scales(R0, A, C, y_real, row_idx, codes, n_units, lapse, device,
                       args.steps, args.lr)
        print(f"{v:<16}{r['s']:>8.3f}{r['f']:>8.3f}{r['gain_mnat']:>11.2f}{r['n_units']:>8d}"
              f"{r['n_rows']:>9d}{r['median_unit_rows']:>10.0f}{r['thin_unit_share']:>8.2f}")

    if args.real_only:
        return
    print("\n=== CERTIFICATE (outcomes regenerated at TRUE s = f = 1; closer to 1.000 is less "
          "biased) ===")
    for drift in DRIFT_LEVELS:
        rec = {v: {"s": [], "f": []} for v in variants}
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 * seed + int(drift * 100))
            y_syn = synth_outcomes(R0, A, C, lapse, user_codes, t_days, drift, rng)
            for v in variants:                     # PAIRED: same synthetic set for every variant
                row_idx, codes, n_units = units[v]
                r = fit_scales(R0, A, C, y_syn, row_idx, codes, n_units, lapse, device,
                               args.steps, args.lr)
                rec[v]["s"].append(r["s"]); rec[v]["f"].append(r["f"])
        print(f"\n-- ability drift sd {drift:.2f} gauge units | {args.seeds} paired replicates --")
        print(f"{'variant':<16}{'s_mean':>9}{'s_sd':>8}{'s_bias':>9}{'f_mean':>9}{'f_bias':>9}")
        for v in variants:
            s_arr, f_arr = np.array(rec[v]["s"]), np.array(rec[v]["f"])
            print(f"{v:<16}{s_arr.mean():>9.3f}{s_arr.std():>8.3f}{s_arr.mean() - 1:>+9.3f}"
                  f"{f_arr.mean():>9.3f}{f_arr.mean() - 1:>+9.3f}")


if __name__ == "__main__":
    main()
