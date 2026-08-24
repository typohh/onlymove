#!/usr/bin/env python3
"""THE evaluation: everything derived from ONE user holdout.

Hold out whole accounts. Nothing they ever did touches training — not their
attempts, not their declarations — so every metric below is leak-free by
construction, and they all come from one split instead of three separate
training runs whose numbers cannot be combined.

    forward walk   one-step-ahead logloss over their novel attempts in time
                   order, from the FIRST one. No warmup, no blocks: predict
                   each attempt from everything before it, then fold it in.
    per sample     the subset on problems other users DID attempt — the
                   client's steady-state condition
    per problem    the subset on problems NO training user attempted, so the
                   parameters come from the feature prior alone
    survey rho     their rank declarations, per-server Spearman, n-weighted
    split-half     whether a user's position on each AXIS is identifiable at
                   all, by parity-splitting their attempts and correlating the
                   halves. Needs a --dumps fit (it reads the per-user arrays a
                   --prod-dirs run does not carry) and is scored on the fit's
                   OWN users, not the held-out ones -- there are too few deep
                   held-out accounts to bucket. So it is in-sample for the
                   loadings and mildly optimistic, unlike every row above it.

The per-sample / per-problem split is a PROPERTY of the held-out users' history,
not a second holdout: measured at roughly 79% seen / 21% unseen and stable across
folds. One pass yields both.

Each is reported three ways — RATING alone, RATING+AXIS, and the difference —
except survey rho, which reads rating and slope only, so the axes never enter it
and no such column exists.

Ability is a running GRID posterior, the same estimator whichever model is being
scored, so the comparison isolates the per-problem parameters rather than also
measuring whose ability field is parameterised better. That standardisation is
what makes two structurally different models comparable at all.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.eval_user_holdout \\
      --prod-dirs /tmp/prod_uh_s0 /tmp/prod_uh_s1 --seeds 0 1 --label production
  ./.venv/bin/python3 -m auxiliary.rating_dev.eval_user_holdout \\
      --dumps /tmp/joint_uh_s0.npz /tmp/joint_uh_s1.npz --seeds 0 1 --label MIRT
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

FLOOR_MAX = 0.78
GRID_N = 161


def project_loadings(axes_npz, params_dir=None):
    """Per-problem loadings from an axes.npz, which stores the LINEAR MAP rather
    than per-problem values: ((F - feat_mu)/feat_sd @ W.T)/rms * v_sd, matching
    train_axes' own output form.

    `params_dir` must point at the SAME rating fit the axes were trained
    against, since four of the feature columns are the rating model's outputs."""
    from utils.rating.residual_axes import build_axis_features
    z = np.load(axes_npz, allow_pickle=True)
    W, rms, v_sd = z["W"], z["rms"], z["v_sd"]
    feats, _r, _p, _a = build_axis_features(params_dir=params_dir)
    tids = sorted(feats)
    F = np.stack([feats[t] for t in tids])
    G = (((F - z["feat_mu"]) / z["feat_sd"]) @ W.T) / rms * v_sd
    return {t: G[i] for i, t in enumerate(tids)}


def _load(prod_dir, dump, axes_npz=None):
    from auxiliary.rating_dev.validate_ratings import load_ratings
    if prod_dir:
        rate, slope, floor, lam, _th = load_ratings(prod_dir)
        # NOT the shipped axes/axes.npz by default. Those were trained on every
        # user, including the ones being scored here, so using them would leak
        # exactly the accounts the holdout exists to protect — while the MIRT's
        # dumped loadings are clean, since its --holdout-user-frac run never saw
        # them. Pass --prod-axes pointing at an axis fit trained WITHOUT the
        # held-out users, or leave the column empty and say so.
        load = project_loadings(axes_npz, params_dir=prod_dir) if axes_npz else None
        return rate, slope, floor, lam, load
    z = np.load(dump, allow_pickle=True)
    rate = {str(t): float(v) for t, v in zip(z["tid"], z["rating"])}
    slope = {str(t): float(v) for t, v in zip(z["tid"], z["slope"])}
    floor = {str(t): float(v) for t, v in zip(z["tid"], z["floor"])}
    load = None
    if "loading" in z.files:
        load = {str(t): np.asarray(v, float)
                for t, v in zip(z["tid"], z["loading"])}
    return rate, slope, floor, float(z["lapse"]), load


# Prior standard deviation of a user's axis contribution g @ v, IN LOGITS. The
# prior lives in PREDICTION space on purpose: a prior variance stated in v-space
# is not scale-free, because rescaling the loadings by c and v by 1/c leaves
# every prediction identical, so only gT.Sigma.g is a well-posed thing to pin.
# Given it, the v-space prior is Sigma0 = (PRIOR_AX_SD^2 / mean|g|^2) * I, which
# is what makes the walk agree with itself across models whose loadings carry
# different magnitude by construction (the MIRT RMS-normalises g to 1 and keeps
# the magnitude in v so V_RIDGE can bound it; production's carry RMS 0.41).
PRIOR_AX_SD = 0.5

# Prefix-length bins: how many attempts PRECEDE the one being scored. Both latent
# user quantities are estimated from that prefix, so this is the axis along which
# to ask whether the axes need history before they pay — a question the
# seen/unseen split is blind to, since it describes the PROBLEM rather than how
# much is known about the USER.
PREFIX_BINS = ((0, 30, "pfx 0-30"), (30, 100, "pfx 30-100"),
               (100, 500, "pfx 100-500"), (500, 10 ** 9, "pfx 500+"))


def forward_walk(rate, slope, floor, lam, seqs, seen_problems, loadings=None,
                 k_axes=0, on_row=None, prior_prec=None):
    """One-step-ahead logloss, split by whether training saw the problem.

    BOTH latent user quantities are estimated from the whole prefix, causally.
    Ability rides a running grid log-posterior — the grid log-likelihood is
    additive, so its sufficient statistic accumulates and theta is the exact
    posterior mean at every attempt, no rate and no approximation. The axis
    position rides a RECURSIVE LAPLACE (online Newton) posterior over the same
    prefix: accumulate the Fisher information and take a Newton step, which is
    the same "use everything seen so far" standard.

    It replaces a single SGD step at a hand-set rate, and that was not a small
    difference. That rate multiplies the loading scale, which is a
    PARAMETERIZATION choice rather than a property of the axes, so one shared
    value scored two models at wildly different effective rates — reading the
    MIRT's axes at -0.47 mnat when they are worth +1.74. A Newton step has no
    rate at all and is invariant to the loading scale by construction, because
    the information term carries |g|^2 and divides it straight back out. So the
    question of whose rate to use, and whether to tune it on the scored rows,
    simply stops existing.

    K is read from the loadings; nothing here assumes a particular axis count,
    since the two models select K independently.

    `on_row(user, tid, y, pos, th, v, g, a, c, r, kap, span, p)` is called once
    per scored attempt, BEFORE the row is folded into either latent estimate, so
    a caller sees exactly the state the prediction was made from. It exists so a
    decomposition of the axis term — which axis, which tail — is read off THIS
    walk rather than off a second copy of it that could drift. `v` is rebound
    rather than mutated on update, so the array handed out stays this row's.

    `prior_prec` overrides the v-space prior precision the loadings imply. The
    default states the prior on the TOTAL contribution `g @ v` in prediction
    space and derives the v-space precision as `mean|g|^2 / PRIOR_AX_SD^2`, which
    is what makes two models with differently-scaled loadings comparable. That
    same derivation makes an arm which SHRINKS the loadings — clipping an axis to
    one side, say — score under a looser prior than the arm it is compared with,
    so the difference would be part ablation and part re-tuned prior. Pass the
    full model's value to every arm of such a comparison.
    """
    from auxiliary.rating_dev.validate_ratings import clear_logit_vec
    if loadings:
        G = np.stack(list(loadings.values()))
        k_axes = G.shape[1]
        # prior in prediction space -> v-space; see PRIOR_AX_SD
        if prior_prec is None:
            g2 = float((G ** 2).sum(axis=1).mean())
            prior_prec = max(g2, 1e-9) / (PRIOR_AX_SD ** 2)
    allr = np.array(list(rate.values()), float)
    grid = np.linspace(allr.min() - 2.0, allr.max() + 2.0, GRID_N)
    prior = -0.5 * ((grid - allr.mean()) / (allr.std() + 1e-9)) ** 2
    # [logloss sum, n, outcomes, |p-y| sum, correct-at-0.5 count]. The last two
    # are the same comparison in probability space: mnat is the right currency
    # for model selection but says nothing a reader can picture, whereas "the
    # predicted chance of success is off by X percentage points on average" does.
    acc = {"all": [0.0, 0, [], 0.0, 0], "seen": [0.0, 0, [], 0.0, 0],
           "unseen": [0.0, 0, [], 0.0, 0]}
    for _lo, _hi, _lbl in PREFIX_BINS:
        acc[_lbl] = [0.0, 0, [], 0.0, 0]
    eye = np.eye(k_axes) if k_axes else None
    bin_of = {}          # prefix length -> label, filled lazily per length seen
    for _u, seq in seqs.items():
        ll = prior.copy()
        v = np.zeros(k_axes) if k_axes else None
        # Laplace posterior precision, seeded at the prior. SPD throughout: the
        # seed is positive definite and every update adds a PSD rank-1 term.
        prec = eye * prior_prec if k_axes else None
        for pos, (tid, y) in enumerate(seq):
            a = float(slope[tid])
            c = float(np.clip(floor[tid], 0.0, FLOOR_MAX))
            r = float(rate[tid])
            span = max(1.0 - c - lam, 1e-6)
            kap = float(clear_logit_vec(np.array([c]), lam)[0])
            w = np.exp(ll - ll.max())
            w /= w.sum()
            th = float((w * grid).sum())
            g = loadings.get(tid) if loadings is not None else None
            ax = float(g @ v) if (g is not None and v is not None) else 0.0
            z = np.clip(a * (th - r) + kap + ax, -35, 35)
            p = float(np.clip(c + span / (1 + np.exp(-z)), 1e-9, 1 - 1e-9))
            loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            # `pos` is the number of attempts that PRECEDE this one for this
            # user, i.e. how much history both latent estimates have had. It is
            # the axis of interest when asking whether the axes need history to
            # pay off, which the seen/unseen split cannot answer.
            pb = bin_of.get(pos)
            if pb is None:
                pb = next(l for lo, hi, l in PREFIX_BINS if lo <= pos < hi)
                bin_of[pos] = pb
            if on_row is not None:
                on_row(_u, tid, y, pos, th, v, g, a, c, r, kap, span, p)
            for key in ("all", "seen" if tid in seen_problems else "unseen", pb):
                acc[key][0] += loss
                acc[key][1] += 1
                acc[key][2].append(y)
                acc[key][3] += abs(p - y)
                acc[key][4] += int((p >= 0.5) == (y >= 0.5))
            zg = np.clip(a * (grid - r) + kap + ax, -35, 35)
            pg = np.clip(c + span / (1 + np.exp(-zg)), 1e-9, 1 - 1e-9)
            ll += y * np.log(pg) + (1 - y) * np.log(1 - pg)
            if g is not None and v is not None:
                # 4PL chain rule, NOT the plain-logistic (y - p) * g: the floor
                # and lapse flatten dP/dz, so a residual near the floor carries
                # far less information about v than the logistic form claims.
                sig = 1.0 / (1.0 + np.exp(-z))
                dpdz = span * sig * (1.0 - sig)
                score = (y / p - (1.0 - y) / (1.0 - p)) * dpdz * g
                w = dpdz * dpdz / (p * (1.0 - p))        # Fisher weight
                prec = prec + w * np.outer(g, g)
                v = v + np.linalg.solve(prec, score)
    out = {}
    for key, (tot, n, ys, abserr, ncorr) in acc.items():
        if not n:
            continue
        base = float(np.clip(np.mean(ys), 1e-9, 1 - 1e-9))
        const = -(base * np.log(base) + (1 - base) * np.log(1 - base))
        out[key] = {"logloss": tot / n, "n": n, "constant": const,
                    "mae": abserr / n, "acc": ncorr / n, "base_rate": base}
    return out


def survey_rho_for(rate, slope, users):
    """Per-server Spearman on the held-out users' declarations only."""
    from auxiliary.rating_dev.compare_survey_rho import (_user_ratings,
                                                         _weighted_rho)
    from auxiliary.rating_dev.validate_ratings import SIZES
    from utils.rating.rank_surveys import (
        load_rank_surveys, latest_per_user_server, load_attempts_for_users,
        SQL_PATH_DEFAULT)
    surveys = latest_per_user_server(load_rank_surveys(SQL_PATH_DEFAULT))
    surveys = surveys[surveys["user_id"].astype(str).isin(set(users))]
    if not len(surveys):
        return None, 0
    att = load_attempts_for_users(SQL_PATH_DEFAULT,
                                  set(surveys["user_id"].unique()),
                                  first_time_only=True)
    ur = _user_ratings(rate, slope, surveys, att, SIZES)
    by = defaultdict(list)
    for _, row in surveys.iterrows():
        g = row["rank_value"]
        pref = str(row["user_id"])[:8]
        if g is None or (isinstance(g, float) and np.isnan(g)) or pref not in ur:
            continue
        by[row["server"]].append((ur[pref], float(g)))
    return _weighted_rho(by), sum(len(v) for v in by.values())


def evaluate_one(prod_dir, dump, seed, frac, axes_npz=None):
    from auxiliary.rating_dev.holdout_set import held_out_users
    from utils.rating.features import load_first_time_attempts

    hu_list, hu_hash = held_out_users(seed, frac)
    hu = set(hu_list)
    rate, slope, floor, lam, loadings = _load(prod_dir, dump, axes_npz)

    df = load_first_time_attempts()
    df = df[df["size"].isin([9, 13])].sort_values("created_at")
    is_held = df["user_id"].astype(str).isin(hu)
    seen = set(df.loc[~is_held, "problem"].astype(str))
    seqs = defaultdict(list)
    for u, t, ok in zip(df.loc[is_held, "user_id"].astype(str),
                        df.loc[is_held, "problem"].astype(str),
                        df.loc[is_held, "success"]):
        if t in rate:
            seqs[u].append((t, int(ok)))

    k = len(next(iter(loadings.values()))) if loadings else 0
    base = forward_walk(rate, slope, floor, lam, seqs, seen)
    # Each model at ITS OWN best online rate — the rate is a property of the
    # loading scale, which is a parameterisation choice, so a shared value
    # measures that choice rather than the axes.
    with_ax = (forward_walk(rate, slope, floor, lam, seqs, seen,
                            loadings=loadings, k_axes=k) if k else None)
    rho, n_decl = survey_rho_for(rate, slope, hu_list)
    n_all = sum(len(v) for v in seqs.values())
    n_unseen = sum(1 for v in seqs.values() for t, _ in v if t not in seen)
    return {"hash": hu_hash, "n_users": len(seqs), "n": n_all,
            "n_unseen": n_unseen, "k": k, "base": base, "axis": with_ax,
            "rho": rho, "n_decl": n_decl}


ROWS = (("all", "forward walk (all)"),
        ("seen", "  per SAMPLE  (problem seen)"),
        ("unseen", "  per PROBLEM (never trained)"))


def _fmt(vals, width=13):
    """mean +/- sd across seeds. With n = 2 the sd is |a-b|/sqrt(2) — a range
    wearing a symbol — so it is printed but should not be read as an interval."""
    a = np.array([v for v in vals if v is not None and np.isfinite(v)])
    if not len(a):
        return "—".center(width)
    if len(a) == 1:
        return f"{a[0]:.4f}".center(width)
    return f"{a.mean():.4f}±{a.std(ddof=1):.4f}".center(width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-dirs", nargs="*", default=[])
    ap.add_argument("--dumps", nargs="*", default=[])
    ap.add_argument("--prod-axes", nargs="*", default=[],
                    help="axes.npz per seed, trained WITHOUT that seed's held-out "
                         "users. The shipped axes/axes.npz saw everyone and must "
                         "not be used here.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--label", default="model")
    ap.add_argument("--axis-labels", nargs="*", default=[],
                    help="names for the axes, in order, used by the split-half "
                         "table; unnamed axes print as 'axis k'")
    ap.add_argument("--no-split-half", action="store_true",
                    help="skip the split-half tables (they add ~1min per dump)")
    ap.add_argument("--prior-ax-sd", type=float, default=None,
                    help="override PRIOR_AX_SD, the scorer's prior sd on a user's "
                         "axis contribution IN LOGITS. This is the measuring "
                         "instrument for every axis number, so it is swept "
                         "post-hoc against fixed dumps — no retraining, and the "
                         "arms being compared are unaffected by the choice.")
    args = ap.parse_args()
    if args.prior_ax_sd is not None:
        globals()["PRIOR_AX_SD"] = args.prior_ax_sd
        print(f"PRIOR_AX_SD = {args.prior_ax_sd}")

    srcs = [(d, None) for d in args.prod_dirs] + [(None, d) for d in args.dumps]
    if len(srcs) != len(args.seeds):
        sys.exit(f"one source per seed: {len(srcs)} sources, "
                 f"{len(args.seeds)} seeds")

    res = []
    for i, ((pd_, dp), sd) in enumerate(zip(srcs, args.seeds)):
        ax = (args.prod_axes[i] if i < len(args.prod_axes) else None)
        r = evaluate_one(pd_, dp, sd, args.frac, axes_npz=ax)
        print(f"  seed {sd}: hash {r['hash']} | {r['n_users']} users, {r['n']} "
              f"novel attempts, {r['n_unseen'] / max(r['n'], 1):.1%} on problems "
              f"training never saw | K={r['k']}", flush=True)
        res.append(r)

    print(f"\n{args.label} — held-out users, {len(res)} seed(s)")
    print(f"  {'':30}{'rating':^13}{'rating+axis':^13}{'axis adds':^13}")
    for key, name in ROWS:
        b = [r["base"].get(key, {}).get("logloss") for r in res]
        a = [(r["axis"] or {}).get(key, {}).get("logloss") for r in res]
        d = [None if (x is None or y is None) else (x - y) * 1000
             for x, y in zip(b, a)]
        print(f"  {name:30}{_fmt(b)}{_fmt(a)}{_fmt(d)}")
    print(f"  {'survey rho':30}{_fmt([r['rho'] for r in res])}"
          f"{'—':^13}{'axes do not enter':^19}")
    print("\n  logloss: lower better. axis adds: mnat/attempt, higher better.")

    if not args.no_split_half:
        from auxiliary.rating_dev.split_half import split_half_report
        for (pd_, dp), sd in zip(srcs, args.seeds):
            if dp is None:
                continue
            print(f"\n  seed {sd}:", end="")
            try:
                split_half_report(dp, args.axis_labels)
            except Exception as exc:      # never let a diagnostic kill the eval
                print(f"  split-half unavailable: {exc}")


if __name__ == "__main__":
    main()
