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


def forward_walk(rate, slope, floor, lam, seqs, seen_problems, loadings=None,
                 k_axes=0, v_lr=0.05):
    """One-step-ahead logloss, split by whether training saw the problem.

    The grid log-likelihood is additive in attempts, so the running posterior
    costs one vector add per attempt rather than a refit per step. With
    loadings, v is learned online from zero — predict, then update — so the
    axes are measured causally rather than fitted on the history they score."""
    from auxiliary.rating_dev.validate_ratings import clear_logit_vec
    allr = np.array(list(rate.values()), float)
    grid = np.linspace(allr.min() - 2.0, allr.max() + 2.0, GRID_N)
    prior = -0.5 * ((grid - allr.mean()) / (allr.std() + 1e-9)) ** 2
    acc = {"all": [0.0, 0, []], "seen": [0.0, 0, []], "unseen": [0.0, 0, []]}
    for _u, seq in seqs.items():
        ll = prior.copy()
        v = np.zeros(k_axes) if k_axes else None
        for tid, y in seq:
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
            for key in ("all", "seen" if tid in seen_problems else "unseen"):
                acc[key][0] += loss
                acc[key][1] += 1
                acc[key][2].append(y)
            zg = np.clip(a * (grid - r) + kap + ax, -35, 35)
            pg = np.clip(c + span / (1 + np.exp(-zg)), 1e-9, 1 - 1e-9)
            ll += y * np.log(pg) + (1 - y) * np.log(1 - pg)
            if g is not None and v is not None:
                v += v_lr * (y - p) * g
    out = {}
    for key, (tot, n, ys) in acc.items():
        if not n:
            continue
        base = float(np.clip(np.mean(ys), 1e-9, 1 - 1e-9))
        const = -(base * np.log(base) + (1 - base) * np.log(1 - base))
        out[key] = {"logloss": tot / n, "n": n, "constant": const}
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
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
