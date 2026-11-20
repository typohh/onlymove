"""Does KataGo-representation similarity of two problems predict ONE user's
outcomes on them beyond what the rating model already predicts?

    ./.venv/bin/python3 -m auxiliary.similarity_residuals [--slope-band 0.15 0.85]

For every user, every pair of their first-time attempts on same-size problems:

  similarity — cosine between the activation network's trunk state at the
               ANSWER cell (`correct_vecs_<size>.npy`, standardized per size),
               the `katago` score of `auxiliary.similar_problems`. The trunk is
               (near-)equivariant, so no orientation search.
  residual   — Pearson residual (y - p) / sqrt(p(1 - p)) against the RATING-ONLY
               4PL, p = c + (1 - c - lam) * sigmoid(a(theta - b)), from the
               shipped rating/slope/floor and user_theta. The axis term is left
               out (no per-user axis vectors are shipped), so whatever the axes
               would explain is still inside the residual.

The statistic per similarity bucket is the residual correlation over pairs —
mean(r_i r_j) / mean((r_i^2 + r_j^2)/2) — and its EXCESS over the same users'
all-pairs value, which absorbs the negative within-user baseline that an
in-sample theta induces (a user's residuals sum to ~0, so any two are slightly
anti-correlated). The excess is the quantity to read.

Only attempts on the SLOPE are used: sigmoid(a(theta - b)) inside --slope-band.
Below it the outcome is the floor (a guess), above it the lapse (a slip); in
both the outcome carries no information about this user on this problem.

Two confounds are split out rather than hoped away:
  * time gap — theta is one number over a whole history, so form and learning
    drift make temporally close attempts' residuals correlate, and similar
    problems are served close together (same shard, same rank band). Every
    bucket is reported per gap class: same session (<1h), 1h-7d, >7d.
  * problem misfit — two similar problems whose difficulty the trunk got wrong
    the SAME way make residuals correlate for every user, which is not a
    property of the user. The `centered` variant subtracts each problem's
    leave-one-out mean residual over OTHER users (problems with at least
    --min-others informative attempts by others; the rest drop out of it).

CIs are a user-clustered bootstrap: heavy users contribute O(n^2) pairs, so
pair-level standard errors would be far too narrow.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.features import load_first_time_attempts  # noqa: E402
from utils.rating.model_constants import clear_logit_vec  # noqa: E402

CACHE_DIR = os.path.join(PROJECT_ROOT, "activations", "_signal_cache")
RATINGS = os.path.join(PROJECT_ROOT, "ratings")
GAPS = [("<1h", 0.0, 1.0), ("1h-7d", 1.0, 24.0 * 7), (">7d", 24.0 * 7, np.inf)]
VARIANTS = ("raw", "centered")
# Fine similarity edges, set from random same-size pair quantiles at run time;
# these are the quantile levels. The top is resolved finely on purpose.
QLEVELS = [0, .25, .5, .75, .9, .95, .98, .99, .995, .998, .999, .9995, .9998,
           .9999, .99995, .99999, 1]


def load_problem_params(size, lam):
    r = pd.read_csv(os.path.join(RATINGS, str(size), "tsumego_ratings.csv"))
    a = pd.read_csv(os.path.join(RATINGS, str(size), "tsumego_slopes.csv"))
    c = pd.read_csv(os.path.join(RATINGS, str(size), "tsumego_floors.csv"))
    d = r.merge(a, on="tsumego_id").merge(c, on="tsumego_id")
    if not (len(d) == len(r) == len(a) == len(c)):
        raise SystemExit(f"size {size}: rating/slope/floor files disagree")
    d["b"] = d["rating"] - clear_logit_vec(d["floor"].to_numpy(), lam) / d["slope"]
    d["size"] = size
    return d


def load_vecs(size):
    ids = json.load(open(os.path.join(CACHE_DIR, f"correct_ids_{size}.json")))
    v = np.load(os.path.join(CACHE_DIR, f"correct_vecs_{size}.npy")).astype(np.float32)
    if len(v) != len(ids):
        raise SystemExit(f"correct_vecs_{size} / correct_ids_{size} disagree in length")
    v = (v - v.mean(0)) / (v.std(0) + 1e-6)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return {t: i for i, t in enumerate(ids)}, v


def random_pair_edges(vecs, rng, n=4_000_000):
    """Similarity bin edges at QLEVELS of RANDOM same-size problem pairs, pooled
    over sizes (equal weight) — so a bucket reads as 'top 0.1% of the corpus'."""
    sims = []
    for v in vecs.values():
        i, j = rng.integers(0, len(v), (2, n // len(vecs)))
        keep = i != j
        sims.append(np.einsum("ij,ij->i", v[i[keep]], v[j[keep]]))
    sims = np.concatenate(sims)
    e = np.quantile(sims, QLEVELS)
    e[0], e[-1] = -1.0001, 1.0001
    return e, sims


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--slope-band", type=float, nargs=2, default=(0.15, 0.85),
                    help="keep attempts with sigmoid(a(theta-b)) inside this band")
    ap.add_argument("--min-others", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the bucket table as CSV")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    lam = json.load(open(os.path.join(RATINGS, "rating_constants.json")))["fat_finger_rate"]
    params = pd.concat([load_problem_params(s, lam) for s in (9, 13)])
    theta = pd.read_csv(os.path.join(RATINGS, "user_theta.csv"))
    theta = dict(zip(theta["profile"].astype(str), theta["theta"]))
    vec_idx, vecs = {}, {}
    for s in (9, 13):
        vec_idx[s], vecs[s] = load_vecs(s)

    df = load_first_time_attempts()
    n0 = len(df)
    df["user_id"] = df["user_id"].astype(str)
    df = df[df["user_id"].isin(theta)]
    df = df.merge(params[["tsumego_id", "size", "b", "slope", "floor"]],
                  left_on=["problem", "size"], right_on=["tsumego_id", "size"])
    df["vi"] = [vec_idx[s].get(t, -1) for s, t in zip(df["size"], df["problem"])]
    df = df[df["vi"] >= 0].copy()
    print(f"{n0} first-time attempts -> {len(df)} by fitted users on rated, "
          f"embedded problems | lambda {lam}", flush=True)

    th = df["user_id"].map(theta).to_numpy()
    sg = 1.0 / (1.0 + np.exp(-df["slope"].to_numpy() * (th - df["b"].to_numpy())))
    c = df["floor"].to_numpy()
    p = np.clip(c + (1 - c - lam) * sg, 1e-6, 1 - 1e-6)
    y = df["success"].to_numpy(float)
    df["p"], df["s"] = p, sg
    df["r"] = (y - p) / np.sqrt(p * (1 - p))
    lo, hi = args.slope_band
    print(f"  model base rate {p.mean():.3f} vs observed {y.mean():.3f} | "
          f"on the slope [{lo},{hi}]: {np.mean((sg >= lo) & (sg <= hi)):.1%} of attempts "
          f"(below {np.mean(sg < lo):.1%}, above {np.mean(sg > hi):.1%})", flush=True)
    df = df[(sg >= lo) & (sg <= hi)].copy()

    # Leave-one-out problem mean over OTHER users' informative attempts.
    g = df.groupby(["size", "problem"])["r"]
    tot, cnt = g.transform("sum"), g.transform("count")
    others = cnt - 1
    df["rc"] = np.where(others >= args.min_others,
                        df["r"] - (tot - df["r"]) / others.clip(lower=1), np.nan)
    df["t"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601").astype("int64") / 3.6e12
    print(f"  {len(df)} informative attempts | {df['user_id'].nunique()} users | "
          f"centered variant keeps {df['rc'].notna().mean():.1%}", flush=True)

    edges, rand_sims = random_pair_edges(vecs, rng)
    nb = len(edges) - 1
    users = sorted(df["user_id"].unique())
    # acc[user, variant, gap, bin, stat]; stats: n, sum r_i r_j, sum (r_i^2+r_j^2)/2
    acc = np.zeros((len(users), len(VARIANTS), len(GAPS), nb, 3))
    gap_edges = np.array([g[1] for g in GAPS] + [np.inf])
    n_pairs = 0
    for ui, (uid, d) in enumerate(df.groupby("user_id", sort=True)):
        assert uid == users[ui]
        for s, ds in d.groupby("size"):
            if len(ds) < 2:
                continue
            V = vecs[s][ds["vi"].to_numpy()]
            t = ds["t"].to_numpy()
            rs = [ds[col].to_numpy() for col in ("r", "rc")]
            n = len(ds)
            m = len(GAPS) * nb
            # Row blocks of the similarity matrix (upper triangle only), so the
            # heaviest user's ~10^8 pairs never sit in memory at once.
            for k in range(0, n - 1, 1024):
                i0 = np.arange(k, min(k + 1024, n - 1))
                S = V[i0] @ V[k + 1:].T                       # [blk, n-k-1]
                jj = np.arange(k + 1, n)
                upper = jj[None, :] > i0[:, None]
                bi, bj = np.nonzero(upper)
                i, j = i0[bi], jj[bj]
                sim = S[bi, bj]
                b = np.searchsorted(edges, sim, side="right") - 1
                gp = np.searchsorted(gap_edges, np.abs(t[i] - t[j]), side="right") - 1
                n_pairs += len(i)
                for vi, r in enumerate(rs):
                    ri, rj = r[i], r[j]
                    ok = ~(np.isnan(ri) | np.isnan(rj))
                    cell = gp[ok] * nb + b[ok]
                    ri, rj = ri[ok], rj[ok]
                    acc[ui, vi, :, :, 0] += np.bincount(cell, minlength=m).reshape(len(GAPS), nb)
                    acc[ui, vi, :, :, 1] += np.bincount(
                        cell, ri * rj, minlength=m).reshape(len(GAPS), nb)
                    acc[ui, vi, :, :, 2] += np.bincount(
                        cell, (ri ** 2 + rj ** 2) / 2, minlength=m).reshape(len(GAPS), nb)
        if ui % 500 == 0:
            print(f"  user {ui}/{len(users)} | {n_pairs / 1e6:.0f}M pairs", flush=True)
    print(f"  {n_pairs / 1e6:.1f}M pairs total", flush=True)

    # User-clustered bootstrap: resample users with replacement.
    W = rng.multinomial(len(users), np.full(len(users), 1 / len(users)), size=args.boot)

    def corr(a):  # a[..., bins, 3] -> correlation per bin
        return a[..., 1] / np.maximum(a[..., 2], 1e-12)

    # Report buckets: the fine quantile bins, plus each gap class and all gaps.
    rows = []
    for vi, vname in enumerate(VARIANTS):
        for gname, gsel in [(n, [k]) for k, (n, _, _) in enumerate(GAPS)] + [("all", list(range(len(GAPS))))]:
            A = acc[:, vi, gsel].sum(1)               # [users, bins, 3]
            base = A.sum(1)                           # [users, 3] all pairs
            pt, pt_base = corr(A.sum(0)), corr(base.sum(0)[None])[0]
            Ab = np.einsum("bu,uks->bks", W, A)       # [boot, bins, 3]
            bb = np.einsum("bu,us->bs", W, base)
            ex_b = corr(Ab) - corr(bb)[:, None]
            # Users with at least one pair in the bin: how many people stand behind it.
            nu = (A[..., 0] > 0).sum(0)
            for k in range(nb):
                rows.append(dict(variant=vname, gap=gname,
                                 q_lo=QLEVELS[k], q_hi=QLEVELS[k + 1],
                                 cos_lo=edges[k], cos_hi=edges[k + 1],
                                 pairs=int(A[:, k, 0].sum()), users=int(nu[k]),
                                 rho=pt[k], excess=pt[k] - pt_base,
                                 ex_lo=np.nanpercentile(ex_b[:, k], 2.5),
                                 ex_hi=np.nanpercentile(ex_b[:, k], 97.5)))
    out = pd.DataFrame(rows)
    if args.out:
        out.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    for vname in VARIANTS:
        for gname in ["all"] + [g[0] for g in GAPS]:
            t = out[(out.variant == vname) & (out.gap == gname)]
            print(f"\n=== {vname} residuals, gap {gname} "
                  f"(corpus pair quantile of similarity; excess = corr - this gap's all-pairs corr) ===")
            print(f"{'quantile':>17} {'cosine':>15} {'pairs':>11} {'users':>6} "
                  f"{'rho':>8} {'excess':>8} {'95% CI':>19}")
            for _, r in t.iterrows():
                print(f"{r.q_lo:>8.5f}-{r.q_hi:<8.5f} {max(r.cos_lo, -1):>7.3f}-{min(r.cos_hi, 1):<7.3f} "
                      f"{r.pairs:>11} {r.users:>6} {r.rho:>8.4f} {r.excess:>+8.4f} "
                      f"[{r.ex_lo:+.4f},{r.ex_hi:+.4f}]")


if __name__ == "__main__":
    main()
