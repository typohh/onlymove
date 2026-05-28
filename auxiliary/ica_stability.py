"""ICASSO-style stability diagnostic for the joint skew-ICA embedding.

Answers "how many independent components actually exist in the data?" — a
question the production fit punts on by hard-coding K=64 (build_embedding.py
runs FastICA with whiten=False *inside* the K-dim PCA subspace, so the number
of ICs is forced to equal K; ICA only rotates, it never selects dimension).

Two things are measured here:

1.  ICASSO stability per component (Himberg et al. 2004, "Validating the
    independent components of neuroimaging time series via clustering and
    visualization"). FastICA is re-run many times on bootstrap resamples with
    different seeds; the estimated components are pooled and clustered; each
    cluster gets an Iq stability index in [0, 1] (tight, reproducible cluster
    -> Iq near 1; algorithm fitting noise differently each run -> Iq near 0).
    The count of clusters with Iq above a threshold is the defensible answer
    to "how many reproducible ICs exist".

2.  A model-order sweep: the same stability analysis at several extraction
    dimensions m, so you can see where reproducibility rolls off as you ask
    for more components.

Why this is cheap (no activation reads). The production fit's PCA basis is
fixed and orthonormal, and the final embeddings E live in per_problem.csv
while the unmixing W and per-axis stds live in pca.npz. We invert
    E = (S @ W.T) / ica_stds
to recover S, the PCA-whitened scores that were FastICA's actual input, then
re-run ICA on S (and on its leading-m columns for the order sweep). Because S
columns ARE the fixed top-K PCA directions, every run's components share one
coordinate system, so cross-run comparison is a plain cosine — none of the
cross-run basis-alignment ICASSO usually needs.

A round-trip self-check (reproject recovered S through the saved W/ica_stds and
compare to E) guards the inversion; the script aborts if it doesn't match.

Run:
    ./.venv/bin/python3 auxiliary/ica_stability.py
    ./.venv/bin/python3 auxiliary/ica_stability.py --orders 8,16,32,64 --n-runs 30
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOINT_DIR = os.path.join(PROJECT_ROOT, "embeddings", "joint")


def _skew_contrast(x):
    """FastICA 3rd-cumulant contrast, identical to build_embedding._skew_contrast.

    G(u)=u^3/3, g(u)=u^2, g'(u)=2u. sklearn calls fun(x, **fun_args) with an
    empty fun_args, so the single-arg signature is fine.
    """
    return x ** 2, (2.0 * x).mean(axis=-1)


def recover_whitened_scores(pca_arc, embeddings):
    """Invert E = (S @ W.T) / ica_stds to get the PCA-whitened FastICA input S.

    Returns (S, roundtrip_max_abs_err). S has shape (N, K).
    """
    W = pca_arc["ica_unmixing"].astype(np.float64)        # (K, K)
    ica_stds = pca_arc["ica_stds"].astype(np.float64)     # (K,)
    E = np.asarray(embeddings, dtype=np.float64)          # (N, K)
    A = E * ica_stds                                      # = S @ W.T
    # S @ W.T = A  ->  W @ S.T = A.T  ->  S.T = solve(W, A.T)
    S = np.linalg.solve(W, A.T).T
    # Round-trip: reproject and compare to the embeddings we started from.
    E_check = (S @ W.T) / ica_stds
    err = float(np.max(np.abs(E_check - E)))
    return S, err


def run_icasso_at_order(S, m, n_runs, sample_size, contrast, max_iter, tol,
                        base_seed, rng):
    """Re-run FastICA n_runs times on bootstrap resamples of S[:, :m].

    Returns the pooled, L2-normalized unmixing rows (n_runs*m, m). Runs that
    fail to converge / error are skipped (their absence just shrinks the pool).
    """
    Sm = S[:, :m]
    N = Sm.shape[0]
    fun = _skew_contrast if contrast == "skew" else contrast
    rows = []
    n_ok = 0
    for r in range(n_runs):
        idx = rng.integers(0, N, size=min(sample_size, N))
        Xb = Sm[idx]
        try:
            ica = FastICA(
                whiten=False, fun=fun,
                max_iter=max_iter, tol=tol,
                random_state=base_seed + r,
            )
            ica.fit(Xb)
        except Exception:
            continue
        W = ica.components_                       # (m, m), rows in PCA basis
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        rows.append(W / norms)
        n_ok += 1
    if not rows:
        return None, 0
    return np.vstack(rows), n_ok


def cluster_and_iq(vecs, n_clusters):
    """Cluster pooled component vectors and compute the Iq stability index.

    Distance is 1 - |cos| (ICA sign ambiguity -> absolute similarity).
    Iq_k = mean|sim| within cluster k  -  mean|sim| from k to everything else
    (Himberg et al. 2004). Returns Iq values sorted descending, plus the
    per-cluster centroid directions (mean of L2-normalized members, renormalized)
    for optional matching against the production axes.
    """
    sim = np.abs(vecs @ vecs.T)                  # (P, P) absolute cosine
    np.fill_diagonal(sim, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    iqs = []
    centroids = []
    for c in np.unique(labels):
        inside = np.where(labels == c)[0]
        outside = np.where(labels != c)[0]
        intra = sim[np.ix_(inside, inside)].mean()
        inter = sim[np.ix_(inside, outside)].mean() if len(outside) else 0.0
        iqs.append(intra - inter)
        v = vecs[inside].mean(axis=0)
        n = np.linalg.norm(v)
        centroids.append(v / n if n > 1e-12 else v)
    iqs = np.array(iqs)
    centroids = np.array(centroids)
    order = np.argsort(-iqs)
    return iqs[order], centroids[order]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orders", default="8,16,24,32,48,64",
                   help="Comma-separated extraction dims m for the sweep "
                        "(the largest should equal K to match production).")
    p.add_argument("--n-runs", type=int, default=25,
                   help="FastICA re-runs (bootstrap + seed) per order.")
    p.add_argument("--sample-size", type=int, default=20000,
                   help="Bootstrap resample size per run (capped at N).")
    p.add_argument("--contrast", default="skew",
                   choices=["skew", "logcosh", "exp", "cube"],
                   help="FastICA contrast; 'skew' matches production.")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iq-thresholds", default="0.9,0.8,0.7",
                   help="Stability cutoffs reported as IC counts.")
    p.add_argument("--out", default=os.path.join(JOINT_DIR, "ica_stability.json"))
    args = p.parse_args()

    pca_path = os.path.join(JOINT_DIR, "pca.npz")
    csv_path = os.path.join(JOINT_DIR, "per_problem.csv")
    if not os.path.exists(pca_path) or not os.path.exists(csv_path):
        raise SystemExit(f"need {pca_path} and {csv_path}; run build_embedding first")

    pca_arc = np.load(pca_path, allow_pickle=False)
    if str(pca_arc["rotation"]) != "ica" or "ica_unmixing" not in pca_arc.files:
        raise SystemExit("pca.npz has no ICA rotation; nothing to test")

    K = int(pca_arc["ica_unmixing"].shape[0])
    thresholds = [float(t) for t in args.iq_thresholds.split(",")]
    orders = sorted({int(o) for o in args.orders.split(",")} | set())
    orders = [m for m in orders if 1 <= m <= K]

    print(f"Loading embeddings from {csv_path} ...", flush=True)
    embed_cols = [f"embed_{i}" for i in range(K)]
    df = pd.read_csv(csv_path, usecols=embed_cols)
    E = df[embed_cols].to_numpy(dtype=np.float64)
    N = E.shape[0]
    print(f"  {N} problems x K={K}", flush=True)

    print("Recovering PCA-whitened FastICA input S by inverting the saved unmixing...",
          flush=True)
    S, err = recover_whitened_scores(pca_arc, E)
    print(f"  round-trip max abs err on E: {err:.3e}", flush=True)
    if err > 1e-3:
        raise SystemExit(
            f"inversion round-trip error {err:.3e} too large; the recovered S "
            "would not match production. Aborting rather than analyzing garbage."
        )
    col_std = S.std(axis=0)
    print(f"  recovered S column std range: {col_std.min():.3f}..{col_std.max():.3f} "
          f"(should be ~1 for whitened PCA scores)", flush=True)

    rng = np.random.default_rng(args.seed)
    print(f"\nICASSO: contrast={args.contrast!r}  n_runs={args.n_runs}  "
          f"sample_size={min(args.sample_size, N)}  orders={orders}", flush=True)

    sweep = []
    iq_at_K = None
    centroids_at_K = None
    for m in orders:
        t0 = time.time()
        vecs, n_ok = run_icasso_at_order(
            S, m, args.n_runs, args.sample_size, args.contrast,
            args.max_iter, args.tol, args.seed * 1000 + m, rng,
        )
        if vecs is None:
            print(f"  m={m:>3d}: all runs failed; skipping", flush=True)
            continue
        iqs, centroids = cluster_and_iq(vecs, m)
        counts = {f"Iq>={t}": int((iqs >= t).sum()) for t in thresholds}
        sweep.append({
            "order": m, "runs_ok": n_ok,
            "iq_mean": float(iqs.mean()),
            "iq_min": float(iqs.min()),
            "stable_counts": counts,
            "iq_sorted": [round(float(x), 4) for x in iqs],
        })
        counts_str = "  ".join(f"{k}:{v}" for k, v in counts.items())
        print(f"  m={m:>3d}: {n_ok} runs ok  mean Iq={iqs.mean():.3f}  "
              f"[{counts_str}]  ({time.time()-t0:.0f}s)", flush=True)
        if m == K:
            iq_at_K = iqs
            centroids_at_K = centroids

    # At the production order, line up each stable cluster against the saved
    # ICA axes (same K-dim PCA basis) so Iq can be read next to the existing
    # skewness ranking the labels/recommender already key on.
    skew_table = None
    if iq_at_K is not None and "ica_skewness" in pca_arc.files:
        prod_W = pca_arc["ica_unmixing"].astype(np.float64)
        prod_norm = prod_W / np.linalg.norm(prod_W, axis=1, keepdims=True)
        skew = pca_arc["ica_skewness"].astype(np.float64)
        # Greedy 1-1 match: each production axis -> best unused cluster by |cos|.
        simmat = np.abs(prod_norm @ centroids_at_K.T)   # (K, n_clusters)
        used = set()
        rows = []
        for ax in range(K):
            cand = np.argsort(-simmat[ax])
            pick = next((c for c in cand if c not in used), int(cand[0]))
            used.add(pick)
            rows.append({
                "prod_axis": ax,
                "skewness": round(float(skew[ax]), 3),
                "matched_iq": round(float(iq_at_K[pick]), 4),
                "match_cos": round(float(simmat[ax, pick]), 4),
            })
        skew_table = rows
        print("\nProduction axes by skewness vs ICASSO stability "
              "(first 20, then last 10):")
        print(f"  {'axis':>4s} {'skew':>8s} {'Iq':>7s} {'match_cos':>9s}")
        for row in rows[:20] + ([{"sep": True}] if K > 30 else []) + rows[-10:]:
            if row.get("sep"):
                print("  ...")
                continue
            print(f"  {row['prod_axis']:>4d} {row['skewness']:>8.3f} "
                  f"{row['matched_iq']:>7.3f} {row['match_cos']:>9.3f}")

    result = {
        "params": {
            "orders": orders, "n_runs": args.n_runs,
            "sample_size": min(args.sample_size, N), "contrast": args.contrast,
            "max_iter": args.max_iter, "tol": args.tol, "seed": args.seed,
            "N": N, "K": K, "roundtrip_err": err,
        },
        "sweep": sweep,
        "skew_vs_iq_at_K": skew_table,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")

    # Headline read.
    if iq_at_K is not None:
        print("\n=== headline (at production K) ===")
        for t in thresholds:
            print(f"  reproducible ICs with Iq >= {t}: "
                  f"{int((iq_at_K >= t).sum())} / {K}")
        print("  (compare with the order sweep above: where mean Iq starts "
              "dropping is the effective model order)")


if __name__ == "__main__":
    main()
