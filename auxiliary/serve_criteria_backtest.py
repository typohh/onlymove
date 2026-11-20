"""Which serve criterion finds the problems a user UNDERPERFORMS on: the
shipped motif categories, or KataGo-representation similarity to the user's
own earlier failures?

    ./.venv/bin/python3 -m auxiliary.similarity_residuals   # the pairwise view
    ./.venv/bin/python3 -m auxiliary.serve_criteria_backtest [--cache DIR]

Both criteria are scored on the attempt log, OUT OF SAMPLE in time: attempt t is
scored with what the client could know before t.

  motif       the categories `flaggedMotifCategories` would hold before t — the
              shipped readings run unmodified under node over rows[0, t)
              (`serve_criteria_backtest.mjs`). Each (attempt, category) is
              scored by the attempt's correct-move loading in the category's
              serve direction, sign * emb[k], which is what the motif slot
              argmaxes.
  similarity  the maximum cosine (trunk state at the answer cell, as in
              `similar_problems`) between attempt t's problem and any problem
              of the same size the user FAILED before t.

Scores are put on ONE scale, the score's quantile among corpus problems:
motif against every published problem of that size on that (axis, direction),
similarity against a fixed random sample of same-size problems scored against
the same failure set at the same t. "Top 0.1%" then means the same selectivity
for both, and the realistic operating point of each is a statement about pool
size: the motif slot argmaxes the nearest 50 (~top 2%), and a similarity search
over a whole rank band could reach 1-in-thousands.

The outcome is y - p under the RATING-ONLY 4PL from the shipped triple and the
backend theta, restricted to attempts on the slope (sigmoid(a(theta-b)) in
[0.15, 0.85]), in percentage points. Negative = the user underperformed.
Theta is in-sample, which shrinks both criteria's effects alike. CIs are a
user-clustered bootstrap.
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations import motif_findings_figures as mf  # noqa: E402
from auxiliary.similarity_residuals import load_vecs  # noqa: E402
from utils.constants import read_ignored_users  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.model_constants import clear_logit_vec  # noqa: E402

HELPER = os.path.join(PROJECT_ROOT, "auxiliary", "serve_criteria_backtest.mjs")
RATINGS = os.path.join(PROJECT_ROOT, "ratings")
QB = [0, .5, .9, .98, .995, .999, .9999, 1.0000001]
KIND_ORDER = ("weak", "underplay", "overplay", "strong")


def history_ids(pub):
    """The problem id of every row `mf.load_histories` builds, in its order —
    the same query and the same filters, so the two line up row for row."""
    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        cur = con.execute(
            "SELECT user_id, problem, created_at FROM tsumego_reports "
            " WHERE is_repeat = 0 ORDER BY user_id, created_at, id")
        by_user = {}
        for uid, tid, ts in cur:
            if str(tid) in pub["pos"]:
                by_user.setdefault(uid, []).append((str(tid), ts))
    finally:
        con.close()
    return {u: r for u, r in by_user.items() if len(r) >= mf.MIN_HISTORY}


def motif_flags(pub, histories, cache):
    path = os.path.join(cache, "motif_flags.json")
    if os.path.exists(path):
        print(f"  motif flags from {path}", flush=True)
        return json.load(open(path))
    meta = json.load(open(os.path.join(mf.TSUMEGOS, "embedding_meta.json")))
    pl = os.path.join(cache, "motif_payload.json")
    with open(pl, "w") as f:
        json.dump({"pwa_root": mf.PWA_ROOT + os.sep, "K": pub["K"],
                   "labels": meta["labels"]["components"],
                   "users": [{"id": u, "rows": r} for u, r in histories.items()]}, f)
    print("  running the shipped motif readings at every position (node) ...", flush=True)
    subprocess.run(["node", HELPER, pl, path], cwd=PROJECT_ROOT, check=True)
    return json.load(open(path))


def boot_ci(values, users, rng, n=1000):
    """Mean with a user-clustered bootstrap 95% CI."""
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    codes, inv = np.unique(users, return_inverse=True)
    s = np.bincount(inv, values, minlength=len(codes))
    c = np.bincount(inv, minlength=len(codes)).astype(float)
    W = rng.multinomial(len(codes), np.full(len(codes), 1 / len(codes)), size=n)
    b = (W @ s) / np.maximum(W @ c, 1)
    return values.mean(), *np.percentile(b, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True, help="directory for the node payload and flags")
    ap.add_argument("--slope-band", type=float, nargs=2, default=(0.15, 0.85))
    ap.add_argument("--ref", type=int, default=4000,
                    help="random corpus problems per size the similarity quantile is taken against")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--since", default=None,
                    help="score only attempts at or after this ISO date (earlier history "
                         "still feeds the flags and the failure set). Served problems were "
                         "chosen under the ratings of their day, so an old attempt's residual "
                         "against today's ratings describes a serve the engine no longer makes")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pub = mf.load_published()
    histories = mf.load_histories(pub)
    ids = history_ids(pub)
    if set(ids) != set(histories) or any(len(ids[u]) != len(histories[u]) for u in ids):
        raise SystemExit("history ids do not line up with mf.load_histories")
    flags = motif_flags(pub, histories, args.cache)

    lam = json.load(open(os.path.join(RATINGS, "rating_constants.json")))["fat_finger_rate"]
    theta = pd.read_csv(os.path.join(RATINGS, "user_theta.csv"))
    theta = dict(zip(theta["profile"].astype(str), theta["theta"]))
    ignored = set(read_ignored_users())
    K = pub["K"]

    # Corpus quantile tables for the motif score, per size and axis.
    size_of = np.array([s for _, _, s in pub["rows"]])
    has_vec = np.any(pub["emb"] != 0, axis=1)
    motif_sorted = {(s, k): np.sort(pub["emb"][(size_of == s) & has_vec, k])
                    for s in (9, 13) for k in range(K)}

    def motif_q(s, k, sign, x):
        arr = motif_sorted[(s, k)]
        if sign > 0:
            return np.searchsorted(arr, x, side="left") / len(arr)
        return 1.0 - np.searchsorted(arr, x, side="right") / len(arr)

    vec_idx, vecs, refs = {}, {}, {}
    for s in (9, 13):
        vec_idx[s], vecs[s] = load_vecs(s)
        refs[s] = vecs[s][rng.choice(len(vecs[s]), args.ref, replace=False)]

    rows_all, motif_rows = [], []
    lo, hi = args.slope_band
    n_users = 0
    for ui, uid in enumerate(histories):
        if uid in ignored or uid not in theta:
            continue
        n_users += 1
        th = theta[uid]
        hist, tids, fl = histories[uid], ids[uid], flags[uid]
        n = len(hist)
        size = np.array([h[0] for h in hist])
        y = np.array([h[1] for h in hist], float)
        r_, a_, c_ = (np.array([h[i] for h in hist], float) for i in (2, 3, 4))
        b_ = r_ - clear_logit_vec(c_, lam) / a_
        sg = 1 / (1 + np.exp(-a_ * (th - b_)))
        p = c_ + (1 - c_ - lam) * sg
        seen, first = set(), np.zeros(n, bool)
        for t, tid in enumerate(tids):
            first[t] = tid[0] not in seen
            seen.add(tid[0])
        use = first & (sg >= lo) & (sg <= hi)
        if args.since:
            use &= np.array([ts >= args.since for _, ts in tids])

        # Similarity to the user's earlier failures, and its corpus quantile.
        sim_q = np.full(n, np.nan)
        sim_v = np.full(n, np.nan)
        for s in (9, 13):
            sel = np.where(size == s)[0]
            vi = np.array([vec_idx[s].get(tids[t][0], -1) for t in sel])
            ok = vi >= 0
            sel, vi = sel[ok], vi[ok]
            if len(sel) < 2:
                continue
            V = vecs[s][vi]
            fail = np.where(y[sel] == 0)[0]          # positions within sel
            if len(fail) == 0:
                continue
            Vf = V[fail]
            # Number of failures strictly before each position.
            nf_before = np.searchsorted(fail, np.arange(len(sel)), side="left")
            # Reference: cumulative max over failures of each ref problem's sim.
            Rc = np.maximum.accumulate(refs[s] @ Vf.T, axis=1)   # [ref, F]
            for k0 in range(0, len(sel), 2048):
                blk = np.arange(k0, min(k0 + 2048, len(sel)))
                S = V[blk] @ Vf.T                                  # [blk, F]
                S[np.arange(len(fail))[None, :] >= nf_before[blk][:, None]] = -np.inf
                best = S.max(1)
                for j, bi in enumerate(blk):
                    nb = nf_before[bi]
                    if nb == 0:
                        continue
                    sim_v[sel[bi]] = best[j]
                    sim_q[sel[bi]] = np.mean(Rc[:, nb - 1] < best[j])

        for t in np.where(use)[0]:
            rows_all.append((uid, size[t], y[t] - p[t], sim_q[t], sim_v[t], len(fl[t]) > 0))
            emb = pub["emb"][pub["pos"][tids[t][0]]]
            if not np.any(emb):
                continue
            for k, sign, kind in fl[t]:
                motif_rows.append((uid, kind, k, y[t] - p[t],
                                   motif_q(size[t], k, sign, emb[k])))
        if ui % 200 == 0:
            print(f"  user {ui}/{len(histories)}", flush=True)

    A = pd.DataFrame(rows_all, columns=["user", "size", "d", "sim_q", "sim", "flagged"])
    M = pd.DataFrame(motif_rows, columns=["user", "kind", "k", "d", "q"])
    A["d"] *= 100
    M["d"] *= 100
    base, blo, bhi = boot_ci(A["d"].to_numpy(), A["user"].to_numpy(), rng)
    print(f"\n{n_users} users | {len(A)} on-slope first-time attempts | "
          f"mean y-p {base:+.2f}pt [{blo:+.2f},{bhi:+.2f}] | "
          f"{A['flagged'].mean():.1%} had >=1 motif flag | "
          f"{A['sim_q'].notna().mean():.1%} had a prior same-size failure")

    def table(name, df, qcol):
        print(f"\n=== {name}: mean (y - p) in points by corpus quantile of the score ===")
        print(f"{'quantile':>17} {'rows':>8} {'users':>6} {'y-p':>7} {'95% CI':>17} {'vs all':>7}")
        for q0, q1 in zip(QB[:-1], QB[1:]):
            d = df[(df[qcol] >= q0) & (df[qcol] < q1)]
            m, l, h = boot_ci(d["d"].to_numpy(), d["user"].to_numpy(), rng)
            print(f"{q0:>8.4f}-{min(q1, 1):<8.4f} {len(d):>8} {d['user'].nunique():>6} "
                  f"{m:>+7.2f} [{l:+6.2f},{h:+6.2f}] {m - base:>+7.2f}")

    table("similarity to an earlier failure", A[A["sim_q"].notna()], "sim_q")
    table("motif, every flagged category", M, "q")
    for kind in KIND_ORDER:
        table(f"motif, {kind}", M[M["kind"] == kind], "q")
    print("\nmotif flags by kind (attempt-category rows):",
          M["kind"].value_counts().to_dict())


if __name__ == "__main__":
    main()
