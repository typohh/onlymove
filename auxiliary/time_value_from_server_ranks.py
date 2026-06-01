"""Rank value of *playing time* from cross-server declared ranks.

A go server's time control is a property of the SERVER, not the user: GoQuest is
exclusively blitz, EGF/AGA are slow over-the-board federation ranks, the rest are
mixed. A user who declares a rank on several servers is their own control, so a
within-user fixed-effects model on rank_surveys isolates each server's offset net
of the user's ability:

    declared_rank[u, s] = alpha_user[u] + gamma_server[s] + eps        (native rank scalar)

alpha_user absorbs ability/seriousness/selection (all the between-user confounds
that sank the time_used analysis); gamma_server is the server's ability-controlled
offset, triangulated across the whole user-server graph (so GoQuest connects to
EGF through shared OGS/Fox users even with no direct GoQuest+EGF declarer). The
fast-vs-slow contrast in gamma is the rank value of time -- a natural experiment
someone else ran for us.

CAVEAT (printed below too): gamma also picks up non-time server differences --
EGF/AGA are OTB and rank-DEFLATED (a 1d EGF is stronger than a 1d online), which
biases their gamma DOWN and thus works AGAINST finding a time benefit. So a
positive slow-minus-fast gap is conservative; a null is confounded.

    ./.venv/bin/python3 auxiliary/time_value_from_server_ranks.py
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "attempts", "database.sqlite")

SPEED = {"GoQuest": "fast",
         "EGF": "slow", "AGA": "slow",
         "OGS": "mixed", "KGS": "mixed", "Fox": "mixed",
         "Tygem": "mixed", "IGS": "mixed", "WBaduk": "mixed"}
REF = "OGS"          # reference server (hub, mixed speed) -> gamma_OGS == 0


def label_to_scalar(label):
    label = str(label).strip().lower()
    try:
        if label.endswith("k"):
            return -(int(label[:-1]) - 1)      # 1k=0, 5k=-4 (stronger = higher)
        if label.endswith("d"):
            return int(label[:-1])             # 1d=1, 3d=3
    except ValueError:
        return None
    return None


def fit_gamma(sv):
    """Two-way FE: native_rank = alpha_user + gamma_server, gamma relative to REF."""
    users = sorted(sv.user_id.unique())
    servers = [s for s in sorted(sv.server.unique()) if s != REF]
    uidx = {u: i for i, u in enumerate(users)}
    sidx = {s: j for j, s in enumerate(servers)}      # REF excluded -> its gamma=0

    nU, nS = len(users), len(servers)
    X = np.zeros((len(sv), nU + nS))
    for row, (_, r) in enumerate(sv.iterrows()):
        X[row, uidx[r.user_id]] = 1.0
        if r.server != REF:
            X[row, nU + sidx[r.server]] = 1.0
    y = sv.scalar.to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    gamma = {REF: 0.0}
    for s, j in sidx.items():
        gamma[s] = beta[nU + j]
    return gamma


def main():
    con = sqlite3.connect(DB)
    sv = pd.read_sql_query("SELECT user_id, server, rank FROM rank_surveys", con)
    con.close()
    sv["scalar"] = sv["rank"].map(label_to_scalar)
    sv = sv.dropna(subset=["scalar"])
    # one declaration per (user, server): latest wins (drop dup servers per user)
    sv = sv.drop_duplicates(subset=["user_id", "server"], keep="last")

    # only multi-server users identify the gamma contrasts
    multi = sv.groupby("user_id").server.nunique()
    multi = multi[multi >= 2].index
    fit = sv[sv.user_id.isin(multi)].copy()
    print(f"declarations: {len(sv)} over {sv.user_id.nunique()} users; "
          f"{len(multi)} users declared >=2 servers ({len(fit)} rows feed the FE fit)")

    gamma = fit_gamma(fit)

    def boot_gamma(n=2000):
        us = list(multi)
        out = {s: [] for s in gamma}
        rng = np.random.default_rng(0)
        for _ in range(n):
            samp = rng.choice(us, len(us), replace=True)
            rows = pd.concat([fit[fit.user_id == u].assign(user_id=f"{u}#{i}")
                              for i, u in enumerate(samp)])
            try:
                g = fit_gamma(rows)
            except Exception:
                continue
            for s in gamma:
                out[s].append(g.get(s, np.nan))
        return out
    boot = boot_gamma()

    print(f"\n=== server offsets gamma (rank steps, relative to {REF}=0) ===")
    print("higher gamma = users look STRONGER there after removing their ability\n")
    rows = []
    for s in sorted(gamma, key=lambda s: gamma[s]):
        b = np.array(boot[s], float)
        se = np.nanstd(b) if len(b) else np.nan
        n_decl = (fit.server == s).sum()
        rows.append((s, SPEED.get(s, "?"), gamma[s], se, n_decl))
    for s, sp, g, se, n in rows:
        print(f"  {s:8s} [{sp:5s}] gamma={g:+5.2f} ± {se:4.2f}   (n_decl={n})")

    # fast vs slow contrast, bootstrapped
    def grp_mean(g, cls):
        vals = [g[s] for s in g if SPEED.get(s) == cls and s in g]
        return np.mean(vals) if vals else np.nan
    slow_minus_fast = []
    for i in range(len(next(iter(boot.values())))):
        g = {s: boot[s][i] for s in boot if i < len(boot[s])}
        sf = grp_mean(g, "slow") - grp_mean(g, "fast")
        if np.isfinite(sf):
            slow_minus_fast.append(sf)
    smf = np.array(slow_minus_fast)
    point = grp_mean(gamma, "slow") - grp_mean(gamma, "fast")
    print(f"\n=== rank value of time (slow servers - fast server, within-user) ===")
    print(f"  slow(EGF,AGA) - fast(GoQuest) = {point:+.2f} rank steps "
          f"[95% CI {np.percentile(smf,2.5):+.2f}, {np.percentile(smf,97.5):+.2f}]  "
          f"(P(>0)={np.mean(smf>0):.2f})")
    print("  POSITIVE = stronger when given more time. Note OTB deflation biases the\n"
          "  slow servers' gamma DOWNWARD, so a positive estimate is a conservative floor.")

    # sanity: direct within-user pairwise diffs for the best-populated pairs
    print("\n=== sanity: raw within-user pairwise rank diffs (server_a - server_b) ===")
    for a, b in [("EGF", "Fox"), ("EGF", "OGS"), ("GoQuest", "OGS"), ("GoQuest", "Fox")]:
        pa = fit[fit.server == a].set_index("user_id").scalar
        pb = fit[fit.server == b].set_index("user_id").scalar
        common = pa.index.intersection(pb.index)
        if len(common) >= 3:
            d = (pa[common] - pb[common])
            print(f"  {a:8s} - {b:8s}: {d.mean():+.2f} (n={len(common)}, "
                  f"sd={d.std():.2f})")


if __name__ == "__main__":
    main()
